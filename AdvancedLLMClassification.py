import sys
import numpy as np
import pandas as pd
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
import os
import time
import zipfile
import urllib.request
from pathlib import Path
from tqdm.auto import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, recall_score
import lightning as L
import dataloader
import Utils
from ClassificationNetLoRA import ClassificationNetLoRA

# ---------------------------------------------------------------------------
# CONFIG — updated for Symptom2Disease dataset
# ---------------------------------------------------------------------------
MODEL_CHECKPOINTS = {
    "danube": "h2oai/h2o-danube-1.8b-chat",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
}
MODEL_NAME = MODEL_CHECKPOINTS["danube"]

CSV_PATH = "/content/Symptom2Disease.csv"   # <-- apna actual path yahan daalo

DO_TEST = False                     # False = training, True = testing
BATCH_SIZE = 4                      # 1 -> 4, texts short hain, memory allow karta hai
CLASSIFICATION_TYPE = 'MULTI_CLASS'  # 'MULTI_LABEL' -> 'MULTI_CLASS' (single-label dataset)
GRADIENT_ACCUMULATION_STEPS = 4      # effective batch size = 16
num_epochs = 10                      # 5 -> 10, chhota dataset, thoda zyada training helps
seed = 252
APPLY_LORA = True
MAX_SEQ_LENGTH = 128                 # 768 -> 128, symptom texts chhote hain

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tqdm.pandas()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def calc_accuracy(dataloader, model, tokenizer, type):  # for binary or multi-class
    with torch.no_grad():
        model.eval()
        pred_scores = []
        actual_scores = []
        max_token_len = 0
        wrong_count = 0
        for batch in tqdm(dataloader, total=len(dataloader), desc=f'Calc {type} accuracy'):
            prompt, targets = batch
            encodings = Utils.tokenize_text(prompt, tokenizer)
            input_ids = encodings['input_ids'].to(device)
            if input_ids.shape[1] > max_token_len:
                max_token_len = input_ids.shape[1]
            attention_mask = encodings['attention_mask'].to(device)
            if input_ids.shape[1] >= MAX_SEQ_LENGTH:
                input_ids = input_ids[:, 0:MAX_SEQ_LENGTH - 1]
            if attention_mask.shape[1] >= MAX_SEQ_LENGTH:
                attention_mask = attention_mask[:, 0:MAX_SEQ_LENGTH - 1]
            with autocast():
                logits = model(input_ids, attention_mask)
            pred_score = F.softmax(logits, dim=-1).argmax(dim=-1).cpu().detach().numpy().tolist()
            pred_scores.extend(pred_score)
            actual_scores.extend(targets.numpy().tolist())
            if (pred_score[0] != targets[0]):
                wrong_count = wrong_count + 1
        pred_scores = np.array(pred_scores)
        accuracy = accuracy_score(actual_scores, pred_scores)
        return accuracy


def calc_accuracy_multi_label(dataloader, model, tokenizer, type):  # for multi-label (not used here)
    with torch.no_grad():
        model.eval()
        labels = []
        predictions = []
        for batch in tqdm(dataloader, total=len(dataloader), desc=f'Calc {type} accuracy'):
            prompt, targets = batch
            encodings = Utils.tokenize_text(prompt, tokenizer)
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            if input_ids.shape[1] >= MAX_SEQ_LENGTH:
                input_ids = input_ids[:, 0:MAX_SEQ_LENGTH - 1]
            if attention_mask.shape[1] >= MAX_SEQ_LENGTH:
                attention_mask = attention_mask[:, 0:MAX_SEQ_LENGTH - 1]
            with autocast():
                logits = model(input_ids, attention_mask)
            preds = torch.sigmoid(logits).detach().cpu()
            predictions.extend(preds)
            for out_labels in targets.detach().cpu():
                labels.append(out_labels)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        y_preds = predictions.numpy()
        y_true = labels.numpy()
        y_pred_labels = np.where(y_preds > 0.5, 1, 0).tolist()
        accuracy = f1_score(y_true, y_pred_labels, average='micro')
        return accuracy


def main():
    model_name = MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # ------------prepare data loaders--------------------------
    text_set, labels_dict, num_labels, label_encoder = dataloader.prepare_symptom_data(
        csv_path=CSV_PATH
    )

    tx_train = [tokenizer.bos_token + x for x in text_set["train"]]
    tx_test = [tokenizer.bos_token + x for x in text_set["test"]]
    tx_val = [tokenizer.bos_token + x for x in text_set["dev"]]

    labels_train = labels_dict['train']
    labels_test = labels_dict['test']
    labels_val = labels_dict['dev']

    train_dataloader, test_dataloader, val_dataloader = Utils.get_train_test_val_Loaders(
        tx_train, tx_test, tx_val, labels_train, labels_test, labels_val, BATCH_SIZE
    )

    learning_rate = 0.0002
    diff_lr = 0.00001
    warmup_steps = 0
    weight_decay = 0.01

    L.seed_everything(seed=seed)

    # create model — NUM_CLASSES passed dynamically from the dataset (24 for Symptom2Disease)
    model = ClassificationNetLoRA(MODEL_NAME, DO_TEST, APPLY_LORA, NUM_CLASSES=num_labels)
    model.to(device)

    if DO_TEST == True:
        if CLASSIFICATION_TYPE == 'MULTI_LABEL':
            test_acc = calc_accuracy_multi_label(test_dataloader, model, tokenizer, type='test')
        else:
            test_acc = calc_accuracy(test_dataloader, model, tokenizer, type='test')
        print('Test accuracy:', test_acc)
        return

    print('Here are the trainable parameters:')
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)

    optimizer = Utils.get_optimizer(
        model,
        learning_rate=learning_rate,
        diff_lr=diff_lr,
        weight_decay=weight_decay
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_epochs * len(train_dataloader)
    )

    scaler = GradScaler()
    optimizer.zero_grad()
    start_time = time.time()

    for epoch in range(num_epochs):
        max_token_len = 0
        for batch_idx, batch in enumerate(train_dataloader):
            model.train()
            prompt, targets = batch
            encodings = Utils.tokenize_text(prompt, tokenizer)
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            if input_ids.shape[1] > max_token_len:
                max_token_len = input_ids.shape[1]

            if input_ids.shape[1] >= MAX_SEQ_LENGTH:
                input_ids = input_ids[:, 0:MAX_SEQ_LENGTH - 1]
            if attention_mask.shape[1] >= MAX_SEQ_LENGTH:
                attention_mask = attention_mask[:, 0:MAX_SEQ_LENGTH - 1]

            targets = targets.to(device)

            with autocast():
                logits = model(input_ids, attention_mask)
                if CLASSIFICATION_TYPE == 'MULTI_LABEL':
                    loss = F.binary_cross_entropy_with_logits(logits, targets.float())
                else:
                    loss = F.cross_entropy(logits, targets)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            scaler.scale(loss).backward()

            if ((batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0) or ((batch_idx + 1) == len(train_dataloader)):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if (batch_idx % 100) == 0:
                print(
                    f'Epoch: {epoch+1} / {num_epochs}'
                    f'| Batch: {batch_idx+1}/{len(train_dataloader)}'
                    f'| Loss: {loss.item():.4f}'
                )

            if ((batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0) or ((batch_idx + 1) == len(train_dataloader)):
                loss.detach()
                attention_mask.detach()
                del attention_mask
                del encodings
                del loss
                del logits
                del prompt
                del targets

        if CLASSIFICATION_TYPE == 'MULTI_LABEL':
            test_acc = calc_accuracy_multi_label(test_dataloader, model, tokenizer, type='test')
        else:
            test_acc = calc_accuracy(test_dataloader, model, tokenizer, type='test')
        print('Epoc=', epoch, ' Test accuracy:', test_acc)
        print(max_token_len)
        model.save_peft_adapter()

    if CLASSIFICATION_TYPE == 'MULTI_LABEL':
        test_acc = calc_accuracy_multi_label(test_dataloader, model, tokenizer, type='test')
    else:
        test_acc = calc_accuracy(test_dataloader, model, tokenizer, type='test')
    print('Final Test accuracy:', test_acc)


if __name__ == "__main__":
    sys.exit(int(main() or 0))
