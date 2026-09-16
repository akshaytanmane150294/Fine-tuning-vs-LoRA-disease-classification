import sys
import os
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import lightning as L

import dataloader
import Utils
from ClassificationNetLoRA import ClassificationNetLoRA

# ---------------------------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------------------------
MODEL_CHECKPOINTS = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "danube": "h2oai/h2o-danube-1.8b-chat",
    "llama": "meta-llama/Llama-3.2-1B-Instruct"
}
# Pick model: "phi3" or "llama" or "danube"
MODEL_NAME = MODEL_CHECKPOINTS["phi3"]

DO_TEST = False                     # False = training, True = testing
BATCH_SIZE = 2                      # Smaller batch size for long sequences on GPU
GRADIENT_ACCUMULATION_STEPS = 8     # Effective batch size = 16
num_epochs = 5
seed = 252
APPLY_LORA = True

# ===========================================================================
# 1. TASK SELECTION TOGGLE (BOOKSUMMARIES vs SYMPTOM2DISEASE)
# ===========================================================================
# --- [MODE A: BOOKSUMMARIES MULTI-LABEL (ACTIVE)] ---
CLASSIFICATION_TYPE = 'MULTI_LABEL'
MAX_SEQ_LENGTH = 512
MAX_SAMPLES = 2000                  # 2000 books subset for fast training
BOOK_PATH = "BookSummaries/BookSummaries/data/booksummaries/booksummaries.txt"

# --- [MODE B: SYMPTOM2DISEASE SINGLE-LABEL (COMMENTED OUT FOR TOGGLE)] ---
# CLASSIFICATION_TYPE = 'MULTI_CLASS'
# MAX_SEQ_LENGTH = 128
# CSV_PATH = "archive/Symptom2Disease.csv"
# ===========================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tqdm.pandas()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def calc_accuracy(dataloader, model, tokenizer, type):
    """Evaluation for Single-Label Multi-Class (Symptom2Disease)"""
    with torch.no_grad():
        model.eval()
        pred_scores = []
        actual_scores = []
        for batch in tqdm(dataloader, total=len(dataloader), desc=f'Calc {type} accuracy'):
            prompt, targets = batch
            encodings = Utils.tokenize_text(prompt, tokenizer, max_length=MAX_SEQ_LENGTH)
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            with autocast():
                logits = model(input_ids, attention_mask)
            pred_score = F.softmax(logits, dim=-1).argmax(dim=-1).cpu().detach().numpy().tolist()
            pred_scores.extend(pred_score)
            actual_scores.extend(targets.numpy().tolist())
            
        pred_scores = np.array(pred_scores)
        accuracy = accuracy_score(actual_scores, pred_scores)
        return accuracy


def calc_accuracy_multi_label(dataloader, model, tokenizer, type, label_names=None, threshold=0.20):
    """Evaluation for Multi-Label (BookSummaries) using Dynamic Sigmoid Threshold & Top-1 Fallback"""
    with torch.no_grad():
        model.eval()
        labels = []
        predictions = []
        for batch in tqdm(dataloader, total=len(dataloader), desc=f'Calc {type} accuracy'):
            prompt, targets = batch
            encodings = Utils.tokenize_text(prompt, tokenizer, max_length=MAX_SEQ_LENGTH)
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            with autocast():
                logits = model(input_ids, attention_mask)
            preds = torch.sigmoid(logits).detach().cpu()
            predictions.extend(preds)
            for out_labels in targets.detach().cpu():
                labels.append(out_labels)
                
        labels = torch.stack(labels).int().numpy()
        predictions = torch.stack(predictions).numpy()
        y_pred_labels = np.where(predictions >= threshold, 1, 0)
        
        # Dynamic Top-1 Fallback: If no class exceeds threshold, pick top-1 class
        for i in range(len(y_pred_labels)):
            if y_pred_labels[i].sum() == 0:
                y_pred_labels[i, predictions[i].argmax()] = 1
        
        micro_f1 = f1_score(labels, y_pred_labels, average='micro', zero_division=0)
        macro_f1 = f1_score(labels, y_pred_labels, average='macro', zero_division=0)
        print(f"[{type}] Micro F1: {micro_f1:.4f} | Macro F1: {macro_f1:.4f}")
        
        if type == 'test' and label_names is not None:
            print("\n=== CLASSIFICATION REPORT ===")
            print(classification_report(labels, y_pred_labels, target_names=label_names, zero_division=0))
            
        return micro_f1


def main():
    model_name = MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # =======================================================================
    # 2. DATASET LOADER TOGGLE
    # =======================================================================
    # --- [MODE A: BOOKSUMMARIES MULTI-LABEL (ACTIVE)] ---
    text_set, labels_dict, num_labels, mlb = dataloader.prepare_book_summaries(
        pairs=False, book_path=BOOK_PATH, max_samples=MAX_SAMPLES
    )
    label_names = list(mlb.classes_)

    # --- [MODE B: SYMPTOM2DISEASE SINGLE-LABEL (COMMENTED OUT FOR TOGGLE)] ---
    # text_set, labels_dict, num_labels, label_encoder = dataloader.prepare_symptom_data(
    #     csv_path=CSV_PATH
    # )
    # label_names = list(label_encoder.classes_)
    # =======================================================================

    bos = tokenizer.bos_token if tokenizer.bos_token else ""
    tx_train = [bos + str(x) for x in text_set["train"]]
    tx_test = [bos + str(x) for x in text_set["test"]]
    tx_val = [bos + str(x) for x in text_set["dev"]]

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

    # Instantiate model with dynamic number of classes
    model = ClassificationNetLoRA(MODEL_NAME, DO_TEST, APPLY_LORA, NUM_CLASSES=num_labels)
    model.to(device)

    # -----------------------------------------------------------------------
    # TEST ONLY MODE
    # -----------------------------------------------------------------------
    if DO_TEST:
        if CLASSIFICATION_TYPE == 'MULTI_LABEL':
            test_acc = calc_accuracy_multi_label(test_dataloader, model, tokenizer, type='test', label_names=label_names)
        else:
            test_acc = calc_accuracy(test_dataloader, model, tokenizer, type='test')
        print('Final Test Score:', test_acc)
        return

    print('Trainable parameters:')
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(" ", n)

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

    # -----------------------------------------------------------------------
    # TRAINING LOOP
    # -----------------------------------------------------------------------
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            prompt, targets = batch
            encodings = Utils.tokenize_text(prompt, tokenizer, max_length=MAX_SEQ_LENGTH)
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            targets = targets.to(device)

            with autocast():
                logits = model(input_ids, attention_mask)
                if CLASSIFICATION_TYPE == 'MULTI_LABEL':
                    loss = F.binary_cross_entropy_with_logits(logits, targets.float())
                else:
                    loss = F.cross_entropy(logits, targets.long())
                    
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            scaler.scale(loss).backward()

            if ((batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0) or ((batch_idx + 1) == len(train_dataloader)):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if (batch_idx % 50) == 0:
                print(
                    f'Epoch: {epoch+1}/{num_epochs} '
                    f'| Batch: {batch_idx+1}/{len(train_dataloader)} '
                    f'| Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}'
                )

        # Validation after epoch
        if CLASSIFICATION_TYPE == 'MULTI_LABEL':
            val_score = calc_accuracy_multi_label(val_dataloader, model, tokenizer, type='val')
        else:
            val_score = calc_accuracy(val_dataloader, model, tokenizer, type='val')
        print(f'Epoch {epoch+1} Complete | Validation Score: {val_score:.4f}')
        
        os.makedirs('SavedAdapters', exist_ok=True)
        os.makedirs('SavedClassificationModels', exist_ok=True)
        model.save_peft_adapter()

    # Final Test
    if CLASSIFICATION_TYPE == 'MULTI_LABEL':
        test_acc = calc_accuracy_multi_label(test_dataloader, model, tokenizer, type='test', label_names=label_names)
    else:
        test_acc = calc_accuracy(test_dataloader, model, tokenizer, type='test')
    print('Final Test Accuracy / F1:', test_acc)


if __name__ == "__main__":
    sys.exit(int(main() or 0))
