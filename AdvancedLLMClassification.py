import argparse
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
# MODEL REGISTRY
# ---------------------------------------------------------------------------
MODEL_CHECKPOINTS = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "danube": "h2oai/h2o-danube-1.8b-chat",
    "llama": "meta-llama/Llama-3.2-1B-Instruct"
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tqdm.pandas()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def calc_accuracy(dataloader, model, tokenizer, type, max_length=512):
    """Evaluation for Single-Label Multi-Class (Symptom2Disease)"""
    with torch.no_grad():
        model.eval()
        pred_scores = []
        actual_scores = []
        for batch in tqdm(dataloader, total=len(dataloader), desc=f'Calc {type} accuracy'):
            prompt, targets = batch
            encodings = Utils.tokenize_text(prompt, tokenizer, max_length=max_length)
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


def calc_accuracy_multi_label(dataloader, model, tokenizer, type, label_names=None, threshold=0.30, max_length=512):
    """Evaluation for Multi-Label (BookSummaries) using Dynamic Sigmoid Threshold & Top-1 Fallback"""
    with torch.no_grad():
        model.eval()
        labels = []
        predictions = []
        for batch in tqdm(dataloader, total=len(dataloader), desc=f'Calc {type} accuracy'):
            prompt, targets = batch
            encodings = Utils.tokenize_text(prompt, tokenizer, max_length=max_length)
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
    parser = argparse.ArgumentParser(description="LoRA/QLoRA training for Decoder LLMs (Phi, LLaMA, Danube)")
    parser.add_argument("--model_type", choices=["phi3", "llama", "danube"], default="phi3", help="Model choice: phi3, llama, or danube (default: phi3)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size (default: 2)")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (default: 8, effective batch size = 16)")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence token length (default: 512)")
    parser.add_argument("--max_samples", type=int, default=5000, help="Number of samples (default: 5000; set 0 for full dataset)")
    parser.add_argument("--top_k_genres", type=int, default=20, help="Top K genres for multi-label (default: 20; set 0 for all)")
    parser.add_argument("--threshold", type=float, default=0.30, help="Multi-label probability threshold (default: 0.30)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 0.0002)")
    parser.add_argument("--seed", type=int, default=252, help="Random seed (default: 252)")
    parser.add_argument("--do_test", action="store_true", help="Run test evaluation only without training")
    args = parser.parse_args()

    model_checkpoint = MODEL_CHECKPOINTS.get(args.model_type, args.model_type)
    print(f"[info] Running {args.model_type} ({model_checkpoint}) | epochs={args.epochs} | batch_size={args.batch_size}x{args.grad_accum} | max_length={args.max_length} | max_samples={args.max_samples} | top_k_genres={args.top_k_genres} | threshold={args.threshold}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # =======================================================================
    # 2. DATASET LOADER (MULTI-LABEL ACTIVE)
    # =======================================================================
    max_samples = None if args.max_samples == 0 else args.max_samples
    top_k_genres = None if args.top_k_genres == 0 else args.top_k_genres
    book_path = "BookSummaries/BookSummaries/data/booksummaries/booksummaries.txt"

    # --- [MODE A: BOOKSUMMARIES MULTI-LABEL (ACTIVE)] ---
    CLASSIFICATION_TYPE = 'MULTI_LABEL'
    text_set, labels_dict, num_labels, mlb = dataloader.prepare_book_summaries(
        pairs=False, book_path=book_path, max_samples=max_samples, top_k_genres=top_k_genres
    )
    label_names = list(mlb.classes_)

    # --- [MODE B: SYMPTOM2DISEASE SINGLE-LABEL (COMMENTED OUT FOR TOGGLE)] ---
    # CLASSIFICATION_TYPE = 'MULTI_CLASS'
    # text_set, labels_dict, num_labels, label_encoder = dataloader.prepare_symptom_data(
    #     csv_path="archive/Symptom2Disease.csv"
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
        tx_train, tx_test, tx_val, labels_train, labels_test, labels_val, args.batch_size
    )

    learning_rate = args.lr
    diff_lr = 0.00001
    warmup_steps = 0
    weight_decay = 0.01

    L.seed_everything(seed=args.seed)

    # Instantiate model with dynamic number of classes
    model = ClassificationNetLoRA(model_checkpoint, args.do_test, APPLY_LORA=True, NUM_CLASSES=num_labels)
    model.to(device)

    # -----------------------------------------------------------------------
    # TEST ONLY MODE
    # -----------------------------------------------------------------------
    if args.do_test:
        if CLASSIFICATION_TYPE == 'MULTI_LABEL':
            test_acc = calc_accuracy_multi_label(test_dataloader, model, tokenizer, type='test', label_names=label_names, threshold=args.threshold, max_length=args.max_length)
        else:
            test_acc = calc_accuracy(test_dataloader, model, tokenizer, type='test', max_length=args.max_length)
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
        num_training_steps=args.epochs * len(train_dataloader)
    )

    scaler = GradScaler()
    optimizer.zero_grad()

    # -----------------------------------------------------------------------
    # TRAINING LOOP
    # -----------------------------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            prompt, targets = batch
            encodings = Utils.tokenize_text(prompt, tokenizer, max_length=args.max_length)
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            targets = targets.to(device)

            with autocast():
                logits = model(input_ids, attention_mask)
                if CLASSIFICATION_TYPE == 'MULTI_LABEL':
                    loss = F.binary_cross_entropy_with_logits(logits, targets.float())
                else:
                    loss = F.cross_entropy(logits, targets.long())
                    
            loss = loss / args.grad_accum
            scaler.scale(loss).backward()

            if ((batch_idx + 1) % args.grad_accum == 0) or ((batch_idx + 1) == len(train_dataloader)):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if (batch_idx % 50) == 0:
                print(
                    f'Epoch: {epoch+1}/{args.epochs} '
                    f'| Batch: {batch_idx+1}/{len(train_dataloader)} '
                    f'| Loss: {loss.item() * args.grad_accum:.4f}'
                )

        # Validation after epoch
        if CLASSIFICATION_TYPE == 'MULTI_LABEL':
            val_score = calc_accuracy_multi_label(val_dataloader, model, tokenizer, type='val', threshold=args.threshold, max_length=args.max_length)
        else:
            val_score = calc_accuracy(val_dataloader, model, tokenizer, type='val', max_length=args.max_length)
        print(f'Epoch {epoch+1} Complete | Validation Score: {val_score:.4f}')
        
        os.makedirs('SavedAdapters', exist_ok=True)
        os.makedirs('SavedClassificationModels', exist_ok=True)
        model.save_peft_adapter()

    # Final Test
    if CLASSIFICATION_TYPE == 'MULTI_LABEL':
        test_acc = calc_accuracy_multi_label(test_dataloader, model, tokenizer, type='test', label_names=label_names, threshold=args.threshold, max_length=args.max_length)
    else:
        test_acc = calc_accuracy(test_dataloader, model, tokenizer, type='test', max_length=args.max_length)
    print('Final Test Accuracy / F1:', test_acc)


if __name__ == "__main__":
    sys.exit(int(main() or 0))
