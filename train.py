"""
train.py
Trains ONE model (bert / roberta / distilbert) on BookSummaries (Multi-label)
or Symptom2Disease (Single-label) using Utils.get_trainvalidtest_loaders().

Usage:
    python train.py --model_type bert
    python train.py --model_type roberta
    python train.py --model_type distilbert
"""

import argparse
import json
import time
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import Utils
from MyModel import MyModel


# ===========================================================================
# EVALUATION FUNCTION
# ===========================================================================
def evaluate(model, dataloader, device, criterion, is_multi_label=True):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    start = time.time()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, att_mask, labels = [d.to(device) for d in batch]
            logits = model(input_id=input_ids, mask=att_mask)
            
            # --- [MULTI-LABEL / BOOKSUMMARIES (ACTIVE)] ---
            if is_multi_label:
                loss = criterion(logits, labels.float())
                preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
            # --- [SINGLE-LABEL / SYMPTOM2DISEASE (COMMENTED / CONDITIONAL)] ---
            else:
                loss = criterion(logits, labels.long())
                preds = np.argmax(logits.cpu().numpy(), axis=-1)
                
            total_loss += loss.item()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            
    elapsed = time.time() - start
    ms_per_batch = 1000 * elapsed / max(len(dataloader), 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss, all_preds, all_labels, ms_per_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["bert", "roberta", "distilbert"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=2000, help="Number of books to sample (default: 2000 for fast training; set None/0 for full dataset)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device} model_type={args.model_type}")

    # =======================================================================
    # 1. TASK SELECTION TOGGLE
    # =======================================================================
    # --- [MULTI-LABEL / BOOKSUMMARIES (ACTIVE)] ---
    IS_MULTI_LABEL = True
    
    # --- [SINGLE-LABEL / SYMPTOM2DISEASE (TOGGLE)] ---
    # IS_MULTI_LABEL = False

    max_samples = None if args.max_samples == 0 else args.max_samples

    # Load dataloaders
    (train_dataloader, valid_dataloader, test_dataloader,
     train_df, valid_df, label_names) = Utils.get_trainvalidtest_loaders(
        model_type=args.model_type,
        BATCH_SIZE=args.batch_size,
        max_length=args.max_length,
        max_samples=max_samples
    )
    num_classes = len(label_names)
    print(f"[info] num_classes={num_classes} | multi_label={IS_MULTI_LABEL}")

    # Create model
    model = MyModel(args.model_type, num_classes).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[info] total params={num_params:,} trainable={num_trainable:,}")

    # =======================================================================
    # 2. LOSS CRITERION TOGGLE
    # =======================================================================
    # --- [MULTI-LABEL / BOOKSUMMARIES (ACTIVE)] ---
    criterion = nn.BCEWithLogitsLoss()
    
    # --- [SINGLE-LABEL / SYMPTOM2DISEASE (COMMENTED)] ---
    # criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * args.epochs
    )

    train_loss_per_epoch, val_loss_per_epoch = [], []
    training_start = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, desc="Training"):
            input_ids, att_mask, labels = [d.to(device) for d in batch]
            logits = model(input_id=input_ids, mask=att_mask)
            
            # --- Loss computation ---
            if IS_MULTI_LABEL:
                loss = criterion(logits, labels.float())
            else:
                loss = criterion(logits, labels.long())
                
            train_loss += loss.item()
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
        train_loss_per_epoch.append(train_loss / max(len(train_dataloader), 1))

        # Validation
        val_loss, val_preds, val_labels, _ = evaluate(
            model, valid_dataloader, device, criterion, is_multi_label=IS_MULTI_LABEL
        )
        val_loss_per_epoch.append(val_loss)

        if IS_MULTI_LABEL:
            val_micro_f1 = f1_score(val_labels, val_preds, average="micro", zero_division=0)
            val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
            print(f"  train_loss={train_loss_per_epoch[-1]:.4f} | val_loss={val_loss:.4f} "
                  f"| val_micro_f1={val_micro_f1:.3f} | val_macro_f1={val_macro_f1:.3f}")
        else:
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
            print(f"  train_loss={train_loss_per_epoch[-1]:.4f} | val_loss={val_loss:.4f} "
                  f"| val_acc={val_acc:.3f} | val_macro_f1={val_f1:.3f}")

    total_train_time = time.time() - training_start
    time_per_epoch = total_train_time / args.epochs

    # ---- Final test-set evaluation ----
    test_loss, test_preds, test_labels, ms_per_batch = evaluate(
        model, test_dataloader, device, criterion, is_multi_label=IS_MULTI_LABEL
    )

    print("\n=== TEST SET RESULTS ===")
    if IS_MULTI_LABEL:
        test_micro_f1 = f1_score(test_labels, test_preds, average="micro", zero_division=0)
        test_macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
        test_weighted_f1 = f1_score(test_labels, test_preds, average="weighted", zero_division=0)
        test_acc = test_micro_f1  # In multi-label micro-F1 reflects overall multi-label accuracy
        print(f"Test Micro F1: {test_micro_f1:.4f} | Macro F1: {test_macro_f1:.4f} | Weighted F1: {test_weighted_f1:.4f}")
        print(classification_report(test_labels, test_preds, target_names=label_names, zero_division=0))
    else:
        test_acc = accuracy_score(test_labels, test_preds)
        test_macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
        test_weighted_f1 = f1_score(test_labels, test_preds, average="weighted", zero_division=0)
        print(f"Test Accuracy: {test_acc:.4f} | Macro F1: {test_macro_f1:.4f} | Weighted F1: {test_weighted_f1:.4f}")
        print(classification_report(test_labels, test_preds, target_names=label_names, zero_division=0))

    summary = {
        "model_type": args.model_type,
        "is_multi_label": IS_MULTI_LABEL,
        "num_classes": num_classes,
        "total_params": num_params,
        "trainable_params": num_trainable,
        "epochs": args.epochs,
        "test_score": round(float(test_acc), 4),
        "test_macro_f1": round(float(test_macro_f1), 4),
        "test_weighted_f1": round(float(test_weighted_f1), 4),
        "total_train_time_sec": round(total_train_time, 1),
        "time_per_epoch_sec": round(time_per_epoch, 1),
        "inference_ms_per_batch": round(ms_per_batch, 2),
    }
    summary_path = f"{args.out_dir}/{args.model_type}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {summary_path}")


if __name__ == "__main__":
    main()