
"""
train.py
Trains ONE model (bert / roberta / distilbert) on the symptom->diagnosis
dataset (gretelai/symptom_to_diagnosis) using Utils.get_trainvalidtest_loaders().

Usage:
    python train.py --model_type bert
    python train.py --model_type roberta
    python train.py --model_type distilbert

Run all three, then use compare_all.py to build the comparison table.
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import Utils
from MyModel import MyModel


def plot_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), max(6, len(labels) * 0.5)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False, xticks_rotation=90)
    plt.title("Normalized confusion matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")


def top_confused_pairs(y_true, y_pred, labels, top_n=10):
    """With >10 classes a full confusion matrix heatmap gets unreadable -
    this prints the worst (true, predicted) label mix-ups instead."""
    cm = confusion_matrix(y_true, y_pred)
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], labels[i], labels[j]))
    pairs.sort(reverse=True)
    print(f"\n[top {top_n} confused pairs] (true -> predicted : count)")
    for count, true_label, pred_label in pairs[:top_n]:
        print(f"  {true_label} -> {pred_label} : {count}")


def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    start = time.time()
    with torch.no_grad():
        for batch in dataloader:
            input_ids, att_mask, labels = [d.to(device) for d in batch]
            logits = model(input_id=input_ids, mask=att_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.append(np.argmax(logits.cpu().numpy(), axis=-1))
            all_labels.append(labels.cpu().numpy())
    elapsed = time.time() - start
    ms_per_batch = 1000 * elapsed / max(len(dataloader), 1)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss, all_preds, all_labels, ms_per_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["bert", "roberta", "distilbert"], required=True)
    parser.add_argument("--text_col", default="input_text")
    parser.add_argument("--label_col", default="output_text")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device} model_type={args.model_type}")

    # ---- matches Utils.py's current signature EXACTLY: 6 return values ----
    (train_dataloader, valid_dataloader, test_dataloader,
     train_df, valid_df, label_names) = Utils.get_trainvalidtest_loaders(
        model_type=args.model_type,
        BATCH_SIZE=args.batch_size,
        text_col=args.text_col,
        label_col=args.label_col,
    )
    num_classes = len(label_names)  # Utils.py doesn't return this separately,
                                     # so we derive it from label_names here
    print(f"[info] num_classes={num_classes}")

    model = MyModel(args.model_type, num_classes).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[info] total params={num_params:,} trainable={num_trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * args.epochs
    )

    train_loss_per_epoch, val_loss_per_epoch = [], []
    training_start = time.time()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, desc="Training"):
            input_ids, att_mask, labels = [d.to(device) for d in batch]
            logits = model(input_id=input_ids, mask=att_mask)
            loss = criterion(logits, labels)
            train_loss += loss.item()
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        train_loss_per_epoch.append(train_loss / max(len(train_dataloader), 1))

        val_loss, val_preds, val_labels, _ = evaluate(model, valid_dataloader, device, criterion)
        val_loss_per_epoch.append(val_loss)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        print(f"  train_loss={train_loss_per_epoch[-1]:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_macro_f1={val_f1:.3f}")

    total_train_time = time.time() - training_start
    time_per_epoch = total_train_time / args.epochs

    # ---- loss curve ----
    epochs_range = range(1, args.epochs + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs_range, train_loss_per_epoch, label="training loss")
    ax.plot(epochs_range, val_loss_per_epoch, label="validation loss")
    ax.set_title(f"Training and Validation loss ({args.model_type})")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    loss_plot_path = f"{args.out_dir}/{args.model_type}_loss_curve.png"
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f"[saved] {loss_plot_path}")

    # ---- final test-set evaluation ----
    test_loss, test_preds, test_labels, ms_per_batch = evaluate(model, test_dataloader, device, criterion)
    test_acc = accuracy_score(test_labels, test_preds)
    test_macro_f1 = f1_score(test_labels, test_preds, average="macro")
    test_weighted_f1 = f1_score(test_labels, test_preds, average="weighted")

    print("\n=== TEST SET RESULTS ===")
    print(classification_report(test_labels, test_preds, target_names=label_names, zero_division=0))

    cm_path = f"{args.out_dir}/{args.model_type}_confusion_matrix.png"
    if num_classes <= 15:
        plot_confusion_matrix(test_labels, test_preds, label_names, cm_path)
    else:
        top_confused_pairs(test_labels, test_preds, label_names)

    summary = {
        "model_type": args.model_type,
        "num_classes": num_classes,
        "total_params": num_params,
        "trainable_params": num_trainable,
        "epochs": args.epochs,
        "test_accuracy": round(float(test_acc), 4),
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
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()