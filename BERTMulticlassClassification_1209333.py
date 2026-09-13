
import Utils
import sys
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
from MyBertModel import MyBertModel
from torch.nn.utils import clip_grad_norm_
from torch import nn
from tqdm import tqdm
import numpy as np
import math
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_preds, y_true, labels=None):
    """Plots a normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized Confusion Matrix")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset and dataloaders
    train_dataloader, valid_dataloader, test_dataloader, train_df, valid_df, label_names = Utils.get_trainvalidtest_loaders()

    # Load pretrained BERT model + classifier
    model = MyBertModel().to(device)

    # Training parameters
    EPOCHS = 15
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 16

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * EPOCHS
    )

    train_loss_per_epoch = []
    val_loss_per_epoch = []

    # -------- Training Loop --------
    for epoch_num in range(EPOCHS):
        print(f"Epoch {epoch_num + 1}/{EPOCHS}")
        model.train()
        train_loss = 0

        for step_num, batch_data in enumerate(tqdm(train_dataloader, desc="Training")):
            input_ids, att_mask, labels = [data.to(device) for data in batch_data]
            output = model(input_id=input_ids, mask=att_mask)
            loss = criterion(output, labels)
            train_loss += loss.item()

            model.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = train_loss / (step_num + 1)
        train_loss_per_epoch.append(avg_train_loss)

        # -------- Validation Loop --------
        model.eval()
        valid_loss = 0
        valid_pred = []

        with torch.no_grad():
            for step_num_e, batch_data in enumerate(tqdm(valid_dataloader, desc="Validation")):
                input_ids, att_mask, labels = [data.to(device) for data in batch_data]
                output = model(input_id=input_ids, mask=att_mask)
                loss = criterion(output, labels)
                valid_loss += loss.item()
                valid_pred.append(np.argmax(output.cpu().numpy(), axis=-1))

        avg_val_loss = valid_loss / (step_num_e + 1)
        val_loss_per_epoch.append(avg_val_loss)
        valid_pred = np.concatenate(valid_pred)

        print(f"Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

    # -------- Plot Training vs Validation Loss --------
    epochs = range(1, EPOCHS + 1)
    plt.plot(epochs, train_loss_per_epoch, label="Training Loss")
    plt.plot(epochs, val_loss_per_epoch, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # -------- Classification Report --------
    print("Classification Report:")
    print(classification_report(valid_df["label"].to_numpy(), valid_pred, target_names=label_names))

    # -------- Confusion Matrix --------
    plot_confusion_matrix(valid_pred, valid_df["label"].to_numpy(), labels=label_names)


if __name__ == "__main__":
    sys.exit(int(main() or 0))
