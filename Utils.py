import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.optim import AdamW

import dataloader

# Each model checkpoint definition
MODEL_CHECKPOINTS = {
    "bert": "bert-base-cased",
    "roberta": "FacebookAI/roberta-base",
    "distilbert": "distilbert/distilbert-base-cased",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "llama": "meta-llama/Llama-3.2-1B-Instruct"  # or h2oai/h2o-danube-1.8b-chat
}


def get_tokenizer(model_type):
    checkpoint = MODEL_CHECKPOINTS.get(model_type, model_type)
    return AutoTokenizer.from_pretrained(checkpoint)


def encode(docs, tokenizer, max_length=512):
    """
    Takes list of texts and returns input_ids and attention_mask tensors.
    --- [MULTI-LABEL / BOOKSUMMARIES (ACTIVE)]: max_length = 512
    --- [SINGLE-LABEL / SYMPTOM2DISEASE (COMMENTED)]: max_length = 128
    """
    encoded_dict = tokenizer(
        docs,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks


# ===========================================================================
# DATALOADER BUILDER FOR BERT / ROBERTA / DISTILBERT
# ===========================================================================
def get_trainvalidtest_loaders(model_type='bert', BATCH_SIZE=16, max_length=512, max_samples=2000):
    """
    Returns train, valid, test dataloaders, dataframes, and class names.
    :param max_samples: subset size (default: 2000 books for fast training; pass None for full dataset)
    """
    tokenizer = get_tokenizer(model_type)

    # -----------------------------------------------------------------------
    # --- [MODE A: BOOKSUMMARIES MULTI-LABEL (ACTIVE)] ---
    # -----------------------------------------------------------------------
    text_set, label_set, num_labels, mlb = dataloader.prepare_book_summaries(pairs=False, max_samples=max_samples)
    label_names = list(mlb.classes_)

    train_input_ids, train_att_masks = encode(text_set['train'], tokenizer, max_length=max_length)
    valid_input_ids, valid_att_masks = encode(text_set['dev'], tokenizer, max_length=max_length)
    test_input_ids, test_att_masks = encode(text_set['test'], tokenizer, max_length=max_length)

    # For multi-label BCEWithLogitsLoss, targets MUST be FloatTensor [N, num_labels]
    train_y = torch.tensor(label_set['train'], dtype=torch.float32)
    valid_y = torch.tensor(label_set['dev'], dtype=torch.float32)
    test_y = torch.tensor(label_set['test'], dtype=torch.float32)

    train_df = pd.DataFrame({'text': text_set['train']})
    valid_df = pd.DataFrame({'text': text_set['dev']})
    test_df = pd.DataFrame({'text': text_set['test']})

    # -----------------------------------------------------------------------
    # --- [MODE B: SYMPTOM2DISEASE SINGLE-LABEL (COMMENTED OUT FOR TOGGLE)] ---
    # To switch back to Symptom2Disease single-label classification:
    # 1. Comment out the Mode A block above.
    # 2. Uncomment the Mode B block below.
    # -----------------------------------------------------------------------
    # from sklearn.preprocessing import LabelEncoder
    # text_set, label_set, num_labels, le = dataloader.prepare_symptom_data()
    # label_names = list(le.classes_)
    # train_input_ids, train_att_masks = encode(text_set['train'], tokenizer, max_length=128)
    # valid_input_ids, valid_att_masks = encode(text_set['dev'], tokenizer, max_length=128)
    # test_input_ids, test_att_masks = encode(text_set['test'], tokenizer, max_length=128)
    # train_y = torch.LongTensor(label_set['train'])
    # valid_y = torch.LongTensor(label_set['dev'])
    # test_y = torch.LongTensor(label_set['test'])
    # train_df = pd.DataFrame({'text': text_set['train']})
    # valid_df = pd.DataFrame({'text': text_set['dev']})
    # test_df = pd.DataFrame({'text': text_set['test']})
    # -----------------------------------------------------------------------

    train_dataset = TensorDataset(train_input_ids, train_att_masks, train_y)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

    valid_dataset = TensorDataset(valid_input_ids, valid_att_masks, valid_y)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE)

    test_dataset = TensorDataset(test_input_ids, test_att_masks, test_y)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

    return train_dataloader, valid_dataloader, test_dataloader, train_df, valid_df, label_names


# ===========================================================================
# HELPER UTILITIES FOR ADVANCED LLM / LORA MODELS (Phi-3, LLaMA)
# ===========================================================================
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], torch.tensor(self.labels[idx])


def tokenize_text(texts, tokenizer, max_length=512):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )


def get_train_test_val_Loaders(tx_train, tx_test, tx_val, labels_train, labels_test, labels_val, batch_size):
    train_ds = TextClassificationDataset(tx_train, labels_train)
    test_ds = TextClassificationDataset(tx_test, labels_test)
    val_ds = TextClassificationDataset(tx_val, labels_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader


def get_optimizer(model, learning_rate=2e-4, diff_lr=1e-5, weight_decay=0.01):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    return AdamW(optimizer_grouped_parameters)