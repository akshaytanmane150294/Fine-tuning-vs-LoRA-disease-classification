
"""
MyModel.py
Pretrained transformer backbone (BERT / RoBERTa / DistilBERT, chosen by
model_type) + a small classification head, generalized to N disease classes.

This replaces the old single-model MyBertModel.py - that file hardcoded
num_classes=6 (wrong for this dataset) and only ever loaded BERT. This
version takes num_classes as an argument and swaps the backbone based on
model_type, matching the MODEL_CHECKPOINTS dict in Utils.py.

DistilBERT has no pooler layer, so we grab the [CLS] token's hidden state
from last_hidden_state instead of pooler_output.
"""

import torch
from torch import nn
from transformers import AutoModel

from Utils import MODEL_CHECKPOINTS


class Classifier(nn.Module):
    def __init__(self, embedding_size, num_classes, hidden_layer=100, dropout=0.2):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(embedding_size, hidden_layer)
        self.dropout1 = nn.Dropout(dropout)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer, num_classes)
        # NOTE: returns raw logits, NOT softmax probabilities. The old
        # MyBertModel.py applied Softmax here AND used CrossEntropyLoss
        # in train.py - that double-applies a nonlinearity, which is a
        # bug. CrossEntropyLoss already does log-softmax internally, so
        # the classifier should output raw logits.

    def forward(self, encoded_input):
        out1 = self.dropout1(self.fc1(encoded_input))
        out2 = self.act1(out1)
        logits = self.fc2(out2)
        return logits


class MyModel(nn.Module):
    def __init__(self, model_type, num_classes, freeze_first_n_layers=0):
        """
        model_type: one of "bert", "roberta", "distilbert"
        num_classes: number of disease classes (NOT hardcoded - pass the
            real value, e.g. len(label_names) from Utils.py)
        freeze_first_n_layers: if > 0, freezes embeddings + first N
            encoder layers (optional - mirrors the original assignment's
            freezing experiment)
        """
        super(MyModel, self).__init__()
        self.model_type = model_type
        checkpoint = MODEL_CHECKPOINTS[model_type]
        self.backbone = AutoModel.from_pretrained(checkpoint)
        print(self.backbone)

        hidden_size = self.backbone.config.hidden_size
        self.classifier = Classifier(hidden_size, num_classes)

        if freeze_first_n_layers > 0:
            self._freeze_layers(freeze_first_n_layers)

    def _freeze_layers(self, n):
        modules = [self.backbone.embeddings]
        if hasattr(self.backbone, "encoder"):
            modules.append(self.backbone.encoder.layer[:n])
        elif hasattr(self.backbone, "transformer"):  # distilbert
            modules.append(self.backbone.transformer.layer[:n])
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        print(f"[info] froze embeddings + first {n} encoder layers")

    def forward(self, input_id, mask):
        output = self.backbone(input_ids=input_id, attention_mask=mask, return_dict=True)

        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            # BERT, RoBERTa both have a pooler
            pooled = output.pooler_output
        else:
            # DistilBERT has no pooler - use [CLS] token's hidden state
            pooled = output.last_hidden_state[:, 0, :]

        logits = self.classifier(pooled)
        return logits