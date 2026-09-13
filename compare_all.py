
"""
MyModel.py
Pretrained transformer backbone (BERT / RoBERTa / DistilBERT, chosen by
model_type) + a small classification head, generalized to N disease classes.

DistilBERT has no pooler layer, so we grab the [CLS] token's hidden state
from last_hidden_state instead - this mirrors what the assignment does
with pooled_output[0][:,0,:] for DistilBert/RoBERTa.
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
        # NOTE: we return raw logits here, not softmax probabilities -
        # nn.CrossEntropyLoss expects logits and applies log-softmax
        # internally. (The original assignment code applies Softmax
        # before CrossEntropyLoss, which double-applies the nonlinearity -
        # worth fixing in your write-up as an improvement.)

    def forward(self, encoded_input):
        out1 = self.dropout1(self.fc1(encoded_input))
        out2 = self.act1(out1)
        logits = self.fc2(out2)
        return logits


class MyModel(nn.Module):
    def __init__(self, model_type, num_classes, freeze_first_n_layers=0):
        """
        model_type: one of "bert", "roberta", "distilbert"
        num_classes: number of disease classes
        freeze_first_n_layers: if > 0, freezes embeddings + first N encoder
            layers (mirrors the assignment's freezing experiment)
        """
        super(MyModel, self).__init__()
        self.model_type = model_type
        checkpoint = MODEL_CHECKPOINTS[model_type]
        self.backbone = AutoModel.from_pretrained(checkpoint)

        hidden_size = self.backbone.config.hidden_size
        self.classifier = Classifier(hidden_size, num_classes)

        if freeze_first_n_layers > 0:
            self._freeze_layers(freeze_first_n_layers)

    def _freeze_layers(self, n):
        modules = [self.backbone.embeddings]
        # encoder layer path differs slightly by architecture name
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
            # BERT, RoBERTa
            pooled = output.pooler_output
        else:
            # DistilBERT has no pooler - use [CLS] token's hidden state
            pooled = output.last_hidden_state[:, 0, :]

        logits = self.classifier(pooled)
        return logits