import torch


class ClassificationHead(torch.nn.Module):
    def __init__(self, hidden_size, num_classes=24) -> None:
        # num_classes = 24 for the Symptom2Disease dataset (single-label, 24 diseases)
        super(ClassificationHead, self).__init__()
        dropout_rate = 0.1
        self.cls_head = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size, 768),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(768),
            torch.nn.Linear(768, num_classes)   # 227 -> 24 (Symptom2Disease classes)
        )

    def forward(self, x):
        return self.cls_head(x)
