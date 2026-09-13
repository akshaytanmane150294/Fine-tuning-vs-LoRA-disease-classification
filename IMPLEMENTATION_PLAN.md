# Implementation Plan: Multi-Label BookSummaries Testing with Single-Label Toggles

This document outlines the changes made to the codebase to support **Multi-Label BookSummaries** classification across all 5 models (**BERT, DistilBERT, RoBERTa, Phi-3 LoRA, and LLaMA LoRA**). All previous **Single-Label (Symptom2Disease)** code is preserved and cleanly commented out with clear toggle markers.

---

## 1. Objective
- Enable Multi-Label Genre Classification on the `booksummaries.txt` dataset.
- Keep the code modular so the user can easily switch between Single-Label (Symptom2Disease) and Multi-Label (BookSummaries) without rewriting code.

---

## 2. Updated Code Components

### 1. `dataloader.py`
- Active: `prepare_book_summaries()` which parses tab-separated `booksummaries.txt`, extracts JSON genres, cleans invalid entries, and applies `MultiLabelBinarizer` (yielding 227 genre classes).
- Preserved / Commented: `prepare_symptom_data()` with `LabelEncoder` for single-label disease classification.

### 2. `Utils.py`
- Active: Multi-label data loaders returning `torch.FloatTensor` binary targets of shape `[batch_size, 227]`.
- Max token length default updated to `512` for long summaries (with `128` commented out).
- Added missing helper functions for LoRA LLMs: `tokenize_text`, `get_train_test_val_Loaders`, and `get_optimizer`.

### 3. `train.py` (BERT, DistilBERT, RoBERTa)
- Active: `IS_MULTI_LABEL = True`, `nn.BCEWithLogitsLoss()`, and Sigmoid thresholding (`torch.sigmoid(logits) >= 0.5`).
- Multi-label metrics: Micro F1, Macro F1, and Weighted F1.
- Preserved / Commented: `nn.CrossEntropyLoss()` and `np.argmax()` for single-label.

### 4. `AdvancedLLMClassification.py` (Phi-3, LLaMA LoRA)
- Active: `CLASSIFICATION_TYPE = 'MULTI_LABEL'`, `MAX_SEQ_LENGTH = 512`, `BOOK_PATH`, and `calc_accuracy_multi_label()`.
- Preserved / Commented: `CLASSIFICATION_TYPE = 'MULTI_CLASS'`, `MAX_SEQ_LENGTH = 128`, and `CSV_PATH`.
