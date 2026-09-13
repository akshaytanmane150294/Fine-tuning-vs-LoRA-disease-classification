# Walkthrough: Multi-Label BookSummaries Setup & Single-Label Switch Guide

All models (**BERT, DistilBERT, RoBERTa, Phi-3 LoRA, and LLaMA LoRA**) are now configured to run on the **Multi-Label BookSummaries** dataset. All previous **Single-Label Symptom2Disease** code has been cleanly preserved and commented out.

---

## 1. Summary of Changes Made

| File | What was Added (Active) | What was Commented (Toggle) |
| :--- | :--- | :--- |
| **[`dataloader.py`](file:///b:/Python-Projects/DiseaseSymptomClassification/DiseaseSymptomClassification/dataloader.py)** | `prepare_book_summaries()` parsing `booksummaries.txt` with `MultiLabelBinarizer` | `prepare_symptom_data()` with `LabelEncoder` |
| **[`Utils.py`](file:///b:/Python-Projects/DiseaseSymptomClassification/DiseaseSymptomClassification/Utils.py)** | BookSummaries loaders returning `torch.FloatTensor` multi-label targets; `max_length=512`; LoRA helper functions | Single-label `LabelEncoder` + `torch.LongTensor` |
| **[`train.py`](file:///b:/Python-Projects/DiseaseSymptomClassification/DiseaseSymptomClassification/train.py)** | `IS_MULTI_LABEL = True`, `BCEWithLogitsLoss`, Sigmoid thresholding ($\ge 0.5$), Micro/Macro F1 metrics | `CrossEntropyLoss`, `np.argmax`, single-label accuracy |
| **[`AdvancedLLMClassification.py`](file:///b:/Python-Projects/DiseaseSymptomClassification/DiseaseSymptomClassification/AdvancedLLMClassification.py)** | `CLASSIFICATION_TYPE = 'MULTI_LABEL'`, `MAX_SEQ_LENGTH = 512`, `BOOK_PATH`, `calc_accuracy_multi_label` | `CLASSIFICATION_TYPE = 'MULTI_CLASS'`, `MAX_SEQ_LENGTH = 128` |

---

## 2. How to Run the 5 Models on BookSummaries (Current Active Mode)

### BERT, DistilBERT, RoBERTa:
```bash
# Default (epochs=10, batch_size=8, max_length=512)
python train.py --model_type bert
python train.py --model_type distilbert
python train.py --model_type roberta

# Custom parameters:
python train.py --model_type bert --epochs 15 --batch_size 8 --max_length 512
```

### Phi-3 & LLaMA with LoRA:
```bash
python AdvancedLLMClassification.py
```
*(In `AdvancedLLMClassification.py`, select `MODEL_NAME = MODEL_CHECKPOINTS["phi3"]` or `MODEL_CHECKPOINTS["llama"]`)*

---

## 3. How to Switch Back to Single-Label (`Symptom2Disease`)

When you want to switch back to single-label classification, follow these exact 4 steps:

### Step A: In [`Utils.py`](file:///b:/Python-Projects/DiseaseSymptomClassification/DiseaseSymptomClassification/Utils.py)
1. Comment out **Mode A** (lines ~46–64):
   ```python
   # text_set, label_set, num_labels, mlb = dataloader.prepare_book_summaries(pairs=False)
   # ...
   ```
2. Uncomment **Mode B** (lines ~71–84):
   ```python
   text_set, label_set, num_labels, le = dataloader.prepare_symptom_data()
   label_names = list(le.classes_)
   train_input_ids, train_att_masks = encode(text_set['train'], tokenizer, max_length=128)
   valid_input_ids, valid_att_masks = encode(text_set['dev'], tokenizer, max_length=128)
   test_input_ids, test_att_masks = encode(text_set['test'], tokenizer, max_length=128)
   train_y = torch.LongTensor(label_set['train'])
   valid_y = torch.LongTensor(label_set['dev'])
   test_y = torch.LongTensor(label_set['test'])
   ```

### Step B: In [`train.py`](file:///b:/Python-Projects/DiseaseSymptomClassification/DiseaseSymptomClassification/train.py) (for BERT, DistilBERT, RoBERTa)
1. Set `IS_MULTI_LABEL = False`:
   ```python
   # IS_MULTI_LABEL = True
   IS_MULTI_LABEL = False
   ```
2. Switch criterion:
   ```python
   # criterion = nn.BCEWithLogitsLoss()
   criterion = nn.CrossEntropyLoss()
   ```

### Step C: In [`AdvancedLLMClassification.py`](file:///b:/Python-Projects/DiseaseSymptomClassification/DiseaseSymptomClassification/AdvancedLLMClassification.py) (for Phi-3 & LLaMA)
1. In the Config Section:
   ```python
   # CLASSIFICATION_TYPE = 'MULTI_LABEL'
   # MAX_SEQ_LENGTH = 512
   CLASSIFICATION_TYPE = 'MULTI_CLASS'
   MAX_SEQ_LENGTH = 128
   CSV_PATH = "archive/Symptom2Disease.csv"
   ```
2. In the Data Loading Section of `main()`:
   ```python
   # text_set, labels_dict, num_labels, mlb = dataloader.prepare_book_summaries(...)
   text_set, labels_dict, num_labels, label_encoder = dataloader.prepare_symptom_data(csv_path=CSV_PATH)
   label_names = list(label_encoder.classes_)
   ```

### Step D: In [`dataloader.py`](file:///b:/Python-Projects/DiseaseSymptomClassification/DiseaseSymptomClassification/dataloader.py)
Uncomment `def prepare_symptom_data(...)` (lines ~128–172).
