from datasets import load_dataset
from transformers import BertTokenizer
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# gretelai/symptom_to_diagnosis usually ships with only a "train" split
# (sometimes "test" too) - NOT train/validation/test like the emotion
# dataset. So we load it once, then split it ourselves.
raw = load_dataset('gretelai/symptom_to_diagnosis')
print(raw)  # <-- run this once and LOOK at the printed output: confirms
            #     the real split names and column names before anything else

# each model needs ITS OWN matching tokenizer/checkpoint - bert-base-cased
# for BERT, roberta-base for RoBERTa (different vocab, BPE not WordPiece),
# distilbert-base-cased for DistilBERT. MyModel.py imports this same dict
# so the tokenizer and the model backbone always stay in sync.
MODEL_CHECKPOINTS = {
    "bert": "bert-base-cased",
    "roberta": "FacebookAI/roberta-base",
    "distilbert": "distilbert/distilbert-base-cased",
}


def get_tokenizer(model_type):
    checkpoint = MODEL_CHECKPOINTS[model_type]
    return AutoTokenizer.from_pretrained(checkpoint)


def encode(docs, tokenizer):
    '''
    This function takes list of texts and returns input_ids and attention_mask of texts
    '''
    encoded_dict = tokenizer(docs, add_special_tokens=True,
        max_length=128, padding='max_length', return_attention_mask=True, truncation=True,
        return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks


def get_trainvalidtest_loaders(model_type='bert', BATCH_SIZE=16, text_col='input_text', label_col='output_text'):
    # ---- Step 1: combine whatever splits exist into ONE dataframe ----
    # (emotion dataset already had train/valid/test separately - this
    # dataset does not, so we merge everything and split it ourselves)
    all_df = None
    for split_name in raw.keys():
        split_df = raw[split_name].to_pandas()
        all_df = split_df if all_df is None else all_df._append(split_df, ignore_index=True)

    print("---------columns available------------")
    print(all_df.columns.tolist())

    # rename to generic text/label so the rest of the code stays simple
    all_df = all_df.rename(columns={text_col: 'text', label_col: 'label_text'})
    all_df = all_df[['text', 'label_text']].dropna().reset_index(drop=True)

    print("---------counts before anything------------")
    print(all_df['label_text'].value_counts())
    print("---------------------------------------------------------")

    # ---- Step 2: string disease names -> integer label ids ----
    # (emotion dataset's labels were ALREADY integers with a ClassLabel
    # feature giving label_names for free - this dataset's labels are
    # plain disease-name strings, so we build label_names ourselves)
    label_encoder = LabelEncoder()
    all_df['label'] = label_encoder.fit_transform(all_df['label_text'])
    label_names = list(label_encoder.classes_)
    print("label_names:", label_names)

    # ---- Step 3: WE create train/valid/test splits (dataset has none) ----
    # num_train=500 like the original code won't work here - this dataset
    # is small (roughly a few dozen rows per class), so we split by
    # PERCENTAGE instead of a fixed count per class.
    train_df, temp_df = train_test_split(
        all_df, test_size=0.30, stratify=all_df['label'], random_state=42)
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("---------counts after split------------")
    print("train:", len(train_df), " valid:", len(valid_df), " test:", len(test_df))
    print(train_df['label'].value_counts())
    print("---------------------------------------------------------")

    # tokenizer must MATCH the model being used - bert needs BERT's
    # tokenizer, roberta needs RoBERTa's, distilbert needs its own
    tokenizer = get_tokenizer(model_type)
    print(tokenizer)

    train_list = train_df['text'].values.tolist()
    train_input_ids, train_att_masks = encode(train_list, tokenizer)
    valid_input_ids, valid_att_masks = encode(valid_df['text'].values.tolist(), tokenizer)
    test_input_ids, test_att_masks = encode(test_df['text'].values.tolist(), tokenizer)

    num_check = 5
    print(train_list[num_check])
    print(train_input_ids[num_check])
    print(train_att_masks[num_check])

    # get the labels
    train_y = torch.LongTensor(train_df['label'].values.tolist())
    valid_y = torch.LongTensor(valid_df['label'].values.tolist())
    test_y = torch.LongTensor(test_df['label'].values.tolist())
    print(train_y.size(), valid_y.size(), test_y.size())

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