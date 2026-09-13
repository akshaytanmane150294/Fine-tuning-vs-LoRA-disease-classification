import json
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
try:
    from lightning import seed_everything
except ImportError:
    try:
        from pytorch_lightning import seed_everything
    except ImportError:
        import random, numpy as np, torch
        def seed_everything(seed=42):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)



def parse_json_column(genre_data):
    """
    Read genre information as a json string and convert it to a dict
    :param genre_data: genre data to be converted
    :return: dict of genre names
    """
    try:
        return json.loads(genre_data)
    except Exception as e:
        return None  # when genre information is missing


# ===========================================================================
# [1] BOOKSUMMARIES DATASET LOADER — Multi-Label (ACTIVE)
# ===========================================================================
def load_booksummaries_data(book_path='BookSummaries/BookSummaries/data/booksummaries/booksummaries.txt'):
    """
    Load the Book Summary data and split it into train/dev/test sets
    :param book_path: path to the booksummaries.txt file
    :return: train, dev, test as pandas data frames
    """
    # Fallback paths check
    if not os.path.exists(book_path):
        alt_paths = [
            'BookSummaries/data/booksummaries/booksummaries.txt',
            './BookSummaries/BookSummaries/data/booksummaries/booksummaries.txt',
            'B:/NLP/BookSummaries/BookSummaries/data/booksummaries/booksummaries.txt'
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                book_path = alt
                break

    if not os.path.exists(book_path):
        raise Exception(f"BookSummaries file not found: {book_path}")

    book_df = pd.read_csv(
        book_path,
        sep='\t',
        names=[
            "Wikipedia article ID",
            "Freebase ID",
            "Book title",
            "Author",
            "Publication date",
            "genres",
            "summary"
        ],
        converters={'genres': parse_json_column}
    )

    book_df = book_df.dropna(subset=['genres', 'summary'])  # remove rows missing any genres or summaries
    book_df['word_count'] = book_df['summary'].str.split().str.len()
    book_df = book_df[book_df['word_count'] >= 10]
    
    train = book_df.sample(frac=0.8, random_state=22)
    rest = book_df.drop(train.index)
    dev = rest.sample(frac=0.5, random_state=22)
    test = rest.drop(dev.index)
    return train, dev, test


def prepare_book_summaries(pairs=False, book_path='BookSummaries/BookSummaries/data/booksummaries/booksummaries.txt'):
    """
    Load the Book Summary data and prepare the datasets for Multi-Label classification
    """
    text_set = {'train': [], 'dev': [], 'test': []}
    label_set = {'train': [], 'dev': [], 'test': []}
    train, dev, test = load_booksummaries_data(book_path)

    if not pairs:
        text_set['train'] = train['summary'].tolist()
        text_set['dev'] = dev['summary'].tolist()
        text_set['test'] = test['summary'].tolist()

        train_genres = train['genres'].tolist()
        label_set['train'] = [list(genre.values()) for genre in train_genres]

        dev_genres = dev['genres'].tolist()
        label_set['dev'] = [list(genre.values()) for genre in dev_genres]

        test_genres = test['genres'].tolist()
        label_set['test'] = [list(genre.values()) for genre in test_genres]
    else:
        train_temp = train['summary'].tolist()
        dev_temp = dev['summary'].tolist()
        test_temp = test['summary'].tolist()
        train_genres = train['genres'].tolist()
        train_genres_temp = [list(genre.values()) for genre in train_genres]
        dev_genres = dev['genres'].tolist()
        dev_genres_temp = [list(genre.values()) for genre in dev_genres]
        test_genres = test['genres'].tolist()
        test_genres_temp = [list(genre.values()) for genre in test_genres]

        for i in range(0, len(train_temp) - 1, 2):
            text_set['train'].append(train_temp[i] + train_temp[i + 1])
            label_set['train'].append(list(set(train_genres_temp[i] + train_genres_temp[i + 1])))

        for i in range(0, len(dev_temp) - 1, 2):
            text_set['dev'].append(dev_temp[i] + dev_temp[i + 1])
            label_set['dev'].append(list(set(dev_genres_temp[i] + dev_genres_temp[i + 1])))

        for i in range(0, len(test_temp) - 1, 2):
            text_set['test'].append(test_temp[i] + test_temp[i + 1])
            label_set['test'].append(list(set(test_genres_temp[i] + test_genres_temp[i + 1])))

    vectorized_labels, num_labels, mlb = vectorize_labels(label_set)
    print(f"[BookSummaries] Train: {len(text_set['train'])}, Dev: {len(text_set['dev'])}, Test: {len(text_set['test'])}")
    print(f"[BookSummaries] Total unique genre classes: {num_labels}")
    return text_set, vectorized_labels, num_labels, mlb


def vectorize_labels(all_labels):
    """
    Used for multi-label classification (BookSummaries)
    """
    all_set = []
    for split in all_labels:
        for labels in all_labels[split]:
            all_set.extend(labels)

    all_set = list(set(all_set))
    mlb = MultiLabelBinarizer()
    mlb.fit([all_set])
    num_labels = len(mlb.classes_)
    result = {}
    for split in all_labels:
        result[split] = mlb.transform(all_labels[split])
    return result, num_labels, mlb


# ===========================================================================
# [2] SYMPTOM2DISEASE DATASET LOADER — Single-Label (COMMENTED OUT FOR TOGGLE)
# To switch back to Symptom2Disease, uncomment this function and use it.
# ===========================================================================
# def prepare_symptom_data(csv_path='archive/Symptom2Disease.csv', test_size=0.15, val_size=0.15, random_state=22):
#     """
#     Load the Symptom2Disease dataset and split into train/dev/test sets.
#     Single-label multi-class classification (NOT multi-label).
#     """
#     if not os.path.exists(csv_path):
#         alt_paths = ['/content/Symptom2Disease.csv', 'Symptom2Disease.csv', './archive/Symptom2Disease.csv']
#         for alt in alt_paths:
#             if os.path.exists(alt):
#                 csv_path = alt
#                 break
#
#     if not os.path.exists(csv_path):
#         raise Exception("Data not found: {}".format(csv_path))
#
#     df = pd.read_csv(csv_path)
#     df = df.dropna(subset=['text', 'label'])
#
#     le = preprocessing.LabelEncoder()
#     df['label_id'] = le.fit_transform(df['label'])
#
#     train, temp = train_test_split(
#         df, test_size=(test_size + val_size),
#         random_state=random_state, stratify=df['label_id']
#     )
#     dev, test = train_test_split(
#         temp, test_size=test_size / (test_size + val_size),
#         random_state=random_state, stratify=temp['label_id']
#     )
#
#     text_set = {
#         'train': train['text'].tolist(),
#         'dev': dev['text'].tolist(),
#         'test': test['text'].tolist(),
#     }
#     label_set = {
#         'train': train['label_id'].tolist(),
#         'dev': dev['label_id'].tolist(),
#         'test': test['label_id'].tolist(),
#     }
#     num_labels = len(le.classes_)
#     print(f'Total number of labels: {num_labels}')
#     print(f'Classes: {list(le.classes_)}')
#     print(f"Train/Dev/Test sizes: {len(train)}/{len(dev)}/{len(test)}")
#     return text_set, label_set, num_labels, le


if __name__ == "__main__":
    seed_everything(3456)

    # --- [ACTIVE] Check BookSummaries multi-label loader ---
    text_set, label_set, num_labels, mlb = prepare_book_summaries(pairs=False)
    print("BookSummaries dataloader test passed. Unique classes:", num_labels)

    # --- [SINGLE-LABEL CHECK - COMMENTED] ---
    # text_set, label_set, num_labels, label_encoder = prepare_symptom_data()
    # assert num_labels == 24
    # print("Symptom2Disease dataloader test passed.")
