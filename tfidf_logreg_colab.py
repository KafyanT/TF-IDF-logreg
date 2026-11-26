# Colab script — запускать в Google Colab, не как обычный Python-скрипт.
# pip install -q pyarrow

import gc
import shutil
import sys

import numpy as np
import pandas as pd
from google.colab import drive
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

drive.mount('/content/drive')

DRIVE    = '/content/drive/MyDrive'
SEED     = 42
N_FOLDS  = 5
MAX_FEAT = 50_000
TARGET   = 'item_contact'

src = f'{DRIVE}/tfidf_from_scratch.py'
if not __import__('os').path.exists(src):
    raise FileNotFoundError(f'Положи tfidf_from_scratch.py в корень Google Drive ({src})')
shutil.copy(src, '/content/tfidf_from_scratch.py')


def load_parquet(path, columns):
    return pd.read_parquet(path, columns=columns)


train_cols = ['query_id', 'item_id', 'query_text', 'item_title', TARGET]
test_cols  = ['query_id', 'item_id', 'query_text', 'item_title']

train = load_parquet(f'{DRIVE}/train-dset.parquet', train_cols)
test  = load_parquet(f'{DRIVE}/test-dset-small.parquet', test_cols)

print(f'train: {train.shape},  positive rate: {train[TARGET].mean():.4f}')
print(f'test:  {test.shape}')
train.head(3)


sys.path.insert(0, '/content')
from tfidf_from_scratch import (
    preprocess_many, compute_tf, compute_df, compute_idf, compute_tfidf
)

def make_text(df):
    return df['query_text'].fillna('') + ' ' + df['item_title'].fillna('')

# Демо: TF-IDF с нуля на 5 объявлениях из train.
# IDF считается по этим 5 документам (N=5) — только для иллюстрации формул,
# цифры не сопоставимы с продакшн-моделью (N ≈ 7.8M).
# Токенизатор: spacy-лемматизация или regex (tfidf_from_scratch.preprocess_many),
# в отличие от CountVectorizer в основном пайплайне — результаты демо не
# воспроизводят признаки модели.
sample = train.sample(5, random_state=SEED)[['query_text', 'item_title', TARGET]]
docs   = make_text(sample).tolist()

tokenized = preprocess_many(docs)
df_dict   = compute_df(tokenized)
idf_dict  = compute_idf(df_dict, len(tokenized))

print('Топ-5 терминов по TF-IDF для каждого объявления:\n')
for i, tokens in enumerate(tokenized):
    tf     = compute_tf(tokens)
    tfidf  = compute_tfidf(tf, idf_dict, len(tokenized))
    top5   = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:5]
    label  = int(sample[TARGET].iloc[i])
    print(f'[contact={label}] {docs[i][:70]}')
    print(f'  {top5}\n')


def build_tfidf(tr_texts, eval_texts, max_features=MAX_FEAT):
    """TF = count / sum_vocab_counts, IDF = ln((N+1)/(DF+1))+1.

    L1-нормировка строки CountVectorizer делит на сумму счётчиков словарных
    токенов (OOV исключены), что близко, но не тождественно формуле
    tfidf_from_scratch (там делитель — все токены документа).
    Одна функция для CV и для финального train→test — гарантирует идентичный пайплайн.
    """
    cv    = CountVectorizer(max_features=max_features)
    trans = TfidfTransformer(smooth_idf=True, norm=None)
    X_tr   = trans.fit_transform(normalize(cv.fit_transform(tr_texts), norm='l1'))
    X_eval = trans.transform(normalize(cv.transform(eval_texts), norm='l1'))
    return X_tr, X_eval


def compute_ndcg(df, pred_col='pred', label_col=TARGET, k=10):
    """Средний NDCG@k по запросам с хотя бы одним позитивом.

    Реализация соответствует ndcg_at_k из основного ноутбука:
    один tie-вектор rng.random(len(df)) для всего df до группировки.
    """
    tie = np.random.default_rng(SEED).random(len(df))
    df  = df.assign(_tie=tie)
    scores = []
    for _, grp in df.groupby('query_id', sort=False):
        if grp[label_col].sum() == 0:
            continue
        idx    = np.lexsort((grp['_tie'].values, -grp[pred_col].values))[:k]
        gains  = grp[label_col].values[idx].astype(float)
        disc   = np.log2(np.arange(2, len(gains) + 2))
        dcg    = (gains / disc).sum()
        ideal  = np.sort(grp[label_col].values)[::-1][:k].astype(float)
        idcg   = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
        if idcg > 0:
            scores.append(dcg / idcg)
    assert scores, 'нет запросов с позитивами в переданной выборке'
    return np.mean(scores)


# Назначение фолдов — точно как в основном ноутбуке:
# только запросы с хотя бы одним позитивом, отсортированные, тот же SEED.
# Запросы без позитивов получают fold=NaN → попадают в каждый train-фолд,
# но никогда не в validation (что соответствует df.filter(n_pos > 0) в основном ноутбуке).
pos_query_ids = (train.groupby('query_id')[TARGET]
                      .sum()
                      .loc[lambda s: s > 0]
                      .index)
unique_q  = np.sort(pos_query_ids)
fold_arr  = np.random.default_rng(SEED).integers(0, N_FOLDS, len(unique_q))
fold_map  = dict(zip(unique_q, fold_arr))
train['fold'] = train['query_id'].map(fold_map)

train['pred'] = 0.0
fold_scores = []

for fold in range(N_FOLDS):
    tr_idx  = train.index[train['fold'] != fold]
    val_idx = train.index[train['fold'] == fold]
    tr, val = train.loc[tr_idx], train.loc[val_idx]

    X_tr, X_val = build_tfidf(make_text(tr), make_text(val))

    lr = LogisticRegression(C=1.0, max_iter=500, solver='saga', random_state=SEED)
    lr.fit(X_tr, tr[TARGET].values)
    assert lr.classes_[1] == 1, f'classes_={lr.classes_}'

    train.loc[val_idx, 'pred'] = lr.predict_proba(X_val)[:, 1]
    del X_tr, X_val; gc.collect()

    score = compute_ndcg(train.loc[val_idx])
    fold_scores.append(score)
    print(f'Fold {fold}: NDCG@10 = {score:.4f}')

# OOF = compute_ndcg на полном train (не среднее фолдов, хотя обычно совпадают до 4 знаков)
oof = compute_ndcg(train)
print(f'\nOOF NDCG@10 = {oof:.4f}  (mean folds = {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f})')


X_full, X_test = build_tfidf(make_text(train), make_text(test))

lr = LogisticRegression(C=1.0, max_iter=500, solver='saga', random_state=SEED)
lr.fit(X_full, train[TARGET].values)
assert lr.classes_[1] == 1, f'classes_={lr.classes_}'

test = test.copy()
test['score'] = lr.predict_proba(X_test)[:, 1]

solution = (
    test.sort_values(['query_id', 'score'], ascending=[True, False])
    [['query_id', 'item_id']]
    .reset_index(drop=True)
)

assert len(solution) == len(test), "часть строк test потеряна в solution"
assert solution.duplicated(['query_id', 'item_id']).sum() == 0, "дублирующиеся пары (query_id, item_id) в solution"

out = f'{DRIVE}/solution_tfidf_logreg.csv'
solution.to_csv(out, index=False)
print(f'Сохранено: {out}  ({len(solution):,} строк)')
solution.head()
