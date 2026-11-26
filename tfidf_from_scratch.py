# TF-IDF с нуля (исправленная версия mentor_tr.py)
#
# TF(t, d)  = count(t, d) / len(d)                  — нормированная частота термина в документе
# DF(t)     = число документов корпуса, где встречается t
# IDF(t)    = ln((N + 1) / (DF(t) + 1)) + 1          — сглаженный IDF (как в sklearn, smooth_idf=True)
# TF-IDF    = TF(t, d) * IDF(t)                      — считается для КАЖДОГО термина отдельно
#
# Запуск: python tfidf_from_scratch.py /path/to/corpus

import math
import os
import random
import re
import sys
from collections import Counter

CORPUS_DIR = "/path/to/corpus"  # переопределяется внутри __main__


# --- Корпус ---------------------------------------------------------------

def load_corpus(corpus_dir):
    """Рекурсивно собирает все .txt файлы и возвращает {путь: текст}."""
    files = [
        os.path.join(root, name)
        for root, _, names in os.walk(corpus_dir)
        for name in names
        if name.endswith(".txt")
    ]
    corpus = {}
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            corpus[path] = f.read()
    return corpus


# --- Предобработка --------------------------------------------------------
# Важно: документ и корпус обрабатываются ОДНОЙ И ТОЙ ЖЕ функцией,
# иначе леммы из TF не совпадут с терминами, по которым считался DF.

try:
    import spacy

    # Парсер и NER для лемматизации не нужны — отключаем, так в разы быстрее
    nlp = spacy.load("ru_core_news_sm", disable=["parser", "ner"])

    def preprocess_many(texts):
        return [
            [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]
            for doc in nlp.pipe(texts)
        ]
except (ImportError, OSError):
    # Запасной вариант, если spacy / модель не установлены: без лемматизации
    def preprocess_many(texts):
        return [re.findall(r"[^\W\d_]+", text.lower()) for text in texts]


# --- TF-IDF ---------------------------------------------------------------

def compute_tf(tokens):
    if not tokens:
        return {}
    total = len(tokens)
    return {term: count / total for term, count in Counter(tokens).items()}


def compute_df(tokenized_corpus):
    # set(): термин учитывается в документе один раз, сколько бы раз он там ни встретился
    df = Counter()
    for tokens in tokenized_corpus:
        df.update(set(tokens))
    return df


def compute_idf(df, n_docs):
    return {term: math.log((n_docs + 1) / (freq + 1)) + 1 for term, freq in df.items()}


def compute_tfidf(tf, idf, n_docs):
    """n_docs должен совпадать с тем, что передавался в compute_idf."""
    # OOV-термин (DF=0) получает ln(N+1)+1 — максимально возможный IDF
    default_idf = math.log(n_docs + 1) + 1
    return {term: tf_val * idf.get(term, default_idf) for term, tf_val in tf.items()}


if __name__ == "__main__":
    CORPUS_DIR = sys.argv[1] if len(sys.argv) > 1 else "/path/to/corpus"
    corpus = load_corpus(CORPUS_DIR)
    n_docs = len(corpus)
    print(f"Документов: {n_docs}")
    if n_docs == 0:
        sys.exit(f"В {CORPUS_DIR} нет .txt файлов")

    paths = list(corpus)
    tokenized_corpus = preprocess_many(corpus[p] for p in paths)

    df = compute_df(tokenized_corpus)
    idf = compute_idf(df, n_docs)

    # Берём случайный документ из корпуса
    random.seed(42)
    i = random.randrange(n_docs)
    print(f"Выбран: {paths[i]}")
    tokens = tokenized_corpus[i]
    print(f"Токенов после предобработки: {len(tokens)}")

    tf = compute_tf(tokens)
    tf_idf = compute_tfidf(tf, idf, n_docs)

    print("\nТоп-10 терминов по TF-IDF:")
    for term, score in sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{term:20s} tf={tf[term]:.4f}  idf={idf[term]:.4f}  tf-idf={score:.4f}")
