import string
import pandas as pd
import numpy as np
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

RUSSIAN_STOPWORDS = ['и', 'в', 'на', 'с', 'по', 'к', 'у', 'о', 'от', 'за', 'из', 'для', 'при', 'без', 'до', 'про',
                     'не', 'да', 'нет', 'бы', 'же', 'ли', 'либо', 'то', 'что', 'как', 'так', 'все', 'это', 'но',
                     'или', 'ибо', 'а', 'ведь', 'вот', 'вон', 'еще', 'уж', 'уже', 'же', 'лишь', 'почти', 'если',
                     'хотя', 'будто', 'словно', 'точно', 'как', 'также', 'тоже', 'ни', 'нибудь', 'что-то', 'кто-то']

def extract_numeric_features(text_series):
    df = pd.DataFrame()
    df['text_len'] = text_series.str.len()
    df['word_count'] = text_series.str.split().str.len()
    df['unique_words'] = text_series.apply(lambda x: len(set(x.split())))
    df['capital_letters'] = text_series.str.findall(r'[A-ZА-Я]').str.len()
    df['digits'] = text_series.str.count(r'\d')
    df['punctuation'] = text_series.apply(lambda x: sum(1 for ch in x if ch in string.punctuation))
    return df

def build_tfidf_vectorizer(max_features=15000, ngram_range=(1,2), min_df=3, sublinear_tf=True):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=RUSSIAN_STOPWORDS,
        min_df=min_df,
        sublinear_tf=sublinear_tf
    )

def transform_text(text_series, vectorizer, batch_size=5000):
    n = len(text_series)
    results = []
    with tqdm(total=(n + batch_size - 1)//batch_size, desc="Transform") as pbar:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = text_series.iloc[start:end]
            transformed = vectorizer.transform(batch)
            results.append(transformed)
            pbar.update(1)
    return vstack(results)

def combine_features(tfidf_matrix, numeric_df):
    return hstack([tfidf_matrix, numeric_df.values])
