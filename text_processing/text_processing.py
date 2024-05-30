import importlib
importlib.import_module('text_processing')

from textblob.en import Spelling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import dill
from textblob.en import Spelling

queries_file = 'queries.pkl'
speller_file = 'speller.pkl'
model_file = 'model.pkl'

def get_vectorizer(pkl_file_path: str = model_file) -> TfidfVectorizer:
    with open(pkl_file_path, 'rb', encoding='utf-8') as f:
        vectorizer: TfidfVectorizer = dill.load(f)
    f.close()
    return vectorizer

def get_spelling(pkl_file_path: str = speller_file) -> Spelling:
    with open(pkl_file_path, 'rb', encoding='utf-8') as f:
        speller: Spelling = dill.load(f)
    f.close()
    return speller

def get_queries(pkl_file_path: str = queries_file) -> dict:
    with open(pkl_file_path, 'rb', encoding='utf-8') as f:
        queries: dict = dill.load(f)
    f.close()
    return queries

def suggest_questions(
    user_query: str, queries: dict, vectorizer: TfidfVectorizer,
    k=2
):
    qs = list(queries.values())
    user_query_vector = vectorizer.transform([user_query])
    queries_matrix = vectorizer.transform(qs)
    similarities = cosine_similarity(user_query_vector, queries_matrix)
    top_indices = np.argpartition(similarities, -k, axis=None)[-k:]
    top_results = [(qs[i], similarities[0, i]) for i in top_indices]
    return top_results

def spell_check(user_query, spelling: Spelling):
    return spelling.suggest(user_query)[0][0]

