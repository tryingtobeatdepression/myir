import importlib
importlib.import_module('text_processing')

from textblob.en import Spelling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def suggest_questions(query: str, queries: dict, vectorizer: TfidfVectorizer, k=2):
    qs = list(queries.values())
    user_query_vector = vectorizer.transform([query])
    queries_matrix = vectorizer.transform(qs)
    similarities = cosine_similarity(user_query_vector, queries_matrix)
    top_indices = np.argpartition(similarities, -k, axis=None)[-k:]
    top_results = [qs[i] for i in top_indices]
    return top_results

def spell_check(query: str, spelling: Spelling):
    return spelling.suggest(query)[0][0]

