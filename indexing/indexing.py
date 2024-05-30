import importlib
importlib.import_module('indexing')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import dill

matrix_file = '../text_processing/matrix.pkl'
model_file = 'model.pkl'
doc_i_file = '../../datafiles/docs/doc'
key_i_file = '../../datafiles/keys/key'

def get_vectorizer(
    dataset: str = 'webis',
    pkl_file_path: str = model_file
) -> TfidfVectorizer:
    # TODO: Dataset choice
    with open(f'{pkl_file_path}', 'rb') as f:
        vectorizer: TfidfVectorizer = dill.load(f)
    f.close()
    return vectorizer

def get_tfidf_matrix(
    dataset: str = 'webis',
    pkl_file_path: str = matrix_file
):
    with open(pkl_file_path, 'rb') as mf:
        tfidf_matrix = dill.load(mf)
    mf.close()
    return tfidf_matrix

def doc(i, file_path: str = doc_i_file):
    with open(f'{file_path}{i}', 'r', encoding='utf-8') as f:
        doc = f.read()
    f.close()
    return doc

def key(i, file_path: str = key_i_file):
    with open(f'{file_path}{i}', 'r', encoding='utf-8') as f:
        key = f.read()
    f.close()
    return key

def search(user_query, tfidf_matrix, k=10): 
    user_query_vector = get_vectorizer().transform([user_query])
    similarities = cosine_similarity(user_query_vector, tfidf_matrix)
    top_indices = np.argpartition(similarities, -k, axis=None)[-k:]

    top_indices_sorted = top_indices[np.argsort(-similarities.ravel()[top_indices])]

    top_results = [(doc(i), key(i), similarities[0, i]) for i in top_indices_sorted]
    return top_results