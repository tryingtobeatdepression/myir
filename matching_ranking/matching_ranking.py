import importlib
importlib.import_module('matching_ranking')
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

def search(query_vector, tfidf_matrix, k=10): 
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    top_indices = np.argpartition(similarities, -k, axis=None)[-k:]

    top_indices_sorted = top_indices[np.argsort(-similarities.ravel()[top_indices])]
    
    # top_results = [(doc(i), key(i), similarities[0, i]) for i in top_indices_sorted]

    top_results = [doc(i) for i in top_indices_sorted]
    return top_results

def word2vec_search(query_vector, corpus_w2v, k=10):
    similarities = cosine_similarity(query_vector, corpus_w2v).flatten()
    top_indices = similarities.argsort()[-k:][::-1]  
    top_indices = [doc(i) for i in top_indices]
    return top_indices

def search_by(
    query_vector,
    dataset: str = 'webis',
    clustering: bool = False,
    embedding: bool = False
):
    top_k_docs = []
    
    top_k_docs = search(
        query_vector=query_vector,
        tfidf_matrix=get_tfidf_matrix(dataset=dataset)
    )
    
    return top_k_docs