import importlib
importlib.import_module('lib')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob.en import Spelling
import dill

# DATASETS CONFIGURATION
antique_path = '../../datasets/antique/antique_'
touche_path = '../../datasets/touche/touche_'
doc_path = '-datafiles/docs/doc'
key_path = '-datafiles/keys/key'

# ANTIQUE DATASET
antique_model_file_path = f'{antique_path}model.pkl'
antique_matrix_file_path = f'{antique_path}matrix.pkl'
antique_spelling_file_path = f'{antique_path}spelling.pkl'
antique_kmeans_file_path = f'{antique_path}kmeans.pkl'
antique_terms_file_path = f'{antique_path}terms.pkl'
antique_queries_file_path = f'{antique_path}queries.tsv'
antique_doc_i_file = f'../../../antique{doc_path}'
antique_key_i_file = f'../../../antique{key_path}'

# TOUCHE DATASET    
touche_model_file_path = f'{touche_path}model.pkl'
touche_matrix_file_path = f'{touche_path}matrix.pkl'
touche_spelling_file_path = f'{touche_path}spelling.pkl'
touche_kmeans_file_path = f'{touche_path}kmeans.pkl'
touche_terms_file_path = f'{touche_path}terms.pkl'
touche_queries_file_path = f'{touche_path}queries.tsv'
touche_doc_i_file = f'../../../touche{doc_path}'
touche_key_i_file = f'../../../touche{key_path}'

# FUNCTIONS #

def get_vectorizer(dataset: str = 'touche') -> TfidfVectorizer:
    file_path = touche_model_file_path if dataset=='touche' else antique_model_file_path
    with open(file_path, 'rb') as f:
        vectorizer: TfidfVectorizer = dill.load(f)
    f.close()
    return vectorizer

def get_tfidf_matrix(dataset: str = 'touche'):
    file_path = touche_matrix_file_path if dataset=='touche' else antique_matrix_file_path
    with open(file_path, 'rb') as mf:
        tfidf_matrix = dill.load(mf)
    mf.close()
    return tfidf_matrix

def get_terms(dataset: str = 'touche') -> list:
    file_path = touche_terms_file_path if dataset=='touche' else antique_terms_file_path
    with open(file_path, 'rb') as f:
        terms = dill.load(f)
    f.close()
    return terms

def doc(i, dataset: str = 'touche'):
    file_path = touche_doc_i_file if dataset=='touche' else antique_doc_i_file
    with open(f'{file_path}{i}', 'r', encoding='utf-8') as f:
        doc = f.read()
    f.close()
    return doc

def key(i, dataset: str = 'touche'):
    file_path = touche_key_i_file if dataset=='touche' else antique_key_i_file
    with open(f'{file_path}{i}', 'r', encoding='utf-8') as f:
        key = f.read()
    f.close()
    return key

def get_spelling(dataset: str = 'touche') -> Spelling:
    file_path = touche_spelling_file_path if dataset=='touche' else antique_spelling_file_path
    with open(file_path, 'rb') as f:
        speller: Spelling = dill.load(f)
    f.close()
    return speller

def get_queries(dataset: str = 'touche') -> dict:
    file_path = touche_queries_file_path if dataset=='touche' else antique_queries_file_path
    inverted_index = {}
    with open(file_path, 'r') as f:
        next(f) # Skip first line of the file
        for line in f:
            query_id, text = line.strip().split('\t')
            inverted_index[query_id] = text
    return inverted_index

def get_kmeans(dataset: str = 'touche') -> KMeans:
    file_path = touche_kmeans_file_path if dataset=='touche' else antique_kmeans_file_path
    with open(file_path, 'rb') as f:
        kmeans: KMeans = dill.load(f)
    f.close()
    return kmeans

def get_terms(dataset: str = 'touche') -> list:
    file_path = touche_terms_file_path if dataset=='touche' else antique_terms_file_path
    with open(file_path, 'rb') as f:
        terms: list = dill.load(f)
    f.close()
    return terms