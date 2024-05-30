import importlib
importlib.import_module('indexing')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import dill
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import normalize

matrix_file = '../text_processing/matrix.pkl'
model_file = 'model.pkl'
w2v_model_file = 'w2v_model.pkl'
feature_names_file = 'feature_names.txt'
doc_i_file = '../../datafiles/docs/doc'
key_i_file = '../../datafiles/keys/key'

def get_vectorizer(
    dataset: str = 'webis', pkl_file_path: str = model_file
) -> TfidfVectorizer:
    # TODO: Dataset choice
    with open(f'{pkl_file_path}', 'rb') as f:
        vectorizer: TfidfVectorizer = dill.load(f)
    f.close()
    return vectorizer

def get_tfidf_matrix(
    dataset: str = 'webis', pkl_file_path: str = matrix_file
):
    with open(pkl_file_path, 'rb') as mf:
        tfidf_matrix = dill.load(mf)
    mf.close()
    return tfidf_matrix

def get_w2v_model(
    dataset: str = 'webis', pkl_file_path: str = w2v_model_file,
) -> Word2Vec:
    with open(pkl_file_path, 'rb') as f:
        word2vec_model: Word2Vec = dill.load(f)
    f.close()
    return word2vec_model

def get_feature_names(
    dataset: str = 'webis', file_path: str = feature_names_file,
) -> list: 
    with open(file_path, 'r', encoding='utf-8') as f:
        feature_names = f.read()
    return feature_names

def get_word_to_index() -> dict:
    return {word: idx for idx, word in enumerate(get_feature_names())}

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

def tfidf_weighted_word2vec(
    doc,
    vectorizer: TfidfVectorizer,
    word2vec_model: Word2Vec,
    word_to_index
):
    words = word_tokenize(doc)
    tfidf_scores = vectorizer.transform([doc])
    
    weighted_word2vec = np.zeros(word2vec_model.vector_size)
    for word in words:
        if word in word_to_index and word in word2vec_model.wv:
            tfidf_score = tfidf_scores[0, word_to_index[word]]
            word_vector = word2vec_model.wv[word]
            weighted_word2vec += tfidf_score * word_vector
            
    return weighted_word2vec

def transform_by(
    query: str,
    dataset: str = 'webis',
    embedding: bool = False,
):
    query_vector = None
    if embedding:
        query_vector = tfidf_weighted_word2vec(
            doc=query,
            vectorizer=get_vectorizer(dataset=dataset),
            word2vec_model=get_w2v_model(),
            word_to_index=get_word_to_index(),
        )
        query_vector = normalize(query_vector.reshape(1, -1))
    else:
        query_vector = get_vectorizer(dataset=dataset).transform([query])

    return query_vector

