import importlib
importlib.import_module('matching_ranking')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from lib import doc, get_tfidf_matrix, get_kmeans

def get_topic_tags(kmeans: KMeans, terms: list, lab_idx, top_n=5):
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    return [str(terms[idx]) for idx in order_centroids[lab_idx, :top_n]]

def get_cluster_docs_indicies(kmeans: KMeans, k=3) -> dict:
    indicies = {i: [] for i in range(k)}
    for i, label in enumerate(kmeans.labels_):
        indicies[label].append(i)
    return indicies

def search(query_vector, tfidf_matrix, dataset, k=10): 
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    top_indices = np.argpartition(similarities, -k, axis=None)[-k:]
    top_indices_sorted = top_indices[np.argsort(-similarities.ravel()[top_indices])]
    top_results = [doc(i, dataset=dataset) for i in top_indices_sorted]
    return top_results

def search_by_cluster(user_query_vec, kmeans: KMeans, dataset: str, tfidf_matrix, top_k=10):
    query_cluster_lb = kmeans.predict(user_query_vec)[0]
    clustered_indicies = get_cluster_docs_indicies(kmeans)
    cluster_docs_indices = clustered_indicies[query_cluster_lb]
    retrieved_docs_vecs = tfidf_matrix[cluster_docs_indices]
    similarities = cosine_similarity(user_query_vec, retrieved_docs_vecs).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1] 
    top_results = [doc(i, dataset=dataset) for i in top_indices]
    return top_results

def search_by(
    query_vector, dataset: str = 'touche', clustering: bool = False,
):
    top_k_docs = []
    if clustering:
        top_k_docs = search_by_cluster(
            user_query_vec=query_vector,
            dataset=dataset,
            kmeans=get_kmeans(dataset=dataset),
            tfidf_matrix=get_tfidf_matrix(dataset=dataset),
        )
    else:
        top_k_docs = search(
            query_vector=query_vector,
            tfidf_matrix=get_tfidf_matrix(dataset=dataset),
            dataset=dataset,
        )
    return top_k_docs