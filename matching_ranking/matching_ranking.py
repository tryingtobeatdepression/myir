import importlib
from typing import List

def create_qrels_inverted(qrels_file):
    inverted_index = {}
    with open(qrels_file, 'r') as f:
        next(f) # Skip first line of the file
        for line in f:
            query_id, document_id, score = line.strip().split('\t')
            score = int(score)
            query_id = int(query_id)
            if query_id not in inverted_index:
                inverted_index[query_id] = []
            inverted_index[query_id].append((document_id, score))
                    
    # Sort index in descending order
    for query_id, doc_scores in inverted_index.items():
        inverted_index[query_id] = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        
    return inverted_index

def get_relevant_docs_for_query_i(qrels, query_id, k=10):
    relevant_docs = set()
    # for query_id, doc_scores in zip(qrels.keys(), qrels.values()):
    for doc_id, score in qrels[query_id]:
        relevant_docs.add(doc_id)
    return relevant_docs

def calculate_average_precision(relevant_docs, retrieved_docs, k=10): 
    precision_sum = 0
    true_positives_at_k = 0
    for i, retrieved_doc in enumerate(retrieved_docs, start=1):
        if retrieved_doc in relevant_docs:
            true_positives_at_k += 1
            precision_sum += true_positives_at_k / i
    return precision_sum / true_positives_at_k if true_positives_at_k > 0 else 0
        
def evaluate(top_results: List, qrels, query_id: str, k=10):
    relevant_docs = get_relevant_docs_for_query_i(qrels, query_id)
    retrieved_docs = set(result[1] for result in top_results)
   
    true_positives = retrieved_docs.intersection(relevant_docs)
    
    # Precision
    precision = len(true_positives) / k if k > 0 else 0
    # Recall 
    recall = len(true_positives)/ len(relevant_docs) if len(relevant_docs) > 0 else 0
    # Average Precision
    ap = calculate_average_precision(relevant_docs, retrieved_docs)
    
    return precision, recall, ap

importlib.import_module('matching_ranking')