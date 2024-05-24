from fastapi import FastAPI, HTTPException
import httpx
from pydantic import BaseModel
import typing
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import dill 
from scipy.sparse import csr_matrix
import json


def get_tfidf_matrix():
    with open('../text-processing/matrix.pkl', 'rb') as mf:
        tfidf_matrix = dill.load(mf)
    mf.close()
    return tfidf_matrix

def doc(i):
    with open(f'../../datafiles/docs/doc{i}', 'r', encoding='utf-8') as f:
        doc = f.read()
    f.close()
    return doc

def key(i):
    with open(f'../../datafiles/keys/key{i}', 'r', encoding='utf-8') as f:
        key = f.read()
    f.close()
    return key

def find_top_k_results(user_query_vector, tfidf_matrix, k=10): 
    similarites = cosine_similarity(user_query_vector, tfidf_matrix)
    top_indices = np.argpartition(similarites, -k, axis=None)[-k:]
    
    top_indices_sorted = top_indices[np.argsort(similarites.ravel()[top_indices])]
    
    top_results = [(doc(i), key(i), similarites[0, i]) for i in top_indices_sorted]    
    return top_results

class Body(BaseModel):
    data: list

app = FastAPI()

@app.post('/')
async def indexing(body: Body):
    # top_results = find_top_k_results()
    
    user_query_vector = body.data
    tfidf_matrix = get_tfidf_matrix()
    
    top_results = find_top_k_results(user_query_vector, tfidf_matrix)
        
    matching_service_url = 'http://localhost:3500'
    
    with open('results.txt', 'w', encoding='utf-8') as f:
        for doc, doc_id, similarity_score in top_results:
            f.write(f'DocId: {doc_id}\n')
    f.close()
            
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=matching_service_url,
            json={"data": top_results },
            timeout=120
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code= response.status_code,
            detail="Error calling matching & ranking service!",
        )
    
    res = response.json()
    
    return {
        "data": [res['data'], [doc for doc, did, sm in top_results]],
    }