from typing import Union
from fastapi import FastAPI, HTTPException
import httpx
import dill
from sklearn.feature_extraction.text import TfidfVectorizer
import json

def get_vectorizer_instance() -> TfidfVectorizer: 
    with open('model.pkl', 'rb') as mf:
        vectorizer: TfidfVectorizer = dill.load(mf)
    mf.close() 
    return vectorizer

app = FastAPI(
    title="Text Processing"
)

@app.get('/', )
async def text_processing_service(q: Union[str, None] = None):  
     
    user_query_vector = get_vectorizer_instance().transform([q])
    indexing_service_url = 'http://localhost:3000'
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=indexing_service_url,
            json= {"data": user_query_vector.toarray().tolist()},
            timeout=120
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code= response.status_code,
            detail="Error calling indexing service!",
        )
        
    res = response.json()
    return res['data']
