from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from indexing import get_tfidf_matrix, get_vectorizer, search
import numpy as np
import httpx
import dill 


class Options(BaseModel):
    dataset: str
    embedding: bool
    clustering: bool

class Body(BaseModel):
    data: str
    options: Options

app = FastAPI()
matching_service_url = 'http://localhost:3500'


@app.post('/')
async def indexing(body: Body):    
    query = body.data
    options = body.options
    
    query_vector = get_vectorizer(dataset=options.dataset).transform([query])
            
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=matching_service_url,
            json={
                "query_vector": query_vector.toarray().tolist(),
                "options": options
            },
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail="Error calling matching & ranking service!",
        )
    
    res = response.json()
    
    return {
        "data": res['data'],
    }