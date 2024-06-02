from typing import Union
from fastapi import FastAPI, HTTPException
import httpx
from lib import get_queries, get_spelling, get_vectorizer
from text_processing import spell_check, suggest_questions

app = FastAPI()

@app.get('/suggest', status_code=200)
async def suggest(
    q: Union[str, None] = None,
    dataset: str = 'touche'
):
    if q is None: 
        return None
    corr_query = spell_check(q, get_spelling(dataset))
    top_res_questions = suggest_questions(
        user_query=corr_query,
        queries=get_queries(dataset),
        vectorizer=get_vectorizer(dataset),
    )
    
    return [query for query in top_res_questions]

@app.get('/', status_code=200)
async def text_processing_service(
    q: Union[str, None] = None,
    dataset: str = "touche",
    clustering: bool = False,
):  
    indexing_service_url = 'http://localhost:3000'
        
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=indexing_service_url,
            json={
                "query": q,
                "options": {
                    "dataset": dataset,
                    "clustering": clustering,
                }
            },
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code= response.status_code,
            detail="Error calling indexing service!",
        )
        
    res = response.json()
    return res['data']

# json= {"data": user_query_vector.toarray().tolist()},
