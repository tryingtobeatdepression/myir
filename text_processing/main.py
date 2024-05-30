from typing import Union
from fastapi import FastAPI, HTTPException
import httpx
from text_processing import get_queries, get_spelling, get_vectorizer, spell_check, suggest_questions

app = FastAPI()

'''
{
    "dataset": string => "webis", "antique",
    "q": string => user input
    "embedding": boolean,
    "clustering": boolean,
}
'''

@app.get('/suggest', status_code=200)
async def suggest(q: Union[str, None] = None):
    if q is None:
        return
    suggested = []
    corr_query = spell_check(q, get_spelling())
    top_res_questions = suggest_questions(
        user_query=corr_query,
        queries=get_queries(),
        vectorizer=get_vectorizer(),
    )
    for query, sim_score in top_res_questions:
        suggested.append(query)   
    return suggested

@app.get('/', status_code=200)
async def text_processing_service(
    q: Union[str, None] = None,
    dataset: str = "webis",
    embedding: bool = False,
    clustering: bool = False,
):  
    indexing_service_url = 'http://localhost:3000'
        
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=indexing_service_url,
            json={
                "data": q,
                "options": {
                    "dataset": dataset,
                    "embedding": embedding,
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
