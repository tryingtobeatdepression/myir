from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from indexing import transform_by
import httpx

class Options(BaseModel):
    dataset: str
    clustering: bool

class Body(BaseModel):
    query: str
    options: Options

app = FastAPI()
matching_service_url = 'http://localhost:3500'


@app.post('/')
async def indexing(body: Body):    
    query = body.query
    options = body.options
    
    query_vector = transform_by(
        query=query,
        dataset=options.dataset,
    )
            
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