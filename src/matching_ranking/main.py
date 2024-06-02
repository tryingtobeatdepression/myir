from fastapi import FastAPI
from pydantic import BaseModel
from matching_ranking import search_by

class Options(BaseModel):
    dataset: str
    clustering: bool

class Body(BaseModel):
    query_vector: list
    options: Options

app = FastAPI()

@app.post('/')
async def matching(body: Body):
    query_vector = body.query_vector
    options = body.options

    top_k_docs = search_by(
        query_vector=query_vector,
        dataset=options.dataset,
        clustering=options.clustering
    )

    return {
        "data": top_k_docs,
    }