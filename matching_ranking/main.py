from fastapi import FastAPI
from pydantic import BaseModel
from matching_ranking import get_tfidf_matrix, search, search_by

class Options(BaseModel):
    dataset: str
    embedding: bool
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
        embedding=options.embedding,
        clustering=options.clustering
    )

    return {
        "data": top_k_docs,
    }