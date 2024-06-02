import importlib
importlib.import_module('indexing')

from lib import get_vectorizer

def transform_by(
    query: str,
    dataset: str = 'touche',
    clustering: bool = False,
):
    query_vector = get_vectorizer(dataset=dataset).transform([query])
    return query_vector

