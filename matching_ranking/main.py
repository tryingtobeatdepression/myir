from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from matching_ranking import evaluate, create_qrels_inverted
import csv

class Body(BaseModel):
    data: list

app = FastAPI()

@app.post('/')
async def matching(body: Body):
    top_results = body.data
    
    qrels_file = '../webis-touche2020/qrels/test.tsv'
    qrels = create_qrels_inverted(qrels_file)
    
    aps = []
    with open('evaluation.tsv', 'w', newline='', encoding='utf-8') as ef:  
        writer = csv.writer(ef, delimiter='\t')  
        writer.writerow(['Query-id', 'Precision', 'Recall', 'AP'])
        for query_id in qrels.keys():
            p, r, ap = evaluate(top_results, qrels, query_id)
            aps.append(ap)
            writer.writerow([query_id, p, r, ap])
        
    ef.close()
    
    mAP = sum(aps) / len(aps)
    return {
        "data": f"Mean Average Precision (mAP): {mAP}",
    }