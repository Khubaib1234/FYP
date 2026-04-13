from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import os
from retriever import HybridRetriever
from dotenv import load_dotenv

router = APIRouter()
load_dotenv()

# Setup config
FAISS_INDEX_DIR = "FAISS_Index"
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, 'candidate_index.faiss')
SNIPPET_METADATA_PATH = os.path.join(FAISS_INDEX_DIR, 'snippet_metadata.json')
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "khubaib123")

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

@router.post("/search")
async def search_candidates(query_data: SearchQuery):
    if not os.path.exists(FAISS_INDEX_PATH):
        raise HTTPException(status_code=500, detail="FAISS index not found. Please upload resumes first.")
    if not os.path.exists(SNIPPET_METADATA_PATH):
        raise HTTPException(status_code=500, detail="Snippet metadata not found. Please upload resumes first.")
        
    try:
        retriever = HybridRetriever(
            faiss_index_path=FAISS_INDEX_PATH,
            snippet_metadata_path=SNIPPET_METADATA_PATH,
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_pass=NEO4J_PASSWORD
        )
        
        results = retriever.retrieve(query_data.query, top_k=query_data.top_k)
        return {"query": query_data.query, "results": results}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
