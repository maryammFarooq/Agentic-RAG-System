"""
FastAPI backend server for serving RAG pipeline results to the frontend UI.
Handles loading the global index and routing queries to the correct pipeline.
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# These imports will work perfectly as long as we run uvicorn from the root folder
from src.corpus import load_index
from src.pipelines.rag_fusion import run_rag_fusion
from src.pipelines.hyde import run_hyde
from src.pipelines.crag import run_crag
from src.pipelines.graph_rag import run_graph_rag

app = FastAPI(title="RAG Evaluation API")  # Main FastAPI app instance

# Enable CORS so the React frontend can access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variable for the loaded index
GLOBAL_INDEX = None
# Path to the pickled index file
INDEX_FILE_PATH = "dataset/global_index.pkl"


# Load the global index into memory at server startup
@app.on_event("startup")
def on_startup():
    """Load the global index into memory when the server starts."""
    global GLOBAL_INDEX
    if os.path.exists(INDEX_FILE_PATH):
        print(f"Loading global index from {INDEX_FILE_PATH} into memory...")
        GLOBAL_INDEX = load_index(INDEX_FILE_PATH)
    else:
        print(f"Warning: Index not found at {INDEX_FILE_PATH}. Please run run_evaluation.py first.")


# Request body model for /query endpoint
class QueryRequest(BaseModel):
    query: str  # User's question
    pipeline: str  # Pipeline name

@app.post("/query")
def process_query(request: QueryRequest):
    """
    API endpoint to process a user query using the selected RAG pipeline.
    """
    if GLOBAL_INDEX is None:
        raise HTTPException(status_code=500, detail="Global index not loaded.")

    user_query = request.query
    pipeline_name = request.pipeline

    try:
        # Route to the correct pipeline based on the user's selection
        if pipeline_name == "rag_fusion":
            ans, ctx, score = run_rag_fusion(user_query, GLOBAL_INDEX)
        elif pipeline == "hyde":
            ans, ctx, score = run_hyde(query, INDEX)
        elif pipeline == "crag":
            ans, ctx, score = run_crag(query, INDEX)
        elif pipeline == "graph_rag":
            ans, ctx, score = run_graph_rag(query, INDEX)
        else:
            raise HTTPException(status_code=400, detail="Invalid pipeline selected.")
            
        return {
            "answer": ans,
            "context": ctx,
            "score": score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)