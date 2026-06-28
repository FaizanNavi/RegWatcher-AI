import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any

from app.agents.orchestrator import RegWatcherOrchestrator
from app.utils.config import HOST, PORT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RegWatcher AI Backend",
    description="Multi-agent orchestrator for US Federal Register data querying using LangGraph.",
    version="1.0.0"
)

# Enable CORS for the frontend app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = RegWatcherOrchestrator()

class ChatRequest(BaseModel):
    message: str = Field(..., description="Query/message to send to the orchestrator")
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Find recent documents by the EPA in 2024"
            }
        }

@app.get("/health")
async def health():
    return {"status": "ok", "service": "regwatcher-ai-orchestrator"}

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Query message cannot be empty")
    
    logger.info(f"Received query: {request.message}")
    try:
        res = orchestrator.run(request.message)
        return {
            "response": res.get("final_response", ""),
            "route": res.get("route", ""),
            "critic_score": res.get("critic_score", 0),
            "critic_iterations": res.get("critic_iterations", 0),
            "is_valid": res.get("is_valid", False),
            "validation_issues": res.get("validation_issues", []),
            "retrieved_chunks": res.get("retrieved_chunks", []),
        }
    except Exception as e:
        logger.error(f"Error during orchestrator execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
