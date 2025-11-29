from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schemas import QueryRequest, QueryResponse, AnalyzeRequest, ContractSchema, HealthResponse
from app.rag_service import RAGService
from app.contract_analyzer import ContractAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
rag_service = None
contract_analyzer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    Initializes RAGService and ContractAnalyzer on startup.
    """
    global rag_service, contract_analyzer
    logger.info("Initializing RAG Service...")
    rag_service = RAGService()
    contract_analyzer = ContractAnalyzer(rag_service)
    yield
    # Clean up resources
    logger.info("Shutting down...")

app = FastAPI(title="Compliance & Contract AI Assistant", lifespan=lifespan)

@app.get("/")
def read_root():
    """Root endpoint to verify API status."""
    return {"message": "Compliance & Contract AI Assistant is running. Visit /docs for API documentation."}

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint to verify if LLM services are loaded."""
    status = "healthy" if rag_service and rag_service.pipe else "degraded (LLM not loaded)"
    return {"status": status}

@app.post("/query", response_model=QueryResponse)
def query_compliance(req: QueryRequest):
    """
    Endpoint for Compliance Q&A.
    Uses RAG to retrieve relevant policy chunks and generate an answer.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = rag_service.answer_question(req.question)
    return response

@app.post("/contract/analyze", response_model=ContractSchema)
def analyze_contract(req: AnalyzeRequest):
    """
    Endpoint for Contract Analysis.
    Retrieves relevant contract sections and extracts structured data using LLM.
    """
    if not contract_analyzer:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    result = contract_analyzer.analyze(req.file_name)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=False)

