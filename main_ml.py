import sys
import os
# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import HTTPException, FastAPI
from pydantic import BaseModel
from typing import List, Optional
import structlog
import uvicorn

# Try to import ML dependencies, but don't fail if they're not available
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from easymodel.endpoints import finetuning, analytics
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some ML dependencies are not available: {e}")
    print("Running in limited mode - some features will be disabled")
    ML_AVAILABLE = False

app = FastAPI(title="EasyModel API", description="A comprehensive model fine-tuning and evaluation platform")
logger = structlog.get_logger()

@app.get("/")
def read_root():
    logger.info("In root path")
    return {
        "message": "EasyModel API - A comprehensive model fine-tuning and evaluation platform",
        "ml_available": ML_AVAILABLE,
        "note": "Some features may be limited due to missing dependencies"
    }

@app.get("/health")
def health_check():
    try:
        if ML_AVAILABLE:
            # Test model loading
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            return {"status": "ok", "details": "Models and tokenizer loaded successfully", "ml_available": True}
        else:
            return {"status": "ok", "details": "API running in limited mode", "ml_available": False}
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {"status": "limited", "details": f"Health check failed: {str(e)}", "ml_available": False}

@app.get("/api-info")
def api_info():
    endpoints = {
        "GET /": "API information",
        "GET /health": "Health check",
        "GET /api-info": "This endpoint information"
    }
    
    if ML_AVAILABLE:
        endpoints.update({
            "POST /finetuning/": "Start a fine-tuning job",
            "POST /analytics/analytics": "Run comprehensive model analytics",
            "POST /analytics/evaluate": "Evaluate a single model",
            "POST /analytics/compare": "Compare two models"
        })
    
    return {
        "name": "EasyModel API",
        "version": "0.1.0",
        "description": "A comprehensive model fine-tuning and evaluation platform",
        "ml_available": ML_AVAILABLE,
        "endpoints": endpoints,
        "note": "ML features require PyTorch and other dependencies" if not ML_AVAILABLE else "All features available"
    }

# Include ML endpoints only if dependencies are available
if ML_AVAILABLE:
    # Include the finetuning router with a specific prefix
    app.include_router(finetuning.router, prefix="/finetuning", tags=["Finetuning"])
    
    # Include the analytics router with a specific prefix
    app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
else:
    @app.get("/finetuning/")
    def finetuning_disabled():
        raise HTTPException(
            status_code=503, 
            detail="Fine-tuning is not available. PyTorch and other ML dependencies are required."
        )
    
    @app.get("/analytics/")
    def analytics_disabled():
        raise HTTPException(
            status_code=503, 
            detail="Analytics are not available. PyTorch and other ML dependencies are required."
        )

if __name__ == "__main__":
    uvicorn.run("main_ml:app", host="0.0.0.0", port=8000, reload=True)
