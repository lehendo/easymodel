from fastapi import HTTPException
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import structlog
from dotenv import load_dotenv
import uvicorn
import os

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

# Set Hugging Face token from environment if available
hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

# Try to import ML dependencies, but don't fail if they're not available
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML dependencies not available. Some features will be disabled.")

from easymodel.endpoints import finetuning, analytics

app = FastAPI(title="EasyModel API", description="A comprehensive model fine-tuning and evaluation platform")
logger = structlog.get_logger()

# Add CORS middleware to allow frontend requests
# Get allowed origins from environment variable or use defaults
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Can be configured via ALLOWED_ORIGINS env var
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    logger.info("In root path")
    return {"message": "EasyModel API - A comprehensive model fine-tuning and evaluation platform"}

@app.get("/health")
def health_check():
    try:
        if ML_AVAILABLE:
            # Test model loading - use token from environment if available
            hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
            # For public models like gpt2, we don't need auth, but we'll use token if provided
            tokenizer = AutoTokenizer.from_pretrained("gpt2", token=hf_token if hf_token else None)
            model = AutoModelForCausalLM.from_pretrained("gpt2", token=hf_token if hf_token else None)
            return {"status": "ok", "details": "Models and tokenizer loaded successfully", "ml_available": True, "token_configured": bool(hf_token)}
        else:
            return {"status": "ok", "details": "API is running but ML dependencies are not available", "ml_available": False}
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        # Return limited status but don't fail completely
        return {"status": "limited", "details": f"Health check warning: {str(e)}", "ml_available": ML_AVAILABLE}


# Include the finetuning router with a specific prefix
app.include_router(finetuning.router, prefix="/finetuning", tags=["Finetuning"])

# Include the analytics router with a specific prefix
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])

if __name__ == "__main__":
    # Only use this if you want to run the app through the `python main.py` command
    # Use import string for reload to work properly
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)