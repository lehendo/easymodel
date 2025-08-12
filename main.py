from fastapi import HTTPException
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import structlog
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

from easymodel.endpoints import finetuning, analytics

app = FastAPI(title="EasyModel API", description="A comprehensive model fine-tuning and evaluation platform")
logger = structlog.get_logger()

@app.get("/")
def read_root():
    logger.info("In root path")
    return {"message": "EasyModel API - A comprehensive model fine-tuning and evaluation platform"}

@app.get("/health")
def health_check():
    try:
        # Test model loading
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        return {"status": "ok", "details": "Models and tokenizer loaded successfully"}
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail="Health check failed: " + str(e))


# Include the finetuning router with a specific prefix
app.include_router(finetuning.router, prefix="/finetuning", tags=["Finetuning"])

# Include the analytics router with a specific prefix
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])

if __name__ == "__main__":
    # Only use this if you want to run the app through the `python main.py` command
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)