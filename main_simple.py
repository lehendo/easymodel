from fastapi import HTTPException, FastAPI
import structlog
import uvicorn

app = FastAPI(title="EasyModel API", description="A comprehensive model fine-tuning and evaluation platform")
logger = structlog.get_logger()

@app.get("/")
def read_root():
    logger.info("In root path")
    return {"message": "EasyModel API - A comprehensive model fine-tuning and evaluation platform"}

@app.get("/health")
def health_check():
    try:
        return {"status": "ok", "details": "EasyModel API is running successfully"}
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail="Health check failed: " + str(e))

@app.get("/api-info")
def api_info():
    return {
        "name": "EasyModel API",
        "version": "0.1.0",
        "description": "A comprehensive model fine-tuning and evaluation platform",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /api-info": "This endpoint information"
        },
        "note": "ML features require additional dependencies (torch, transformers, etc.)"
    }

if __name__ == "__main__":
    uvicorn.run("main_simple:app", host="0.0.0.0", port=8000, reload=True)
