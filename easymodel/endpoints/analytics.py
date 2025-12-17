from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

# Try to import ML dependencies, but don't fail if they're not available
ANALYTICS_AVAILABLE = False
MISSING_DEPENDENCY = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from easymodel.utils.text_analytics.textanalytics import run_all_analytics, run_model_evaluation, run_comparative_analysis
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    ANALYTICS_AVAILABLE = False
    MISSING_DEPENDENCY = str(e)
    logger = logging.getLogger(__name__)
    logger.warning(f"Analytics not available: {e}")
    logger.warning("Missing dependencies. Required: transformers, torch, sentence-transformers, scikit-learn")
    def run_all_analytics(*args, **kwargs):
        raise RuntimeError(f"ML dependencies not available. Missing: {MISSING_DEPENDENCY}. Please install PyTorch, transformers, sentence-transformers, and scikit-learn.")
    def run_model_evaluation(*args, **kwargs):
        raise RuntimeError(f"ML dependencies not available. Missing: {MISSING_DEPENDENCY}. Please install PyTorch, transformers, sentence-transformers, and scikit-learn.")
    def run_comparative_analysis(*args, **kwargs):
        raise RuntimeError(f"ML dependencies not available. Missing: {MISSING_DEPENDENCY}. Please install PyTorch, transformers, sentence-transformers, and scikit-learn.")

# Init
router = APIRouter()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Cache for models and tokenizers to avoid reloading frequently
model_cache = {}


class AnalyticsRequest(BaseModel):
    dataset_url: str
    model_name: str = "gpt2"
    task_type: str = "text_generation"


class ComparativeAnalysisRequest(BaseModel):
    model1_name: str
    model2_name: str
    dataset_url: str


@router.post("/analytics")
async def analytics_endpoint(data: AnalyticsRequest):
    if not ANALYTICS_AVAILABLE:
        error_detail = f"Analytics not available. ML dependencies not installed."
        if MISSING_DEPENDENCY:
            error_detail += f" Missing: {MISSING_DEPENDENCY}"
        raise HTTPException(status_code=503, detail=error_detail)
    
    try:
        logger.info(f"Starting analytics for model: {data.model_name}, dataset: {data.dataset_url}")
        
        # Load/fetch model and tokenizer from cache
        if data.model_name not in model_cache:
            logger.info(f"Loading model and tokenizer: {data.model_name}")
            model = AutoModelForCausalLM.from_pretrained(data.model_name)
            tokenizer = AutoTokenizer.from_pretrained(data.model_name)
            model_cache[data.model_name] = (model, tokenizer)
        else:
            logger.info(f"Using cached model and tokenizer: {data.model_name}")
            model, tokenizer = model_cache[data.model_name]

        # Run analytics
        logger.info("Running analytics computation...")
        analytics_results = run_all_analytics(model, tokenizer, data.dataset_url)
        logger.info("Analytics computation completed successfully")
        return {"success": True, "results": analytics_results}
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error in analytics endpoint: {e}")
        logger.error(f"Full traceback: {error_trace}")
        
        # Return detailed error in development, generic in production
        import os
        is_dev = os.getenv("ENVIRONMENT", "development").lower() != "production"
        error_detail = f"An error occurred while running analytics: {str(e)}" if is_dev else "An error occurred while running analytics."
        raise HTTPException(status_code=500, detail=error_detail)


@router.post("/evaluate")
async def evaluate_model(data: AnalyticsRequest):
    if not ANALYTICS_AVAILABLE:
        error_detail = f"Analytics not available. ML dependencies not installed."
        if MISSING_DEPENDENCY:
            error_detail += f" Missing: {MISSING_DEPENDENCY}"
        raise HTTPException(status_code=503, detail=error_detail)
    
    try:
        logger.info(f"Starting model evaluation for model: {data.model_name}, dataset: {data.dataset_url}, task: {data.task_type}")
        
        # Run model evaluation
        evaluation_results = run_model_evaluation(data.model_name, data.dataset_url, data.task_type)
        logger.info("Model evaluation completed successfully")
        return {"success": True, "results": evaluation_results}
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error in model evaluation: {e}")
        logger.error(f"Full traceback: {error_trace}")
        
        # Return detailed error in development, generic in production
        import os
        is_dev = os.getenv("ENVIRONMENT", "development").lower() != "production"
        error_detail = f"An error occurred while evaluating the model: {str(e)}" if is_dev else "An error occurred while evaluating the model."
        raise HTTPException(status_code=500, detail=error_detail)


@router.post("/compare")
async def compare_models(data: ComparativeAnalysisRequest):
    if not ANALYTICS_AVAILABLE:
        error_detail = f"Analytics not available. ML dependencies not installed."
        if MISSING_DEPENDENCY:
            error_detail += f" Missing: {MISSING_DEPENDENCY}"
        raise HTTPException(status_code=503, detail=error_detail)
    
    try:
        logger.info(f"Starting comparative analysis for models: {data.model1_name} vs {data.model2_name}, dataset: {data.dataset_url}")
        
        # Run comparative analysis
        comparison_results = run_comparative_analysis(data.model1_name, data.model2_name, data.dataset_url)
        logger.info("Comparative analysis completed successfully")
        return {"success": True, "results": comparison_results}
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error in comparative analysis: {e}")
        logger.error(f"Full traceback: {error_trace}")
        
        # Return detailed error in development, generic in production
        import os
        is_dev = os.getenv("ENVIRONMENT", "development").lower() != "production"
        error_detail = f"An error occurred while comparing models: {str(e)}" if is_dev else "An error occurred while comparing models."
        raise HTTPException(status_code=500, detail=error_detail)
