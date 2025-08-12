from fastapi import APIRouter, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
from easymodel.utils.text_analytics.textanalytics import run_all_analytics, run_model_evaluation, run_comparative_analysis
from pydantic import BaseModel
import logging

# Init
router = APIRouter()
logger = logging.getLogger(__name__)

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
    try:
        # Load/fetch model and tokenizer from cache
        if data.model_name not in model_cache:
            model = AutoModelForCausalLM.from_pretrained(data.model_name)
            tokenizer = AutoTokenizer.from_pretrained(data.model_name)
            model_cache[data.model_name] = (model, tokenizer)
        else:
            model, tokenizer = model_cache[data.model_name]

        # Run analytics
        analytics_results = run_all_analytics(model, tokenizer, data.dataset_url)
        return {"success": True, "results": analytics_results}
    except Exception as e:
        logger.error(f"Error in analytics endpoint: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while running analytics.")


@router.post("/evaluate")
async def evaluate_model(data: AnalyticsRequest):
    try:
        # Run model evaluation
        evaluation_results = run_model_evaluation(data.model_name, data.dataset_url, data.task_type)
        return {"success": True, "results": evaluation_results}
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while evaluating the model.")


@router.post("/compare")
async def compare_models(data: ComparativeAnalysisRequest):
    try:
        # Run comparative analysis
        comparison_results = run_comparative_analysis(data.model1_name, data.model2_name, data.dataset_url)
        return {"success": True, "results": comparison_results}
    except Exception as e:
        logger.error(f"Error in comparative analysis: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while comparing models.")
