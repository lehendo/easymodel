# from fastapi import APIRouter, HTTPException
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from ..utils.text_analytics.textanalytics import run_all_analytics
# import logging
#
# # Init
# router = APIRouter()
# logger = logging.getLogger(__name__)
#
# # Cache for models and tokenizers to avoid reloading frequently
# model_cache = {}
#
#
# @router.post("/analytics")
# async def analytics_endpoint(dataset_url: str, model_name: str = "gpt2"):
#
#     try:
#         # Load/fetch model and tokenizer from cache
#         if model_name not in model_cache:
#             model = AutoModelForCausalLM.from_pretrained(model_name)
#             tokenizer = AutoTokenizer.from_pretrained(model_name)
#             model_cache[model_name] = (model, tokenizer)
#         else:
#             model, tokenizer = model_cache[model_name]
#
#         # Run analytics
#         analytics_results = run_all_analytics(model, tokenizer, dataset_url)
#         return {"success": True, "results": analytics_results}
#     except Exception as e:
#         logger.error(f"Error in analytics endpoint: {e}")
#         raise HTTPException(status_code=500, detail="An error occurred while running analytics.")
