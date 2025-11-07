from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from easymodel.utils.finetune2 import finetune_model
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()

class FinetuningRequest(BaseModel):
    model_name: str
    datasets: list[str]
    output_space: str
    num_epochs: int = 1
    batch_size: int = 8
    max_length: int = 128
    subset_size: int = 1000
    api_key: Optional[str] = None  # Optional - will use environment variable if not provided
    task_type: str  # Task type like 'generation', 'classification', etc.
    text_field: str  # Single text field that the user wants to use
    label_field: str = None  # Label field is optional and only required for non-generation tasks

@router.post("/")
def fine_tune(data: FinetuningRequest):
    try:
        logger.info(f"Starting fine-tuning for model: {data.model_name}")

        # For non-generation tasks, we check that a label_field is provided
        if data.task_type != "generation" and not data.label_field:
            raise HTTPException(status_code=400, detail="Label field is required for non-generation tasks.")

        # Get API key from environment variable if not provided in request
        api_key = data.api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        if not api_key:
            raise HTTPException(
                status_code=400, 
                detail="Hugging Face API key is required. Please set HUGGINGFACE_API_KEY or HF_TOKEN environment variable, or provide api_key in the request."
            )

        # Call the finetune_model function with the required parameters
        finetune_model(
            base_model=data.model_name,
            datasets=data.datasets,
            output_space=data.output_space,
            api_key=api_key,
            num_epochs=data.num_epochs,
            batch_size=data.batch_size,
            max_length=data.max_length,
            subset_size=data.subset_size,
            task_type=data.task_type,
            text_field=data.text_field,
            label_field=data.label_field if data.task_type != "generation" else None
        )

        logger.info(f"Fine-tuning completed for model: {data.model_name}")
        return {"message": f"Fine-tuning initiated for model: {data.model_name} with datasets {data.datasets}"}

    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during fine-tuning: {str(e)}")
