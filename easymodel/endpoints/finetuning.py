from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
import os
import json
from typing import Optional
from queue import Queue, Empty
import threading

# Try to import finetune_model, but don't fail if ML dependencies aren't available
try:
    from easymodel.utils.finetune2 import finetune_model
    FINETUNING_AVAILABLE = True
except ImportError as e:
    FINETUNING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Finetuning not available: {e}")
    def finetune_model(*args, **kwargs):
        raise RuntimeError("ML dependencies not available. Please install PyTorch and transformers.")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()

# Store progress queues for active training jobs
progress_queues: dict[str, Queue] = {}
# Store cancel flags for active training jobs
cancel_flags: dict[str, bool] = {}
# Store training threads for cancellation
training_threads: dict[str, threading.Thread] = {}

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

def run_training(job_id: str, data: FinetuningRequest, api_key: str):
    """Run training in background thread and emit progress updates."""
    if not FINETUNING_AVAILABLE:
        progress_queue = progress_queues.get(job_id)
        if progress_queue:
            progress_queue.put({
                "stage": "error",
                "progress": 0,
                "message": "ML dependencies not available. Please install PyTorch and transformers."
            })
        return
    
    progress_queue = progress_queues.get(job_id)
    
    def progress_callback(update: dict):
        """Callback to emit progress updates."""
        if progress_queue:
            progress_queue.put(update)
    
    def cancel_check() -> bool:
        """Check if training should be cancelled."""
        return cancel_flags.get(job_id, False)
    
    try:
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
            label_field=data.label_field if data.task_type != "generation" else None,
            progress_callback=progress_callback,
            cancel_flag=cancel_check
        )
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        if progress_queue:
            # Check if it was cancelled
            if cancel_flags.get(job_id, False):
                progress_queue.put({
                    "stage": "cancelled",
                    "progress": 0,
                    "message": "Training cancelled by user"
                })
            else:
                progress_queue.put({
                    "stage": "error",
                    "progress": 0,
                    "message": f"Error: {str(e)}"
                })
    finally:
        # Clean up after a delay to allow final messages to be sent
        if job_id in progress_queues:
            threading.Timer(5.0, lambda: progress_queues.pop(job_id, None)).start()
        if job_id in cancel_flags:
            threading.Timer(5.0, lambda: cancel_flags.pop(job_id, None)).start()
        if job_id in training_threads:
            threading.Timer(5.0, lambda: training_threads.pop(job_id, None)).start()


@router.post("/")
def fine_tune(data: FinetuningRequest, background_tasks: BackgroundTasks):
    try:
        import uuid
        job_id = str(uuid.uuid4())
        logger.info(f"Starting fine-tuning job {job_id} for model: {data.model_name}")

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

        # Create progress queue and cancel flag for this job
        progress_queues[job_id] = Queue()
        cancel_flags[job_id] = False
        
        # Start training in background thread
        training_thread = threading.Thread(target=run_training, args=(job_id, data, api_key), daemon=True)
        training_threads[job_id] = training_thread
        training_thread.start()

        return {"job_id": job_id, "message": f"Fine-tuning initiated for model: {data.model_name} with datasets {data.datasets}"}

    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during fine-tuning: {str(e)}")


@router.get("/progress/{job_id}")
async def stream_progress(job_id: str):
    """Stream training progress via Server-Sent Events."""
    if job_id not in progress_queues:
        raise HTTPException(status_code=404, detail="Job not found")
    
    progress_queue = progress_queues[job_id]
    
    async def event_generator():
        """Generate SSE events from progress queue."""
        try:
            while True:
                try:
                    # Wait for progress update with timeout
                    update = progress_queue.get(timeout=1.0)
                    
                    # Send progress update
                    yield f"data: {json.dumps(update)}\n\n"
                    
                    # If completed, error, or cancelled, close connection
                    if update.get("stage") in ["completed", "error", "cancelled"]:
                        break
                        
                except Empty:
                    # Timeout - send keepalive
                    yield ": keepalive\n\n"
                    continue
                except Exception as e:
                    logger.error(f"Error getting progress update: {str(e)}")
                    yield ": keepalive\n\n"
                    continue
                    
        except Exception as e:
            logger.error(f"Error in event generator: {str(e)}")
            yield f"data: {json.dumps({'stage': 'error', 'message': str(e)})}\n\n"
        finally:
            # Clean up queue after a delay
            if job_id in progress_queues:
                threading.Timer(2.0, lambda: progress_queues.pop(job_id, None)).start()
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/cancel/{job_id}")
def cancel_training(job_id: str):
    """Cancel a running training job."""
    # Check if job exists in any of our tracking dictionaries
    job_exists = (
        job_id in progress_queues or 
        job_id in cancel_flags or 
        job_id in training_threads
    )
    
    if not job_exists:
        # Job might have already completed or never existed
        logger.warning(f"Cancel requested for job {job_id} but job not found. It may have already completed.")
        return {
            "message": f"Job {job_id} not found. It may have already completed or never existed.",
            "job_id": job_id,
            "status": "not_found"
        }
    
    # Set cancel flag (create if it doesn't exist)
    cancel_flags[job_id] = True
    
    # Send cancellation message to progress queue if it exists
    progress_queue = progress_queues.get(job_id)
    if progress_queue:
        try:
            progress_queue.put({
                "stage": "cancelling",
                "progress": 0,
                "message": "Cancelling training..."
            }, timeout=0.1)
        except:
            # Queue might be full, that's okay
            pass
    
    logger.info(f"Training job {job_id} cancellation requested")
    return {
        "message": f"Training job {job_id} cancellation requested",
        "job_id": job_id,
        "status": "cancelling"
    }
