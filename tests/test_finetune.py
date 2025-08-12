import sys
import os
sys.path.insert(0, '..')

from easymodel.endpoints.finetuning import fine_tune, FinetuningRequest
import asyncio

async def test_finetune():
    try:
        # Create request data
        data = FinetuningRequest(
            model_name="gpt2",
            datasets=["ag_news"],
            output_space="test_output",
            num_epochs=1,
            batch_size=4,
            max_length=128,
            subset_size=100,
            api_key="test_key",
            task_type="generation",
            text_field="text"
        )
        
        # Test the endpoint function directly
        result = fine_tune(data)
        print("Success!")
        print("Result:", result)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_finetune())
