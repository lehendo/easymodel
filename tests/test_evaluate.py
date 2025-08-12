import sys
import os
sys.path.insert(0, '..')

from easymodel.endpoints.analytics import evaluate_model, AnalyticsRequest
import asyncio

async def test_evaluate():
    try:
        # Create request data
        data = AnalyticsRequest(
            dataset_url="ag_news",
            model_name="gpt2",
            task_type="text_generation"
        )
        
        # Test the endpoint function directly
        result = await evaluate_model(data)
        print("Success!")
        print("Result:", result)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_evaluate())
