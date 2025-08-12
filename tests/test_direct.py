import sys
import os
sys.path.insert(0, '..')

from easymodel.endpoints.analytics import analytics_endpoint, AnalyticsRequest
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio

async def test_analytics():
    try:
        # Create request data
        data = AnalyticsRequest(
            dataset_url="ag_news",
            model_name="gpt2",
            task_type="text_generation"
        )
        
        # Test the endpoint function directly
        result = await analytics_endpoint(data)
        print("Success!")
        print("Result:", result)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_analytics())
