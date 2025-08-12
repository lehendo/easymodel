import requests
import json
import sys
import os
sys.path.insert(0, '..')

# Test the analytics endpoint
url = "http://localhost:8000/analytics/analytics"
data = {
    "dataset_url": "ag_news",
    "model_name": "gpt2", 
    "task_type": "text_generation"
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
