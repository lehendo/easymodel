import traceback
import sys
import os
sys.path.insert(0, '..')

from easymodel.utils.text_analytics.textanalytics import run_all_analytics
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    print("Running analytics...")
    result = run_all_analytics(model, tokenizer, 'ag_news')
    print("Success!")
    print("Result:", result)
    
except Exception as e:
    print("Error:", e)
    traceback.print_exc()
