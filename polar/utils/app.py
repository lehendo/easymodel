
import gradio as gr
from transformers import pipeline

# Load the model
model = pipeline("text-classification", model=".")

def classify_text(text):
    return model(text)

# Create the Gradio app
iface = gr.Interface(fn=classify_text, inputs="text", outputs="label", title="Text Classification")
iface.launch()
        