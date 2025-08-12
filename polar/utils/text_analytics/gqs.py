import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
import nltk
import math
from datasets import load_dataset

# Download necessary NLTK data
nltk.download('punkt')

# Load models
grammar_model_name = "textattack/roberta-base-CoLA"
grammar_model = AutoModelForSequenceClassification.from_pretrained(grammar_model_name)
grammar_tokenizer = AutoTokenizer.from_pretrained(grammar_model_name)

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


# Function for grammaticality check
def check_grammar(text):
    inputs = grammar_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = grammar_model(**inputs)
    logits = outputs.logits
    grammar_score = torch.softmax(logits, dim=-1)[0][1].item()
    return grammar_score


# Function for fluency check (using perplexity)
def calculate_perplexity(text):
    inputs = gpt2_tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
    return math.exp(outputs.loss.item())


# Function to compute semantic similarity
def compute_similarity(text1, text2):
    embeddings1 = sentence_model.encode([text1])
    embeddings2 = sentence_model.encode([text2])
    similarity = cosine_similarity(embeddings1, embeddings2)
    return similarity[0][0]


# Function to check coherence
def check_coherence(text):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 2:
        return 1.0  # Perfect coherence for single sentence
    embeddings = sentence_model.encode(sentences)
    similarities = cosine_similarity(embeddings)
    # Average of off-diagonal elements
    coherence = (similarities.sum() - similarities.trace()) / (similarities.size - similarities.shape[0])
    return coherence


def scale_to_100(score, is_perplexity=False):
    if is_perplexity:
        # For perplexity, lower is better, so we invert the scale
        return 100 * max(0, min(1, 1 / (score + 1)))
    else:
        return 100 * score


def compute_metrics(predictions, references):
    grammar_scores = [check_grammar(text) for text in predictions]
    fluency_scores = [calculate_perplexity(text) for text in predictions]
    coherence_scores = [check_coherence(text) for text in predictions]
    relevance_scores = [compute_similarity(pred, ref) for pred, ref in zip(predictions, references)]

    return {
        "grammar": scale_to_100(np.mean(grammar_scores)),
        "fluency": scale_to_100(np.mean(fluency_scores), is_perplexity=True),
        "coherence": scale_to_100(np.mean(coherence_scores)),
        "relevance": scale_to_100(np.mean(relevance_scores))
    }


def generate_text_from_model(model_path, tokenizer_path, dataset):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    generated_texts = []
    for text in dataset['text']:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_length=50)  # Adjust max_length as needed
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated_text)

    return generated_texts


def compute_aggregate_gqs_from_finetuned_model(model_url, dataset_urls):
    # Load fine-tuned model
    model_path = f"./fine_tuned_models/{model_url}"  # Replace with the actual path of the fine-tuned model
    tokenizer_path = model_path  # Assuming tokenizer is also saved in the same directory

    total_gqs = {'fluency': 0, 'grammar': 0, 'coherence': 0, 'relevance': 0}
    num_datasets = len(dataset_urls)

    for dataset_url in dataset_urls:
        # Load the dataset
        dataset = load_dataset(dataset_url, split="train")

        # Generate text from the fine-tuned model for the current dataset
        generated_texts = generate_text_from_model(model_path, tokenizer_path, dataset)

        # Compute the GQS metrics using the generated texts
        gqs_results = compute_metrics(generated_texts, dataset['text'])  # Using dataset text as references

        # Aggregate the results
        for metric in total_gqs:
            total_gqs[metric] += gqs_results[metric]

    # Calculate the average score for each metric
    aggregate_gqs = {metric: total / num_datasets for metric, total in total_gqs.items()}

    return aggregate_gqs
