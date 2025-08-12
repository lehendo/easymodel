import os
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer


def encode_texts(texts, tokenizer, model, max_length=128):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings


def calculate_semantic_similarity(original_texts, fine_tuned_model, tokenizer, max_length=128):
    original_embeddings = encode_texts(original_texts, tokenizer, fine_tuned_model, max_length)
    fine_tuned_embeddings = encode_texts(original_texts, tokenizer, fine_tuned_model, max_length)

    similarities = cosine_similarity(original_embeddings, fine_tuned_embeddings)
    return similarities.mean()


def analyze_semantic_drift(base_model, datasets, output_space, api_key, num_epochs, batch_size, max_length, subset_size, texts_to_evaluate):
    if not os.path.exists("./semantic_drift_results"):
        os.makedirs("./semantic_drift_results")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModel.from_pretrained(base_model)

    semantic_similarities = []

    for epoch in range(num_epochs):
        # Note: This would need to be integrated with the actual fine-tuning process
        # For now, we'll just compute similarity with the base model
        fine_tuned_model = AutoModel.from_pretrained(base_model)

        similarity = calculate_semantic_similarity(texts_to_evaluate, fine_tuned_model, tokenizer, max_length)
        semantic_similarities.append(similarity)
        print(f"Epoch {epoch + 1}: Semantic Similarity = {similarity: .4f}")

    return semantic_similarities


def compute_semantic_similarity(predictions, references):
    """
    Compute semantic similarity between predictions and references using sentence transformers.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
    
    Returns:
        float: Average semantic similarity score
    """
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    pred_embeddings = sentence_model.encode(predictions)
    ref_embeddings = sentence_model.encode(references)
    similarities = cosine_similarity(pred_embeddings, ref_embeddings)
    return similarities.diagonal().mean()


def compute_semantic_similarity_batch(texts1, texts2):
    """
    Compute semantic similarity between two batches of texts.
    
    Args:
        texts1: First batch of texts
        texts2: Second batch of texts
    
    Returns:
        numpy.ndarray: Similarity matrix
    """
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings1 = sentence_model.encode(texts1)
    embeddings2 = sentence_model.encode(texts2)
    similarities = cosine_similarity(embeddings1, embeddings2)
    return similarities


def compute_semantic_drift_over_time(base_model, fine_tuned_models, texts_to_evaluate, max_length=128):
    """
    Compute semantic drift between base model and fine-tuned models over time.
    
    Args:
        base_model: Base model name or path
        fine_tuned_models: List of fine-tuned model paths
        texts_to_evaluate: Texts to evaluate semantic similarity on
        max_length: Maximum sequence length
    
    Returns:
        list: Semantic similarity scores over time
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base_model_instance = AutoModel.from_pretrained(base_model)
    
    drift_scores = []
    
    for model_path in fine_tuned_models:
        fine_tuned_model = AutoModel.from_pretrained(model_path)
        similarity = calculate_semantic_similarity(texts_to_evaluate, fine_tuned_model, tokenizer, max_length)
        drift_scores.append(similarity)
    
    return drift_scores


def compute_semantic_coherence(texts):
    """
    Compute semantic coherence within a set of texts.
    
    Args:
        texts: List of texts to evaluate coherence
    
    Returns:
        float: Coherence score
    """
    if len(texts) < 2:
        return 1.0  # Perfect coherence for single text
    
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = sentence_model.encode(texts)
    similarities = cosine_similarity(embeddings)
    
    # Average of off-diagonal elements (excluding self-similarity)
    coherence = (similarities.sum() - similarities.trace()) / (similarities.size - similarities.shape[0])
    return coherence
