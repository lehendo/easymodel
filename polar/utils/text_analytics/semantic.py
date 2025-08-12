# import os
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import AutoModel, AutoTokenizer
# from ..finetune_model import finetune_model
# import torch
#
#
# def encode_texts(texts, tokenizer, model, max_length=128):
#     inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
#     return embeddings
#
#
# def calculate_semantic_similarity(original_texts, fine_tuned_model, tokenizer, max_length=128):
#     original_embeddings = encode_texts(original_texts, tokenizer, fine_tuned_model, max_length)
#     fine_tuned_embeddings = encode_texts(original_texts, tokenizer, fine_tuned_model, max_length)
#
#     similarities = cosine_similarity(original_embeddings, fine_tuned_embeddings)
#     return similarities.mean()
#
#
# def analyze_semantic_drift(base_model, datasets, output_space, api_key, num_epochs, batch_size, max_length, subset_size, texts_to_evaluate):
#     if not os.path.exists("./semantic_drift_results"):
#         os.makedirs("./semantic_drift_results")
#
#     tokenizer = AutoTokenizer.from_pretrained(base_model)
#     model = AutoModel.from_pretrained(base_model)
#
#     semantic_similarities = []
#
#     for epoch in range(num_epochs):
#         finetune_model(
#             base_model=base_model,
#             datasets=datasets,
#             output_space=f"{output_space}_epoch_{epoch + 1}",
#             api_key=api_key,
#             num_epochs=1,
#             batch_size=batch_size,
#             max_length=max_length,
#             subset_size=subset_size
#         )
#
#         fine_tuned_model = AutoModel.from_pretrained(f"{output_space}_epoch_{epoch + 1}")
#
#         similarity = calculate_semantic_similarity(texts_to_evaluate, fine_tuned_model, tokenizer, max_length)
#         semantic_similarities.append(similarity)
#         print(f"Epoch {epoch + 1}: Semantic Similarity = {similarity: .4f}")
#
#     return semantic_similarities
