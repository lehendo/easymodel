# from .gqs import compute_fluency_grammar_coherence_relevance
#
#
# def run_all_analytics(model, tokenizer, dataset_url):
#     try:
#         # Step 1: Prepare dataset
#         train_data, val_data, test_data = prepare_dataset(dataset_url)
#
#         # Step 2: Compute Perplexity
#         test_texts = [item["text"] for item in test_data]
#         perplexity_results = compute_perplexity_lm(model, tokenizer, test_texts)
#
#         # Step 3: Compute GQS Metrics
#         predictions = test_texts  # Replace with model outputs if needed
#         references = [item["text"] for item in test_data]
#         fluency_scores, grammar_scores, coherence_scores, relevance_scores = compute_fluency_grammar_coherence_relevance(
#             predictions, references
#         )
#
#         # Step 4: Analyze Semantic Drift
#         texts_to_evaluate = test_texts[:5]  # Evaluate a sample of texts
#         semantic_similarities = analyze_semantic_drift(
#             base_model="bert-base-uncased",
#             datasets=[train_data, val_data, test_data],
#             output_space="output_space",
#             api_key="api_key",  # Replace with your actual API key
#             num_epochs=3,
#             batch_size=32,
#             max_length=128,
#             subset_size=0.1,
#             texts_to_evaluate=texts_to_evaluate,
#         )
#
#         return {
#             "perplexity": perplexity_results,
#             "fluency": fluency_scores,
#             "grammar": grammar_scores,
#             "coherence": coherence_scores,
#             "relevance": relevance_scores,
#             "semantic_similarity": semantic_similarities,
#             # "token_efficiency": token_efficiency
#         }
#
#     except Exception as e:
#         raise RuntimeError(f"Error in running analytics: {str(e)}")
