import logging
from .gqs import compute_metrics, compute_aggregate_gqs_from_finetuned_model
from .perplexity import prepare_dataset, compute_perplexity_lm, compute_batch_perplexity
from .semantic import compute_semantic_similarity, compute_semantic_coherence
from .tokenefficiency import (
    compute_summarization_efficiency,
    compute_qa_efficiency,
    compute_translation_efficiency,
    compute_paraphrasing_efficiency,
    compute_code_generation_efficiency
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

logger = logging.getLogger(__name__)


def run_all_analytics(model, tokenizer, dataset_url):
    """
    Run comprehensive analytics on a model and dataset.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        dataset_url: URL or path to the dataset
    
    Returns:
        dict: Comprehensive analytics results
    """
    try:
        logger.info(f"Starting analytics computation for dataset: {dataset_url}")
        
        # Fix tokenizer padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.debug("Set tokenizer pad_token to eos_token")
        
        # Step 1: Load dataset and extract texts
        try:
            logger.info(f"Loading dataset: {dataset_url}")
            dataset = load_dataset(dataset_url)
            # Check available splits
            available_splits = list(dataset.keys())
            logger.info(f"Available splits for {dataset_url}: {available_splits}")
            
            # Choose the best available split
            if "validation" in available_splits:
                split_name = "validation"
            elif "test" in available_splits:
                split_name = "test"
            elif "train" in available_splits:
                split_name = "train"
            else:
                raise ValueError(f"No suitable split found in dataset {dataset_url}. Available splits: {available_splits}")
            
            dataset = dataset[split_name]
            logger.info(f"Using split: {split_name} with {len(dataset)} examples")
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_url}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to load dataset {dataset_url}: {str(e)}")
        
        # Extract texts based on dataset structure
        texts = []
        for i, item in enumerate(dataset[:100]):  # Use first 100 examples
            try:
                if isinstance(item, dict):
                    if "text" in item:
                        texts.append(item["text"])
                    elif "context" in item and "question" in item:
                        # For QA datasets like SQuAD, combine context and question
                        texts.append(f"Context: {item['context']} Question: {item['question']}")
                    elif "sentence" in item:
                        texts.append(item["sentence"])
                    elif "premise" in item and "hypothesis" in item:
                        # For NLI datasets
                        texts.append(f"Premise: {item['premise']} Hypothesis: {item['hypothesis']}")
                    else:
                        # Fallback: use the first string field
                        for key, value in item.items():
                            if isinstance(value, str):
                                texts.append(value)
                                break
                else:
                    # If item is not a dict, try to convert to string
                    texts.append(str(item))
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue
        
        if not texts:
            error_msg = f"No suitable text fields found in dataset {dataset_url}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Successfully extracted {len(texts)} texts from dataset")

        # Step 2: Compute Perplexity
        try:
            logger.info("Computing perplexity...")
            perplexity_results = compute_perplexity_lm(model, tokenizer, texts[0])  # Compute for first text as example
            logger.info(f"Perplexity computation completed: {perplexity_results}")
        except Exception as e:
            logger.error(f"Error computing perplexity: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            perplexity_results = {"error": str(e)}
        
        # Step 3: Compute GQS Metrics
        try:
            logger.info("Computing GQS metrics...")
            predictions = texts[:10]  # Use first 10 texts as predictions
            references = texts[:10]   # Use same texts as references
            gqs_results = compute_metrics(predictions, references)
            logger.info(f"GQS metrics computation completed: {gqs_results}")
        except Exception as e:
            logger.error(f"Error computing GQS metrics: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            gqs_results = {"error": str(e)}
        
        # Step 4: Compute Semantic Similarity
        try:
            logger.info("Computing semantic similarity...")
            semantic_similarity = compute_semantic_similarity(predictions, references)
            logger.info(f"Semantic similarity computation completed: {semantic_similarity}")
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            semantic_similarity = {"error": str(e)}
        
        # Step 5: Compute Semantic Coherence
        try:
            logger.info("Computing semantic coherence...")
            coherence_score = compute_semantic_coherence(texts[:5])
            logger.info(f"Semantic coherence computation completed: {coherence_score}")
        except Exception as e:
            logger.error(f"Error computing semantic coherence: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            coherence_score = {"error": str(e)}

        # Step 6: Compute Token Efficiency (for different tasks)
        # Note: This requires specific dataset formats, so we'll provide a template
        token_efficiency_results = {
            "summarization": "Requires dataset with 'text' field",
            "question_answering": "Requires dataset with 'question' and 'context' fields",
            "translation": "Requires dataset with 'text' field",
            "paraphrasing": "Requires dataset with 'text' field",
            "code_generation": "Requires dataset with 'prompt' field"
        }

        return {
            "perplexity": perplexity_results,
            "gqs_metrics": gqs_results,
            "semantic_similarity": semantic_similarity,
            "semantic_coherence": coherence_score,
            "token_efficiency": token_efficiency_results,
            "dataset_info": {
                "dataset_name": dataset_url,
                "num_examples_processed": len(texts),
                "sample_texts": texts[:3]  # Show first 3 texts as examples
            }
        }

    except Exception as e:
        logger.error(f"Error in running analytics: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Error in running analytics: {str(e)}")


def run_model_evaluation(model_name, dataset_url, task_type="text_generation"):
    """
    Run comprehensive model evaluation for a specific task.
    
    Args:
        model_name: Name or path of the model
        dataset_url: URL or path to the dataset
        task_type: Type of task to evaluate
    
    Returns:
        dict: Evaluation results
    """
    try:
        logger.info(f"Starting model evaluation for model: {model_name}, dataset: {dataset_url}, task: {task_type}")
        
        # Load model and tokenizer
        logger.info(f"Loading model and tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Fix tokenizer padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load dataset
        try:
            logger.info(f"Loading dataset: {dataset_url}")
            dataset = load_dataset(dataset_url)
            # Check available splits
            available_splits = list(dataset.keys())
            logger.info(f"Available splits for {dataset_url}: {available_splits}")
            
            # Choose the best available split
            if "validation" in available_splits:
                split_name = "validation"
            elif "test" in available_splits:
                split_name = "test"
            elif "train" in available_splits:
                split_name = "train"
            else:
                raise ValueError(f"No suitable split found in dataset {dataset_url}. Available splits: {available_splits}")
            
            dataset = dataset[split_name]
            logger.info(f"Using split: {split_name} with {len(dataset)} examples")
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_url}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to load dataset {dataset_url}: {str(e)}")
        
        # Extract texts based on dataset structure
        texts = []
        for i, item in enumerate(dataset[:100]):  # Use first 100 examples
            try:
                if isinstance(item, dict):
                    if "text" in item:
                        texts.append(item["text"])
                    elif "context" in item and "question" in item:
                        # For QA datasets like SQuAD, combine context and question
                        texts.append(f"Context: {item['context']} Question: {item['question']}")
                    elif "sentence" in item:
                        texts.append(item["sentence"])
                    elif "premise" in item and "hypothesis" in item:
                        # For NLI datasets
                        texts.append(f"Premise: {item['premise']} Hypothesis: {item['hypothesis']}")
                    else:
                        # Fallback: use the first string field
                        for key, value in item.items():
                            if isinstance(value, str):
                                texts.append(value)
                                break
                else:
                    # If item is not a dict, try to convert to string
                    texts.append(str(item))
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue
        
        if not texts:
            error_msg = f"No suitable text fields found in dataset {dataset_url}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Successfully extracted {len(texts)} texts from dataset")
        
        # Compute perplexity
        try:
            logger.info("Computing batch perplexity...")
            perplexities = compute_batch_perplexity(model, tokenizer, texts[:10])
            avg_perplexity = sum(perplexities) / len(perplexities)
            logger.info(f"Average perplexity: {avg_perplexity}")
        except Exception as e:
            logger.error(f"Error computing perplexity: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            avg_perplexity = 0.0
        
        # Compute GQS metrics
        try:
            logger.info("Computing GQS metrics...")
            gqs_results = compute_metrics(texts[:10], texts[:10])
            logger.info(f"GQS metrics: {gqs_results}")
        except Exception as e:
            logger.error(f"Error computing GQS metrics: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            gqs_results = {"error": str(e)}
        
        # Compute semantic coherence
        try:
            logger.info("Computing semantic coherence...")
            coherence = compute_semantic_coherence(texts[:5])
            logger.info(f"Semantic coherence: {coherence}")
        except Exception as e:
            logger.error(f"Error computing semantic coherence: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            coherence = 0.0
        
        return {
            "model_name": model_name,
            "dataset": dataset_url,
            "task_type": task_type,
            "perplexity": avg_perplexity,
            "gqs_metrics": gqs_results,
            "semantic_coherence": coherence
        }
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Error in model evaluation: {str(e)}")


def run_comparative_analysis(model1_name, model2_name, dataset_url):
    """
    Run comparative analysis between two models.
    
    Args:
        model1_name: Name or path of the first model
        model2_name: Name or path of the second model
        dataset_url: URL or path to the dataset
    
    Returns:
        dict: Comparative analysis results
    """
    try:
        logger.info(f"Starting comparative analysis: {model1_name} vs {model2_name} on {dataset_url}")
        
        # Evaluate both models
        logger.info(f"Evaluating first model: {model1_name}")
        results1 = run_model_evaluation(model1_name, dataset_url)
        logger.info(f"Evaluating second model: {model2_name}")
        results2 = run_model_evaluation(model2_name, dataset_url)
        
        # Compute differences
        comparison = {
            "model1": results1,
            "model2": results2,
            "differences": {
                "perplexity_diff": results1["perplexity"] - results2["perplexity"],
                "gqs_grammar_diff": results1["gqs_metrics"]["grammar"] - results2["gqs_metrics"]["grammar"],
                "gqs_fluency_diff": results1["gqs_metrics"]["fluency"] - results2["gqs_metrics"]["fluency"],
                "gqs_coherence_diff": results1["gqs_metrics"]["coherence"] - results2["gqs_metrics"]["coherence"],
                "gqs_relevance_diff": results1["gqs_metrics"]["relevance"] - results2["gqs_metrics"]["relevance"],
                "coherence_diff": results1["semantic_coherence"] - results2["semantic_coherence"]
            }
        }
        
        return comparison
        
    except Exception as e:
        logger.error(f"Error in comparative analysis: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Error in comparative analysis: {str(e)}")


def run_finetuning_analytics(base_model, fine_tuned_model, dataset_urls):
    """
    Run analytics specifically for fine-tuning evaluation.
    
    Args:
        base_model: Base model name or path
        fine_tuned_model: Fine-tuned model name or path
        dataset_urls: List of dataset URLs used for fine-tuning
    
    Returns:
        dict: Fine-tuning analytics results
    """
    try:
        # Run comparative analysis
        comparison = run_comparative_analysis(base_model, fine_tuned_model, dataset_urls[0])
        
        # Compute aggregate GQS metrics
        gqs_aggregate = compute_aggregate_gqs_from_finetuned_model(fine_tuned_model, dataset_urls)
        
        return {
            "comparison": comparison,
            "aggregate_gqs": gqs_aggregate,
            "datasets_used": dataset_urls
        }
        
    except Exception as e:
        raise RuntimeError(f"Error in fine-tuning analytics: {str(e)}")
