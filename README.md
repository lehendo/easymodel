# EasyModel

A comprehensive model fine-tuning and evaluation platform that provides powerful tools for training, analyzing, and comparing language models.

## Features

### Model Fine-tuning
- Support for multiple task types: classification, sequence-to-sequence, text generation, and token classification
- Automatic dataset field detection
- Configurable training parameters (epochs, batch size, sequence length)
- Integration with Hugging Face Hub for model sharing

### Comprehensive Analytics
- **GQS Metrics**: Grammar, Quality, and Semantic evaluation
- **Perplexity Analysis**: Language model performance assessment
- **Semantic Similarity**: Measure semantic drift and coherence
- **Token Efficiency**: Analyze input/output token ratios for different tasks
- **Comparative Analysis**: Compare multiple models side-by-side

### Easy-to-Use API
- RESTful API endpoints for all operations
- Automatic model and tokenizer caching
- Comprehensive error handling and logging
- Health check endpoints

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/lehendo/easymodel.git
cd easymodel

# Install dependencies
poetry install

# Run the application
poetry run python main.py
```

### API Usage

#### Fine-tune a Model (you might want to alter the parameters depending on the type of model and dataset)

```bash
curl -X POST "http://localhost:8000/finetuning/" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt2",
    "datasets": ["squad"],
    "output_space": "my-finetuned-model",
    "num_epochs": 3,
    "batch_size": 8,
    "max_length": 128,
    "subset_size": 1000,
    "api_key": "your-huggingface-api-key",
    "task_type": "generation",
    "text_field": "text"
  }'
```

#### Run Model Analytics

```bash
curl -X POST "http://localhost:8000/analytics/analytics" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_url": "squad",
    "model_name": "gpt2",
    "task_type": "text_generation"
  }'
```

#### Compare Two Models

```bash
curl -X POST "http://localhost:8000/analytics/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "model1_name": "gpt2",
    "model2_name": "gpt2-medium",
    "dataset_url": "squad"
  }'
```

## API Endpoints

### Fine-tuning Endpoints
- `POST /finetuning/` - Start a fine-tuning job

### Analytics Endpoints
- `POST /analytics/analytics` - Run comprehensive model analytics
- `POST /analytics/evaluate` - Evaluate a single model
- `POST /analytics/compare` - Compare two models

### Utility Endpoints
- `GET /` - API information
- `GET /health` - Health check

## Project Structure

```
easymodel/
├── main.py                          # FastAPI application entry point
├── main_ml.py                       # ML-enabled FastAPI application
├── main_simple.py                   # Simple FastAPI app (no ML dependencies)
├── easymodel/                       # Main package
│   ├── endpoints/
│   │   ├── finetuning.py           # Fine-tuning API endpoints
│   │   └── analytics.py            # Analytics API endpoints
│   └── utils/
│       ├── finetune2.py            # Core fine-tuning logic
│       └── text_analytics/
│           ├── gqs.py              # Grammar, Quality, Semantic metrics
│           ├── perplexity.py       # Perplexity analysis
│           ├── semantic.py         # Semantic similarity and drift
│           ├── tokenefficiency.py  # Token efficiency metrics
│           └── textanalytics.py    # Analytics orchestration
├── tests/                           # Comprehensive test suite
│   ├── test_analytics.py           # Core analytics tests
│   ├── test_evaluate.py            # Model evaluation tests
│   ├── test_finetune_local.py      # Fine-tuning infrastructure tests
│   ├── test_finetune.py            # Full fine-tuning tests
│   ├── test_direct.py              # Direct endpoint tests
│   ├── test_endpoint.py            # HTTP endpoint tests
│   └── run_all_tests.py            # Automated test runner
├── pyproject.toml                   # Poetry configuration
└── README.md                       # This comprehensive documentation
```

## Analytics Features

### GQS Metrics
- **Grammar**: Evaluates grammatical correctness using RoBERTa-based models
- **Fluency**: Measures text fluency using perplexity scores
- **Coherence**: Analyzes semantic coherence within text
- **Relevance**: Computes semantic similarity between predictions and references

### Perplexity Analysis
- Language model performance evaluation
- Training and validation perplexity tracking
- Batch processing for multiple texts

### Semantic Analysis
- Semantic similarity computation
- Semantic drift detection during fine-tuning
- Coherence analysis for text sequences

### Token Efficiency
- Summarization efficiency
- Question-answering efficiency
- Translation efficiency
- Paraphrasing efficiency
- Code generation efficiency

## Development

### Adding New Analytics
1. Create new functions in the appropriate `text_analytics` module
2. Update `textanalytics.py` to include the new functionality
3. Add corresponding API endpoints in `analytics.py`

### Adding New Task Types
1. Update the `finetune_model` function in `finetune2.py`
2. Add appropriate model loading logic
3. Update the API schema in `finetuning.py`

## Testing

EasyModel includes a comprehensive test suite to verify all functionality.

### Running Tests

#### Run All Tests
```bash
python tests/run_all_tests.py
```

#### Run Individual Tests
```bash
cd tests
python test_analytics.py      # Test core analytics functionality
python test_evaluate.py       # Test model evaluation endpoints
python test_finetune_local.py # Test fine-tuning infrastructure
python test_direct.py         # Test analytics endpoint directly
```

### Test Files

- **`test_analytics.py`** - Tests the core analytics functionality (perplexity, GQS metrics, semantic analysis)
- **`test_evaluate.py`** - Tests model evaluation endpoints and cross-dataset compatibility
- **`test_finetune_local.py`** - Tests fine-tuning infrastructure without requiring Hugging Face Hub
- **`test_finetune.py`** - Tests full fine-tuning endpoint (requires valid HF API key)
- **`test_direct.py`** - Tests analytics endpoint directly without HTTP server
- **`test_endpoint.py`** - Tests API endpoints via HTTP requests (requires running server)
- **`run_all_tests.py`** - Automated test runner that executes all tests and reports results

### Test Categories

#### ✅ **Core Functionality Tests** (All Passing)
- **Analytics Function**: Core text analytics and metrics computation
- **Model Evaluation**: Model assessment and comparison capabilities
- **Fine-tuning Infrastructure**: Training pipeline setup and model loading
- **Analytics Endpoint**: API endpoint functionality and data processing

#### ⚠️ **External Service Tests**
- **Fine-tuning with HF Hub**: Requires valid Hugging Face API key for model pushing
- **HTTP Endpoint Tests**: Requires running FastAPI server

### Test Results

All core functionality tests are passing:
- ✅ Analytics processing (perplexity, GQS, semantic analysis)
- ✅ Model evaluation (cross-dataset compatibility)
- ✅ Fine-tuning infrastructure (model loading, tokenization, training setup)
- ✅ API endpoints (request handling, response formatting)

### Test Environment

- Tests are designed to work with the conda environment (`easymodel`)
- Requires specific datasets (ag_news, squad, etc.) for comprehensive testing
- Fine-tuning tests work locally but require API keys for HF Hub integration
- All tests include proper error handling and timeout protection

## Contributing

If you work like to contribute to this project, feel free to do so! Listed below tells you how to do so:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with FastAPI, Transformers, and Hugging Face libraries
- Inspired by the need for comprehensive model evaluation tools
