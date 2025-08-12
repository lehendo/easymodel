# THIS FILE NEEDS A LOT OF WORK

from http import HTTPStatus
from fastapi.testclient import TestClient
from main import app
from structlog.testing import capture_logs
from unittest.mock import patch, mock_open
import json
import tempfile

client = TestClient(app)

# Health check test
def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "status": "ok",
        "details": "Models and tokenizer loaded successfully"
    }

# Root endpoint test
def test_read_main():
    with capture_logs() as cap_logs:
        response = client.get("/")
        assert {"event": "In root path", "log_level": "info"} in cap_logs
        assert response.status_code == HTTPStatus.OK
        assert response.json() == {"message": "Welcome to the eRegion API!"}

# Test model evaluation with successful scenarios
@patch("polar.textanalytics.evaluate_model", return_value={"perplexity": 1.0, "gqs": 0.5})
@patch("builtins.open", new_callable=mock_open, read_data=json.dumps({
    "original_texts": ["This is an example text."],
    "generated_texts": ["This is a generated text."]
}))
@patch("os.path.exists", return_value=True)
def test_evaluate_model(mock_perplexity, mock_exists, mock_file):
    request_data = {
        "model_url": "bert-base-uncased",
        "dataset": "dataset_path_placeholder",
        "prompts": ["What is AI?", "Explain neural networks."],
        "references": ["Artificial Intelligence (AI) is a field of computer science.",
                       "Neural networks are a subset of machine learning."]
    }

    # Test encoder-only model
    response = client.post("/evaluate", json=request_data)
    assert response.status_code == HTTPStatus.OK
    assert response.json()["success"] is True
    assert "perplexity" in response.json()["data"]

    # Test decoder-only model
    request_data["model_url"] = "gpt2"
    response = client.post("/evaluate", json=request_data)
    assert response.status_code == HTTPStatus.OK
    assert response.json()["success"] is True

    # Test encoder-decoder model
    request_data["model_url"] = "t5-small"
    response = client.post("/evaluate", json=request_data)
    assert response.status_code == HTTPStatus.OK
    assert response.json()["success"] is True

# Test model evaluation with invalid model or dataset
@patch("builtins.open", new_callable=mock_open, read_data=json.dumps({}))
@patch("os.path.exists", return_value=False)
def test_invalid_evaluate_model(mock_exists, mock_file):
    request_data = {
        "model_url": "invalid-model-name",
        "dataset": "",
        "prompts": [],
        "references": []
    }

    # Invalid model URL
    response = client.post("/evaluate", json=request_data)
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "detail" in response.json()

    # Nonexistent dataset path
    request_data["model_url"] = "bert-base-uncased"
    request_data["dataset"] = "nonexistent_path"
    response = client.post("/evaluate", json=request_data)
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert "detail" in response.json()

# Test model evaluation with a temporary file
def test_evaluate_with_tempfile():
    temp_dataset = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    dataset_content = {
        "original_texts": ["This is an example text."],
        "generated_texts": ["This is a generated text."]
    }
    with open(temp_dataset.name, "w") as f:
        json.dump(dataset_content, f)

    request_data = {
        "model_url": "bert-base-uncased",
        "dataset": temp_dataset.name,
        "prompts": ["What is AI?"],
        "references": ["Artificial Intelligence (AI) is a field of computer science."]
    }

    response = client.post("/evaluate", json=request_data)
    assert response.status_code == HTTPStatus.OK
    assert response.json()["success"] is True

    temp_dataset.close()

# Test fine-tuning endpoint
@patch("polar.fine_tuning.fine_tune_model", return_value={"status": "success", "model": "fine_tuned_model"})
def test_fine_tune_model(mock_fine_tune):
    request_data = {
        "model_url": "bert-base-uncased",
        "dataset_url": "dataset_path_placeholder",
        "output_repo": "new_model_repo"
    }

    response = client.post("/fine_tune", json=request_data)
    assert response.status_code == HTTPStatus.OK
    assert response.json()["success"] is True
    assert "model" in response.json()["details"]
