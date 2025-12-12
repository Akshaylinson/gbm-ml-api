import pytest
import asyncio
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# Test data
valid_payload = {
    "rows": [
        {
            "age": 35,
            "income": 55000,
            "balance": 1200,
            "city": "A",
            "has_credit_card": 1
        }
    ]
}

headers = {"Authorization": "Bearer demo-key-123"}

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Advanced Gradient Boosting ML API" in response.json()["message"]

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_with_auth():
    response = client.post("/predict", json=valid_payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 1
    assert "prediction" in data["results"][0]
    assert "probability" in data["results"][0]

def test_prediction_without_auth():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 403

def test_invalid_input_validation():
    invalid_payload = {
        "rows": [
            {
                "age": 150,  # Invalid age
                "income": 55000,
                "balance": 1200,
                "city": "A",
                "has_credit_card": 1
            }
        ]
    }
    response = client.post("/predict", json=invalid_payload, headers=headers)
    assert response.status_code == 422

def test_metrics_endpoint():
    response = client.get("/metrics", headers=headers)
    assert response.status_code == 200

def test_models_endpoint():
    response = client.get("/models", headers=headers)
    assert response.status_code == 200
    assert "available_models" in response.json()

def test_rate_limiting():
    # This would need to be adjusted based on actual rate limits
    for _ in range(5):
        response = client.post("/predict", json=valid_payload, headers=headers)
        assert response.status_code in [200, 429]

if __name__ == "__main__":
    pytest.main([__file__])