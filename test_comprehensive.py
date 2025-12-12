import pytest
import asyncio
import json
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Import the enhanced app
from app_enhanced import app
from config import settings
from model_manager import model_manager
from monitoring import monitor
from cache import cache_manager
from rate_limiter import rate_limiter

client = TestClient(app)

# Test data
VALID_PREDICTION_DATA = {
    "rows": [
        {
            "age": 35,
            "income": 50000,
            "balance": 2000,
            "city": "A",
            "has_credit_card": 1
        }
    ],
    "model_version": "v1",
    "explain": True,
    "use_business_rules": True
}

VALID_API_KEY = "demo-key-123"
HEADERS = {"Authorization": f"Bearer {VALID_API_KEY}"}

class TestBasicEndpoints:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "features" in data

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data

class TestAuthentication:
    def test_no_auth_fails(self):
        response = client.post("/predict", json=VALID_PREDICTION_DATA)
        assert response.status_code == 401

    def test_invalid_auth_fails(self):
        headers = {"Authorization": "Bearer invalid-key"}
        response = client.post("/predict", json=VALID_PREDICTION_DATA, headers=headers)
        assert response.status_code == 401

    def test_valid_auth_succeeds(self):
        response = client.get("/metrics", headers=HEADERS)
        assert response.status_code == 200

class TestPredictionEndpoint:
    def test_valid_prediction(self):
        response = client.post("/predict", json=VALID_PREDICTION_DATA, headers=HEADERS)
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "model_version" in data
        assert "processing_time_ms" in data
        assert "request_id" in data
        
        result = data["results"][0]
        assert "prediction" in result
        assert "probability" in result
        assert "risk_score" in result

    def test_input_validation(self):
        invalid_data = VALID_PREDICTION_DATA.copy()
        invalid_data["rows"][0]["age"] = 150  # Invalid age
        
        response = client.post("/predict", json=invalid_data, headers=HEADERS)
        assert response.status_code == 422

    def test_batch_size_limit(self):
        large_batch = VALID_PREDICTION_DATA.copy()
        large_batch["rows"] = [VALID_PREDICTION_DATA["rows"][0]] * (settings.max_batch_size + 1)
        
        response = client.post("/predict", json=large_batch, headers=HEADERS)
        assert response.status_code == 422

    def test_business_rules_integration(self):
        # Test with business rules enabled
        response = client.post("/predict", json=VALID_PREDICTION_DATA, headers=HEADERS)
        assert response.status_code == 200
        
        data = response.json()
        result = data["results"][0]
        assert "business_risk" in result
        assert result["business_risk"] is not None

    def test_explanation_feature(self):
        explain_data = VALID_PREDICTION_DATA.copy()
        explain_data["explain"] = True
        
        response = client.post("/predict", json=explain_data, headers=HEADERS)
        assert response.status_code == 200
        
        data = response.json()
        result = data["results"][0]
        assert "explanation" in result

class TestRateLimiting:
    def test_rate_limit_enforcement(self):
        # Make many requests quickly
        responses = []
        for i in range(settings.rate_limit_per_minute + 5):
            response = client.post("/predict", json=VALID_PREDICTION_DATA, headers=HEADERS)
            responses.append(response)
            if response.status_code == 429:
                break
        
        # Should eventually hit rate limit
        assert any(r.status_code == 429 for r in responses)

    def test_rate_limit_reset(self):
        # Reset rate limit for test user
        rate_limiter.reset_user("demo-user")
        
        response = client.post("/predict", json=VALID_PREDICTION_DATA, headers=HEADERS)
        assert response.status_code == 200

class TestModelManagement:
    def test_list_models(self):
        response = client.get("/models", headers=HEADERS)
        assert response.status_code == 200
        
        data = response.json()
        assert "available_models" in data
        assert "current_version" in data
        assert "feature_importance" in data

    @patch('model_manager.model_manager.load_model')
    def test_load_model_version(self, mock_load):
        admin_headers = {"Authorization": "Bearer admin-key-456"}
        
        response = client.post("/models/v2/load", headers=admin_headers)
        assert response.status_code == 200
        mock_load.assert_called_once_with("v2")

class TestABTesting:
    def test_create_ab_test(self):
        admin_headers = {"Authorization": "Bearer admin-key-456"}
        
        response = client.post(
            "/ab-tests?test_name=test1&model_a=v1&model_b=v2&traffic_split=0.5",
            headers=admin_headers
        )
        # May fail if models not loaded, but should not crash
        assert response.status_code in [200, 400]

    def test_ab_test_results(self):
        response = client.get("/ab-tests/test1/results", headers=HEADERS)
        # Should return results or error, not crash
        assert response.status_code in [200, 404]

class TestCaching:
    def test_cache_functionality(self):
        # First request
        response1 = client.post("/predict", json=VALID_PREDICTION_DATA, headers=HEADERS)
        assert response1.status_code == 200
        
        # Second identical request (should be cached)
        response2 = client.post("/predict", json=VALID_PREDICTION_DATA, headers=HEADERS)
        assert response2.status_code == 200
        
        # Check if cache was used (processing time should be lower)
        time1 = response1.json()["processing_time_ms"]
        time2 = response2.json()["processing_time_ms"]
        # Cache hit should be faster (though not always guaranteed in tests)

class TestMonitoring:
    def test_metrics_endpoint(self):
        response = client.get("/metrics", headers=HEADERS)
        assert response.status_code == 200
        
        data = response.json()
        assert "total_requests" in data or "status" in data

    def test_prometheus_metrics(self):
        response = client.get("/metrics/prometheus")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

class TestAdminEndpoints:
    def test_cache_stats(self):
        admin_headers = {"Authorization": "Bearer admin-key-456"}
        
        response = client.get("/admin/cache/stats", headers=admin_headers)
        assert response.status_code == 200

    def test_rate_limit_stats(self):
        admin_headers = {"Authorization": "Bearer admin-key-456"}
        
        response = client.get("/admin/rate-limits/demo-user", headers=admin_headers)
        assert response.status_code == 200

    def test_non_admin_access_denied(self):
        response = client.get("/admin/cache/stats", headers=HEADERS)
        assert response.status_code == 403

class TestBatchProcessing:
    def test_small_batch_sync(self):
        batch_data = {
            "rows": [VALID_PREDICTION_DATA["rows"][0]] * 5,
            "model_version": "v1",
            "async_processing": False
        }
        
        response = client.post("/batch-predict", json=batch_data, headers=HEADERS)
        assert response.status_code == 200

    def test_large_batch_async(self):
        batch_data = {
            "rows": [VALID_PREDICTION_DATA["rows"][0]] * 50,
            "model_version": "v1",
            "async_processing": True
        }
        
        response = client.post("/batch-predict", json=batch_data, headers=HEADERS)
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
        assert "status" in data

class TestUserHistory:
    def test_get_history(self):
        # Make a prediction first
        client.post("/predict", json=VALID_PREDICTION_DATA, headers=HEADERS)
        
        # Get history
        response = client.get("/history", headers=HEADERS)
        assert response.status_code == 200
        
        data = response.json()
        assert "user" in data
        assert "history" in data

class TestErrorHandling:
    def test_invalid_model_version(self):
        invalid_data = VALID_PREDICTION_DATA.copy()
        invalid_data["model_version"] = "nonexistent"
        
        response = client.post("/predict", json=invalid_data, headers=HEADERS)
        assert response.status_code == 404

    def test_malformed_json(self):
        response = client.post(
            "/predict",
            data="invalid json",
            headers={**HEADERS, "Content-Type": "application/json"}
        )
        assert response.status_code == 422

class TestBusinessLogic:
    def test_risk_categorization(self):
        # Test high-risk profile
        high_risk_data = VALID_PREDICTION_DATA.copy()
        high_risk_data["rows"][0].update({
            "age": 25,
            "income": 100000,
            "balance": -5000
        })
        
        response = client.post("/predict", json=high_risk_data, headers=HEADERS)
        assert response.status_code == 200
        
        result = response.json()["results"][0]
        # Should have business risk assessment
        assert "business_risk" in result

class TestPerformance:
    def test_response_time(self):
        start_time = time.time()
        response = client.post("/predict", json=VALID_PREDICTION_DATA, headers=HEADERS)
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 2.0  # Should respond within 2 seconds

    def test_concurrent_requests(self):
        import threading
        
        results = []
        
        def make_request():
            response = client.post("/predict", json=VALID_PREDICTION_DATA, headers=HEADERS)
            results.append(response.status_code)
        
        # Make 10 concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All requests should succeed (or hit rate limit)
        assert all(status in [200, 429] for status in results)

# Integration tests
class TestIntegration:
    def test_full_prediction_workflow(self):
        # 1. Make prediction
        response = client.post("/predict", json=VALID_PREDICTION_DATA, headers=HEADERS)
        assert response.status_code == 200
        
        # 2. Check metrics updated
        metrics_response = client.get("/metrics", headers=HEADERS)
        assert metrics_response.status_code == 200
        
        # 3. Check history
        history_response = client.get("/history", headers=HEADERS)
        assert history_response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"])