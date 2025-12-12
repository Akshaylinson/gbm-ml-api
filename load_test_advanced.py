from locust import HttpUser, task, between
import json
import random
import time

class MLAPIUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup for each user"""
        self.api_key = "demo-key-123"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Test data variations
        self.test_data_pool = [
            {
                "age": random.randint(18, 80),
                "income": random.randint(20000, 150000),
                "balance": random.randint(-5000, 50000),
                "city": random.choice(["A", "B", "C"]),
                "has_credit_card": random.choice([0, 1])
            }
            for _ in range(100)
        ]
    
    @task(10)
    def predict_single(self):
        """Test single prediction - most common use case"""
        data = {
            "rows": [random.choice(self.test_data_pool)],
            "model_version": "v1",
            "explain": random.choice([True, False]),
            "use_business_rules": True
        }
        
        with self.client.post("/predict", json=data, headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if "results" in result and len(result["results"]) > 0:
                    response.success()
                else:
                    response.failure("Invalid response format")
            elif response.status_code == 429:
                response.success()  # Rate limiting is expected
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(3)
    def predict_batch_small(self):
        """Test small batch predictions"""
        batch_size = random.randint(2, 10)
        data = {
            "rows": random.sample(self.test_data_pool, batch_size),
            "model_version": "v1",
            "explain": False,
            "use_business_rules": True
        }
        
        with self.client.post("/predict", json=data, headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if len(result["results"]) == batch_size:
                    response.success()
                else:
                    response.failure("Batch size mismatch")
            elif response.status_code == 429:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(1)
    def predict_batch_large_async(self):
        """Test large batch with async processing"""
        batch_size = random.randint(50, 100)
        data = {
            "rows": random.sample(self.test_data_pool, min(batch_size, len(self.test_data_pool))),
            "model_version": "v1",
            "async_processing": True
        }
        
        with self.client.post("/batch-predict", json=data, headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if "job_id" in result:
                    response.success()
                    # Optionally check job status
                    job_id = result["job_id"]
                    if job_id != "sync":
                        self.check_job_status(job_id)
                else:
                    response.failure("No job_id in response")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    def check_job_status(self, job_id):
        """Check async job status"""
        with self.client.get(f"/batch-predict/{job_id}", headers=self.headers, catch_response=True) as response:
            if response.status_code in [200, 404]:  # 404 is acceptable for completed jobs
                response.success()
            else:
                response.failure(f"Job status check failed: {response.status_code}")
    
    @task(2)
    def get_metrics(self):
        """Test metrics endpoint"""
        with self.client.get("/metrics", headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics endpoint failed: {response.status_code}")
    
    @task(1)
    def get_models(self):
        """Test model listing"""
        with self.client.get("/models", headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if "available_models" in result:
                    response.success()
                else:
                    response.failure("Invalid models response")
            else:
                response.failure(f"Models endpoint failed: {response.status_code}")
    
    @task(1)
    def get_history(self):
        """Test user history"""
        with self.client.get("/history?limit=10", headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"History endpoint failed: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Test health endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if "status" in result:
                    response.success()
                else:
                    response.failure("Invalid health response")
            else:
                response.failure(f"Health check failed: {response.status_code}")

class AdminUser(HttpUser):
    """Simulate admin user behavior"""
    wait_time = between(5, 15)  # Admins make fewer requests
    
    def on_start(self):
        self.api_key = "admin-key-456"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    @task(1)
    def get_cache_stats(self):
        """Test admin cache stats"""
        with self.client.get("/admin/cache/stats", headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Cache stats failed: {response.status_code}")
    
    @task(1)
    def get_rate_limit_stats(self):
        """Test rate limit stats"""
        user_id = "demo-user"
        with self.client.get(f"/admin/rate-limits/{user_id}", headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Rate limit stats failed: {response.status_code}")

class StressTestUser(HttpUser):
    """High-frequency user for stress testing"""
    wait_time = between(0.1, 0.5)  # Very fast requests
    
    def on_start(self):
        self.api_key = "demo-key-123"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Simple test data for fast requests
        self.simple_data = {
            "rows": [{
                "age": 35,
                "income": 50000,
                "balance": 2000,
                "city": "A",
                "has_credit_card": 1
            }],
            "model_version": "v1",
            "explain": False,
            "use_business_rules": False  # Faster without business rules
        }
    
    @task
    def rapid_predictions(self):
        """Rapid-fire predictions for stress testing"""
        with self.client.post("/predict", json=self.simple_data, headers=self.headers, catch_response=True) as response:
            if response.status_code in [200, 429]:  # Accept rate limiting
                response.success()
            else:
                response.failure(f"Stress test failed: {response.status_code}")

# Custom test scenarios
class ScenarioTest:
    """Custom test scenarios for specific use cases"""
    
    @staticmethod
    def test_cache_effectiveness():
        """Test cache hit rates"""
        import requests
        
        base_url = "http://localhost:8000"
        headers = {"Authorization": "Bearer demo-key-123"}
        
        # Same request multiple times
        data = {
            "rows": [{
                "age": 35,
                "income": 50000,
                "balance": 2000,
                "city": "A",
                "has_credit_card": 1
            }],
            "model_version": "v1"
        }
        
        times = []
        for i in range(10):
            start = time.time()
            response = requests.post(f"{base_url}/predict", json=data, headers=headers)
            end = time.time()
            
            if response.status_code == 200:
                times.append(end - start)
                result = response.json()
                print(f"Request {i+1}: {(end-start)*1000:.2f}ms, Cache hit: {result.get('cache_hit', False)}")
        
        print(f"Average response time: {np.mean(times)*1000:.2f}ms")
        print(f"Min response time: {np.min(times)*1000:.2f}ms")
        print(f"Max response time: {np.max(times)*1000:.2f}ms")
    
    @staticmethod
    def test_rate_limiting():
        """Test rate limiting behavior"""
        import requests
        
        base_url = "http://localhost:8000"
        headers = {"Authorization": "Bearer demo-key-123"}
        
        data = {
            "rows": [{
                "age": 35,
                "income": 50000,
                "balance": 2000,
                "city": "A",
                "has_credit_card": 1
            }]
        }
        
        success_count = 0
        rate_limited_count = 0
        
        for i in range(150):  # Exceed rate limit
            response = requests.post(f"{base_url}/predict", json=data, headers=headers)
            
            if response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:
                rate_limited_count += 1
                print(f"Rate limited at request {i+1}")
            
            time.sleep(0.1)  # Small delay
        
        print(f"Successful requests: {success_count}")
        print(f"Rate limited requests: {rate_limited_count}")

if __name__ == "__main__":
    # Run specific scenarios
    print("Testing cache effectiveness...")
    ScenarioTest.test_cache_effectiveness()
    
    print("\nTesting rate limiting...")
    ScenarioTest.test_rate_limiting()