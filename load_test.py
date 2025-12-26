from locust import HttpUser, task, between
import json

class MLAPIUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        self.headers = {"Authorization": "Bearer demo-key-123"}
        self.payload = {
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
    
    @task(3)
    def predict(self):
        self.client.post("/predict", json=self.payload, headers=self.headers)
    
    @task(1)
    def health_check(self):
        self.client.get("/health")
    
    @task(1)
    def get_metrics(self):
        self.client.get("/metrics", headers=self.headers)

# Run with: locust -f load_test.py --host=http://localhost:8000


