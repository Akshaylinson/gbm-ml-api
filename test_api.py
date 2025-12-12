#!/usr/bin/env python3
"""
Simple test script to verify the API works correctly
"""
import requests
import json

def test_api():
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code} - {response.json()}")
    except requests.exceptions.ConnectionError:
        print("API is not running. Start it with: python app.py")
        return
    
    # Test prediction endpoint
    test_data = {
        "rows": [
            {
                "age": 35,
                "income": 55000,
                "balance": 1200,
                "city": "A",
                "has_credit_card": 1
            },
            {
                "age": 25,
                "income": 30000,
                "balance": 500,
                "city": "B",
                "has_credit_card": 0
            }
        ]
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=test_data)
        print(f"Prediction: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print(f"Results: {json.dumps(results, indent=2)}")
        else:
            print(f"Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to API")

if __name__ == "__main__":
    test_api()