import time
import asyncio
import aiohttp
import statistics
from concurrent.futures import ThreadPoolExecutor
import json

class APIBenchmark:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.headers = {"Authorization": "Bearer demo-key-123"}
        self.test_payload = {
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
    
    async def single_request(self, session):
        start_time = time.time()
        try:
            async with session.post(
                f"{self.base_url}/predict",
                json=self.test_payload,
                headers=self.headers
            ) as response:
                await response.json()
                return time.time() - start_time, response.status == 200
        except Exception as e:
            return time.time() - start_time, False
    
    async def run_concurrent_test(self, num_requests=100, concurrency=10):
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(concurrency)
            
            async def bounded_request():
                async with semaphore:
                    return await self.single_request(session)
            
            tasks = [bounded_request() for _ in range(num_requests)]
            results = await asyncio.gather(*tasks)
        
        response_times = [r[0] for r in results]
        success_count = sum(1 for r in results if r[1])
        
        return {
            "total_requests": num_requests,
            "successful_requests": success_count,
            "success_rate": success_count / num_requests,
            "avg_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "p95_response_time": sorted(response_times)[int(0.95 * len(response_times))],
            "p99_response_time": sorted(response_times)[int(0.99 * len(response_times))],
            "requests_per_second": num_requests / max(response_times)
        }

async def main():
    benchmark = APIBenchmark()
    
    print("Running API Performance Benchmark...")
    print("=" * 50)
    
    # Test different concurrency levels
    for concurrency in [1, 5, 10, 20]:
        print(f"\nTesting with {concurrency} concurrent requests:")
        results = await benchmark.run_concurrent_test(100, concurrency)
        
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"Avg Response Time: {results['avg_response_time']*1000:.2f}ms")
        print(f"P95 Response Time: {results['p95_response_time']*1000:.2f}ms")
        print(f"Requests/sec: {results['requests_per_second']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
