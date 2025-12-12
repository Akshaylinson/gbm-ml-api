# Simple cache module
import json
import hashlib
from typing import Any, Optional

class SimpleCache:
    def __init__(self):
        self.cache = {}
        self.client = True  # Simulate Redis connection
    
    def get_key(self, data: dict) -> str:
        """Generate cache key from data"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache"""
        self.cache[key] = value
        # In a real implementation, you'd handle TTL
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()

cache = SimpleCache()