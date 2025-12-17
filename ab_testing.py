# Simple A/B testing module
import random
from typing import Dict, Any

class SimpleABManager:
    def __init__(self):
        self.tests = {}
    
    def get_variant(self, test_name: str, user_id: str) -> str:
        """Get variant for user"""
        # Simple hash-based assignment
        hash_val = hash(f"{test_name}_{user_id}") % 100
        return "v1" if hash_val < 50 else "v1"  # Always return v1 for now
    
    def create_test(self, test_name: str, variants: Dict[str, float]):
        """Create A/B test"""
        self.tests[test_name] = variants

ab_manager = SimpleABManager()

