# Simple model registry
from datetime import datetime
from typing import Dict, Any

class SimpleModelRegistry:
    def __init__(self):
        self.models = {}
        self.active_model = "v1"
    
    def register_model(self, version: str, path: str, metadata: Dict[str, Any]):
        """Register a model"""
        self.models[version] = {
            "path": path,
            "metadata": metadata,
            "registered_at": datetime.now()
        }
    
    def activate_model(self, version: str):
        """Activate a model version"""
        if version in self.models:
            self.active_model = version
    
    def list_models(self) -> Dict[str, Any]:
        """List all registered models"""
        return self.models

registry = SimpleModelRegistry()