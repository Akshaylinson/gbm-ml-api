import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib
import random
import shap
from datetime import datetime
import logging
from config import settings
from database import ModelVersion, ABTest, get_db

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.models = {}
        self.current_version = "v1"
        self.feature_names = ['age', 'income', 'balance', 'city', 'has_credit_card']
        self.explainers = {}
        self.ab_tests = {}
        
    def load_model(self, version: str = "v1", file_path: Optional[str] = None):
        """Load a model version"""
        try:
            if file_path:
                model_path = Path(file_path)
            else:
                model_path = Path(settings.model_path).parent / f"gbm_pipeline_{version}.pkl"
                if version == "v1":
                    model_path = Path(settings.model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model = joblib.load(model_path)
            self.models[version] = model
            
            # Initialize SHAP explainer for this model
            try:
                # Create a small sample for SHAP background
                sample_data = pd.DataFrame({
                    'age': [30, 45, 60],
                    'income': [50000, 75000, 100000],
                    'balance': [1000, 2000, 3000],
                    'city': ['A', 'B', 'C'],
                    'has_credit_card': [0, 1, 1]
                })
                
                # Transform the sample data
                transformed_sample = model.named_steps['preprocessor'].transform(sample_data)
                
                # Create SHAP explainer
                self.explainers[version] = shap.TreeExplainer(
                    model.named_steps['gbm'],
                    transformed_sample
                )
                
            except Exception as e:
                logger.warning(f"Could not initialize SHAP explainer for {version}: {e}")
            
            logger.info(f"Model {version} loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model {version}: {e}")
            raise
    
    def get_model(self, version: str = "v1"):
        """Get a specific model version"""
        return self.models.get(version)
    
    def list_models(self) -> List[str]:
        """List all loaded model versions"""
        return list(self.models.keys())
    
    def set_current_version(self, version: str):
        """Set the current default model version"""
        if version in self.models:
            self.current_version = version
            logger.info(f"Current model version set to {version}")
        else:
            raise ValueError(f"Model version {version} not loaded")
    
    def get_feature_importance(self, version: str = "v1") -> Dict[str, float]:
        """Get feature importance for a model version"""
        model = self.get_model(version)
        if not model or not hasattr(model.named_steps['gbm'], 'feature_importances_'):
            return {}
        
        try:
            importances = model.named_steps['gbm'].feature_importances_
            preprocessor = model.named_steps['preprocessor']
            
            # Get feature names after preprocessing
            feature_names = []
            
            # Numeric features
            feature_names.extend(['age', 'income', 'balance'])
            
            # Categorical features (after one-hot encoding)
            try:
                cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['city', 'has_credit_card'])
                feature_names.extend(cat_features)
            except:
                # Fallback if feature names not available
                feature_names.extend(['city_A', 'city_B', 'city_C', 'has_credit_card'])
            
            return dict(zip(feature_names[:len(importances)], importances.astype(float)))
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def explain_prediction(self, data: pd.DataFrame, version: str = "v1") -> Dict[str, Any]:
        """Get SHAP explanation for a prediction"""
        if version not in self.explainers:
            return {"error": "SHAP explainer not available for this model version"}
        
        try:
            model = self.get_model(version)
            explainer = self.explainers[version]
            
            # Transform the input data
            transformed_data = model.named_steps['preprocessor'].transform(data)
            
            # Get SHAP values
            shap_values = explainer.shap_values(transformed_data)
            
            # If binary classification, take positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
            # Get feature names
            feature_importance = self.get_feature_importance(version)
            feature_names = list(feature_importance.keys())
            
            # Create explanation dictionary
            explanation = {}
            for i, feature in enumerate(feature_names[:len(shap_values[0])]):
                explanation[feature] = float(shap_values[0][i])
            
            return {
                "shap_values": explanation,
                "base_value": float(explainer.expected_value),
                "prediction_explanation": self._interpret_shap_values(explanation)
            }
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {"error": str(e)}
    
    def _interpret_shap_values(self, shap_values: Dict[str, float]) -> str:
        """Interpret SHAP values into human-readable explanation"""
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        
        explanations = []
        for feature, value in sorted_features[:3]:  # Top 3 features
            direction = "increases" if value > 0 else "decreases"
            explanations.append(f"{feature} {direction} prediction by {abs(value):.3f}")
        
        return "; ".join(explanations)
    
    def create_ab_test(self, test_name: str, model_a: str, model_b: str, 
                      traffic_split: float = 0.5) -> bool:
        """Create an A/B test between two model versions"""
        try:
            if model_a not in self.models or model_b not in self.models:
                raise ValueError("Both models must be loaded")
            
            self.ab_tests[test_name] = {
                "model_a": model_a,
                "model_b": model_b,
                "traffic_split": traffic_split,
                "created_at": datetime.now(),
                "results": {"a": [], "b": []}
            }
            
            # Store in database
            db = next(get_db())
            ab_test = ABTest(
                test_name=test_name,
                model_a=model_a,
                model_b=model_b,
                traffic_split=traffic_split
            )
            db.add(ab_test)
            db.commit()
            db.close()
            
            logger.info(f"A/B test '{test_name}' created: {model_a} vs {model_b}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create A/B test: {e}")
            return False
    
    def get_ab_test_model(self, test_name: str, user_id: str) -> str:
        """Get model version for A/B test based on user"""
        if test_name not in self.ab_tests:
            return self.current_version
        
        test = self.ab_tests[test_name]
        
        # Use user_id hash for consistent assignment
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        
        # Assign to model A or B based on traffic split
        if (user_hash % 100) / 100 < test["traffic_split"]:
            return test["model_a"]
        else:
            return test["model_b"]
    
    def log_ab_test_result(self, test_name: str, model_version: str, 
                          prediction: int, probability: float):
        """Log A/B test result"""
        if test_name in self.ab_tests:
            test = self.ab_tests[test_name]
            if model_version == test["model_a"]:
                test["results"]["a"].append({"pred": prediction, "prob": probability})
            elif model_version == test["model_b"]:
                test["results"]["b"].append({"pred": prediction, "prob": probability})
    
    def get_ab_test_results(self, test_name: str) -> Dict[str, Any]:
        """Get A/B test results"""
        if test_name not in self.ab_tests:
            return {"error": "Test not found"}
        
        test = self.ab_tests[test_name]
        results_a = test["results"]["a"]
        results_b = test["results"]["b"]
        
        if not results_a or not results_b:
            return {"status": "insufficient_data"}
        
        # Calculate metrics
        metrics_a = self._calculate_model_metrics(results_a)
        metrics_b = self._calculate_model_metrics(results_b)
        
        return {
            "test_name": test_name,
            "model_a": test["model_a"],
            "model_b": test["model_b"],
            "metrics_a": metrics_a,
            "metrics_b": metrics_b,
            "sample_sizes": {"a": len(results_a), "b": len(results_b)},
            "winner": self._determine_winner(metrics_a, metrics_b)
        }
    
    def _calculate_model_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate metrics for model results"""
        if not results:
            return {}
        
        predictions = [r["pred"] for r in results]
        probabilities = [r["prob"] for r in results]
        
        return {
            "positive_rate": sum(predictions) / len(predictions),
            "avg_probability": np.mean(probabilities),
            "confidence": np.std(probabilities),
            "sample_size": len(results)
        }
    
    def _determine_winner(self, metrics_a: Dict, metrics_b: Dict) -> str:
        """Determine A/B test winner based on metrics"""
        if not metrics_a or not metrics_b:
            return "inconclusive"
        
        # Simple comparison based on average probability
        # In practice, you'd use proper statistical tests
        if metrics_a["avg_probability"] > metrics_b["avg_probability"]:
            return "model_a"
        elif metrics_b["avg_probability"] > metrics_a["avg_probability"]:
            return "model_b"
        else:
            return "tie"
    
    def rollback_model(self, target_version: str) -> bool:
        """Rollback to a previous model version"""
        try:
            if target_version not in self.models:
                self.load_model(target_version)
            
            self.current_version = target_version
            logger.info(f"Rolled back to model version {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

# Global model manager instance
model_manager = ModelManager()