# Advanced Model Interpretability with SHAP
import numpy as np
import pandas as pd
import shap
import joblib
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelInterpreter:
    """Advanced model interpretability using SHAP and custom methods"""
    
    def __init__(self, model_path: str):
        self.model = None
        self.explainer = None
        self.feature_names = ['age', 'income', 'balance', 'city', 'has_credit_card']
        self.background_data = None
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load model and initialize SHAP explainer"""
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            # Create background dataset for SHAP
            self._create_background_data()
            
            # Initialize SHAP explainer
            self._initialize_explainer()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _create_background_data(self):
        """Create representative background data for SHAP"""
        # Generate synthetic background data based on typical ranges
        np.random.seed(42)
        n_samples = 100
        
        background = pd.DataFrame({
            'age': np.random.normal(45, 15, n_samples).clip(18, 100),
            'income': np.random.normal(50000, 20000, n_samples).clip(-50000, 500000),
            'balance': np.random.normal(2000, 3000, n_samples).clip(-10000, 100000),
            'city': np.random.choice(['A', 'B', 'C'], n_samples),
            'has_credit_card': np.random.choice([0, 1], n_samples)
        })
        
        self.background_data = background
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer"""
        try:
            # Use TreeExplainer for tree-based models
            if hasattr(self.model.named_steps['gbm'], 'estimators_'):
                self.explainer = shap.TreeExplainer(
                    self.model.named_steps['gbm'],
                    self.model.named_steps['preprocessor'].transform(self.background_data)
                )
            else:
                # Fallback to KernelExplainer
                def model_predict(X):
                    return self.model.predict_proba(
                        pd.DataFrame(X, columns=self.feature_names)
                    )[:, 1]
                
                self.explainer = shap.KernelExplainer(
                    model_predict,
                    self.background_data.values
                )
            
            logger.info("SHAP explainer initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.explainer = None
    
    def explain_prediction(self, input_data: Dict[str, Any], 
                          explanation_type: str = "local") -> Dict[str, Any]:
        """
        Generate explanations for predictions
        
        Args:
            input_data: Input features as dictionary
            explanation_type: 'local', 'global', or 'both'
        
        Returns:
            Dictionary containing explanations
        """
        try:
            df = pd.DataFrame([input_data])
            
            explanations = {
                "feature_importance": self._get_feature_importance(),
                "prediction_confidence": self._calculate_confidence(df),
                "risk_factors": self._identify_risk_factors(input_data),
                "counterfactuals": self._generate_counterfactuals(input_data)
            }
            
            if self.explainer and explanation_type in ["local", "both"]:
                explanations["shap_values"] = self._get_shap_explanation(df)
            
            if explanation_type in ["global", "both"]:
                explanations["global_importance"] = self._get_global_importance()
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {"error": str(e)}
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get model feature importance"""
        try:
            if hasattr(self.model.named_steps['gbm'], 'feature_importances_'):
                importances = self.model.named_steps['gbm'].feature_importances_
                
                # Get feature names after preprocessing
                preprocessor = self.model.named_steps['preprocessor']
                feature_names = ['age', 'income', 'balance']
                
                try:
                    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['city', 'has_credit_card'])
                    feature_names.extend(cat_features)
                except:
                    feature_names.extend(['city_A', 'city_B', 'city_C', 'has_credit_card_0', 'has_credit_card_1'])
                
                return dict(zip(feature_names[:len(importances)], importances.tolist()))
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def _calculate_confidence(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate prediction confidence metrics"""
        try:
            probabilities = self.model.predict_proba(df)[:, 1]
            
            # Confidence based on distance from decision boundary (0.5)
            confidence = abs(probabilities - 0.5) * 2
            
            # Uncertainty quantification
            entropy = -probabilities * np.log2(probabilities + 1e-8) - \
                     (1 - probabilities) * np.log2(1 - probabilities + 1e-8)
            
            return {
                "confidence_score": float(confidence[0]),
                "entropy": float(entropy[0]),
                "probability": float(probabilities[0])
            }
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return {}
    
    def _identify_risk_factors(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify key risk factors in the input"""
        risk_factors = []
        
        try:
            # Age-based risk
            age = input_data.get('age', 0)
            if age < 25:
                risk_factors.append({
                    "factor": "young_age",
                    "description": "Young age may indicate higher risk",
                    "impact": "negative",
                    "severity": "medium"
                })
            elif age > 70:
                risk_factors.append({
                    "factor": "elderly_age",
                    "description": "Advanced age may indicate higher risk",
                    "impact": "negative",
                    "severity": "low"
                })
            
            # Income-balance ratio
            income = input_data.get('income', 0)
            balance = input_data.get('balance', 0)
            
            if income > 0 and balance < 0:
                risk_factors.append({
                    "factor": "negative_balance",
                    "description": "Negative balance despite positive income",
                    "impact": "negative",
                    "severity": "high"
                })
            
            if income > 0 and balance / income > 0.5:
                risk_factors.append({
                    "factor": "high_balance_ratio",
                    "description": "High balance relative to income",
                    "impact": "positive",
                    "severity": "medium"
                })
            
            # Credit card factor
            if input_data.get('has_credit_card', 0) == 0 and income > 50000:
                risk_factors.append({
                    "factor": "no_credit_card_high_income",
                    "description": "High income but no credit card",
                    "impact": "negative",
                    "severity": "low"
                })
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
            return []
    
    def _generate_counterfactuals(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations"""
        counterfactuals = []
        
        try:
            original_pred = self.model.predict_proba(pd.DataFrame([input_data]))[0, 1]
            
            # Test modifications
            modifications = [
                {"age": input_data["age"] + 10, "description": "10 years older"},
                {"income": input_data["income"] * 1.5, "description": "50% higher income"},
                {"balance": max(0, input_data["balance"] * 2), "description": "Double the balance"},
                {"has_credit_card": 1 - input_data["has_credit_card"], "description": "Opposite credit card status"}
            ]
            
            for mod in modifications:
                modified_data = input_data.copy()
                modified_data.update({k: v for k, v in mod.items() if k != "description"})
                
                new_pred = self.model.predict_proba(pd.DataFrame([modified_data]))[0, 1]
                impact = new_pred - original_pred
                
                if abs(impact) > 0.05:  # Only significant changes
                    counterfactuals.append({
                        "modification": mod["description"],
                        "probability_change": float(impact),
                        "new_probability": float(new_pred),
                        "direction": "increase" if impact > 0 else "decrease"
                    })
            
            return counterfactuals
            
        except Exception as e:
            logger.error(f"Error generating counterfactuals: {e}")
            return []
    
    def _get_shap_explanation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get SHAP values for local explanation"""
        try:
            if self.explainer is None:
                return {}
            
            # Transform data for SHAP
            X_transformed = self.model.named_steps['preprocessor'].transform(df)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_transformed)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            
            # Get feature names
            feature_names = self._get_transformed_feature_names()
            
            return {
                "shap_values": shap_values[0].tolist()[:len(feature_names)],
                "feature_names": feature_names,
                "base_value": float(self.explainer.expected_value)
            }
            
        except Exception as e:
            logger.error(f"Error getting SHAP explanation: {e}")
            return {}
    
    def _get_global_importance(self) -> Dict[str, float]:
        """Get global feature importance"""
        try:
            if self.explainer is None:
                return self._get_feature_importance()
            
            # Calculate SHAP values for background data
            X_transformed = self.model.named_steps['preprocessor'].transform(self.background_data)
            shap_values = self.explainer.shap_values(X_transformed)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Calculate mean absolute SHAP values
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            feature_names = self._get_transformed_feature_names()
            
            return dict(zip(feature_names, mean_shap.tolist()))
            
        except Exception as e:
            logger.error(f"Error getting global importance: {e}")
            return self._get_feature_importance()
    
    def _get_transformed_feature_names(self) -> List[str]:
        """Get feature names after preprocessing"""
        try:
            preprocessor = self.model.named_steps['preprocessor']
            feature_names = ['age', 'income', 'balance']
            
            try:
                cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['city', 'has_credit_card'])
                feature_names.extend(cat_features)
            except:
                feature_names.extend(['city_A', 'city_B', 'city_C', 'has_credit_card_0', 'has_credit_card_1'])
            
            return feature_names
            
        except Exception as e:
            logger.error(f"Error getting feature names: {e}")
            return self.feature_names

# Global interpreter instance
interpreter = None

def get_interpreter(model_path: str = "./models/gbm_pipeline.pkl") -> ModelInterpreter:
    """Get or create model interpreter instance"""
    global interpreter
    if interpreter is None:
        interpreter = ModelInterpreter(model_path)
    return interpreter