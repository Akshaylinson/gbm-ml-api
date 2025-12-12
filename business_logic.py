import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime
import logging
from config import settings

logger = logging.getLogger(__name__)

class BusinessRuleEngine:
    def __init__(self):
        self.rules = {
            "age_risk": self._age_risk_rule,
            "income_balance_ratio": self._income_balance_rule,
            "high_risk_profile": self._high_risk_profile_rule,
            "credit_card_factor": self._credit_card_rule,
            "city_risk": self._city_risk_rule
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            "low": 0.3,
            "medium": 0.7,
            "high": 0.9
        }
        
        # City risk scores (example business logic)
        self.city_risk_scores = {
            "A": 0.1,  # Low risk
            "B": 0.2,  # Medium risk  
            "C": 0.3   # Higher risk
        }
    
    def apply_business_rules(self, features: Dict[str, Any], 
                           ml_probability: float) -> Dict[str, Any]:
        """Apply business rules to ML prediction"""
        
        # Calculate individual rule scores
        rule_scores = {}
        for rule_name, rule_func in self.rules.items():
            try:
                rule_scores[rule_name] = rule_func(features)
            except Exception as e:
                logger.warning(f"Rule {rule_name} failed: {e}")
                rule_scores[rule_name] = 0.0
        
        # Calculate business risk score
        business_risk = self._calculate_business_risk(rule_scores)
        
        # Combine ML probability with business rules
        adjusted_probability = self._combine_scores(ml_probability, business_risk, rule_scores)
        
        # Determine final risk category
        risk_category = self._categorize_risk(adjusted_probability)
        
        # Generate explanation
        explanation = self._generate_explanation(features, rule_scores, ml_probability, adjusted_probability)
        
        return {
            "original_ml_probability": ml_probability,
            "business_risk_score": business_risk,
            "adjusted_probability": adjusted_probability,
            "risk_category": risk_category,
            "rule_scores": rule_scores,
            "explanation": explanation,
            "confidence_level": self._calculate_confidence(rule_scores, ml_probability)
        }
    
    def _age_risk_rule(self, features: Dict[str, Any]) -> float:
        """Age-based risk assessment"""
        age = features.get("age", 0)
        
        if age < 25:
            return 0.3  # Higher risk for very young
        elif age > 70:
            return 0.2  # Moderate risk for elderly
        else:
            return 0.0  # Lower risk for middle age
    
    def _income_balance_rule(self, features: Dict[str, Any]) -> float:
        """Income to balance ratio risk"""
        income = features.get("income", 0)
        balance = features.get("balance", 0)
        
        if income <= 0:
            return 0.5  # High risk for no income
        
        ratio = balance / income if income > 0 else 0
        
        if ratio < -0.1:  # Negative balance relative to income
            return 0.4
        elif ratio > 0.5:  # Very high balance relative to income
            return -0.2  # Actually reduces risk
        else:
            return 0.0
    
    def _high_risk_profile_rule(self, features: Dict[str, Any]) -> float:
        """Identify high-risk profiles"""
        age = features.get("age", 0)
        income = features.get("income", 0)
        balance = features.get("balance", 0)
        
        # Young with high income but negative balance
        if age < 30 and income > 80000 and balance < 0:
            return 0.3
        
        # Elderly with very low income
        if age > 65 and income < 20000:
            return 0.2
        
        return 0.0
    
    def _credit_card_rule(self, features: Dict[str, Any]) -> float:
        """Credit card ownership factor"""
        has_card = features.get("has_credit_card", 0)
        income = features.get("income", 0)
        
        # High income without credit card might be suspicious
        if income > 100000 and has_card == 0:
            return 0.1
        
        # Low income with credit card
        if income < 30000 and has_card == 1:
            return 0.05
        
        return 0.0
    
    def _city_risk_rule(self, features: Dict[str, Any]) -> float:
        """City-based risk assessment"""
        city = features.get("city", "A")
        return self.city_risk_scores.get(city, 0.1)
    
    def _calculate_business_risk(self, rule_scores: Dict[str, float]) -> float:
        """Calculate overall business risk score"""
        # Weighted combination of rule scores
        weights = {
            "age_risk": 0.2,
            "income_balance_ratio": 0.3,
            "high_risk_profile": 0.3,
            "credit_card_factor": 0.1,
            "city_risk": 0.1
        }
        
        total_score = sum(rule_scores.get(rule, 0) * weight 
                         for rule, weight in weights.items())
        
        # Normalize to 0-1 range
        return max(0, min(1, total_score))
    
    def _combine_scores(self, ml_prob: float, business_risk: float, 
                       rule_scores: Dict[str, float]) -> float:
        """Combine ML probability with business risk"""
        
        # Base combination (weighted average)
        ml_weight = 0.7
        business_weight = 0.3
        
        combined = (ml_prob * ml_weight) + (business_risk * business_weight)
        
        # Apply hard rules (overrides)
        if rule_scores.get("high_risk_profile", 0) > 0.25:
            combined = max(combined, 0.8)  # Force high risk
        
        # Ensure bounds
        return max(0, min(1, combined))
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk level"""
        if probability < self.risk_thresholds["low"]:
            return "LOW"
        elif probability < self.risk_thresholds["medium"]:
            return "MEDIUM"
        elif probability < self.risk_thresholds["high"]:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _calculate_confidence(self, rule_scores: Dict[str, float], 
                            ml_prob: float) -> str:
        """Calculate confidence level in the prediction"""
        
        # Check consistency between rules and ML
        rule_avg = np.mean(list(rule_scores.values()))
        consistency = 1 - abs(ml_prob - rule_avg)
        
        if consistency > 0.8:
            return "HIGH"
        elif consistency > 0.6:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_explanation(self, features: Dict[str, Any], 
                            rule_scores: Dict[str, float],
                            ml_prob: float, adjusted_prob: float) -> str:
        """Generate human-readable explanation"""
        
        explanations = []
        
        # ML component
        explanations.append(f"ML model predicts {ml_prob:.1%} probability")
        
        # Significant business rules
        for rule, score in rule_scores.items():
            if abs(score) > 0.1:
                direction = "increases" if score > 0 else "decreases"
                explanations.append(f"{rule.replace('_', ' ')} {direction} risk by {abs(score):.1%}")
        
        # Final adjustment
        if abs(adjusted_prob - ml_prob) > 0.05:
            direction = "increased" if adjusted_prob > ml_prob else "decreased"
            explanations.append(f"Business rules {direction} final probability to {adjusted_prob:.1%}")
        
        return "; ".join(explanations)

class RiskScorer:
    def __init__(self):
        self.business_engine = BusinessRuleEngine()
    
    def calculate_comprehensive_risk(self, features: Dict[str, Any], 
                                   ml_probability: float,
                                   confidence_interval: List[float] = None) -> Dict[str, Any]:
        """Calculate comprehensive risk assessment"""
        
        # Apply business rules
        business_result = self.business_engine.apply_business_rules(features, ml_probability)
        
        # Calculate uncertainty metrics
        uncertainty_metrics = self._calculate_uncertainty(
            ml_probability, confidence_interval, business_result["rule_scores"]
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            features, business_result, uncertainty_metrics
        )
        
        return {
            **business_result,
            "uncertainty_metrics": uncertainty_metrics,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_uncertainty(self, ml_prob: float, 
                             confidence_interval: List[float] = None,
                             rule_scores: Dict[str, float] = None) -> Dict[str, Any]:
        """Calculate prediction uncertainty"""
        
        uncertainty = {}
        
        # Confidence interval width
        if confidence_interval:
            ci_width = confidence_interval[1] - confidence_interval[0]
            uncertainty["confidence_interval_width"] = ci_width
            uncertainty["uncertainty_level"] = "high" if ci_width > 0.3 else "medium" if ci_width > 0.15 else "low"
        
        # Rule consistency
        if rule_scores:
            rule_variance = np.var(list(rule_scores.values()))
            uncertainty["rule_consistency"] = "high" if rule_variance < 0.01 else "medium" if rule_variance < 0.05 else "low"
        
        # Probability extremeness (closer to 0.5 = more uncertain)
        prob_uncertainty = 1 - 2 * abs(ml_prob - 0.5)
        uncertainty["probability_uncertainty"] = prob_uncertainty
        
        return uncertainty
    
    def _generate_recommendations(self, features: Dict[str, Any],
                                business_result: Dict[str, Any],
                                uncertainty_metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        risk_category = business_result["risk_category"]
        confidence = business_result["confidence_level"]
        
        # Risk-based recommendations
        if risk_category == "CRITICAL":
            recommendations.append("REJECT: Critical risk level detected")
            recommendations.append("Manual review required before any approval")
        elif risk_category == "HIGH":
            recommendations.append("CAUTION: High risk - require additional verification")
            recommendations.append("Consider reduced limits or additional collateral")
        elif risk_category == "MEDIUM":
            recommendations.append("REVIEW: Medium risk - standard verification process")
        else:
            recommendations.append("APPROVE: Low risk profile")
        
        # Confidence-based recommendations
        if confidence == "LOW":
            recommendations.append("Low confidence prediction - consider manual review")
        
        # Uncertainty-based recommendations
        uncertainty_level = uncertainty_metrics.get("uncertainty_level", "medium")
        if uncertainty_level == "high":
            recommendations.append("High uncertainty detected - gather additional data")
        
        # Feature-specific recommendations
        age = features.get("age", 0)
        income = features.get("income", 0)
        balance = features.get("balance", 0)
        
        if age < 25:
            recommendations.append("Young applicant - verify income stability")
        
        if income > 100000 and balance < 1000:
            recommendations.append("High income with low balance - verify income source")
        
        if balance < -5000:
            recommendations.append("Significant negative balance - assess debt situation")
        
        return recommendations

# Global instances
business_engine = BusinessRuleEngine()
risk_scorer = RiskScorer()