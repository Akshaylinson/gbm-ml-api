# Advanced Real-time Monitoring System
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict
from scipy import stats
import json

logger = logging.getLogger(__name__)

@dataclass
class PredictionMetrics:
    """Metrics for a single prediction"""
    timestamp: datetime
    user_id: str
    model_version: str
    prediction: int
    probability: float
    confidence: float
    response_time_ms: float
    input_features: Dict[str, Any]
    risk_score: str

@dataclass
class ModelPerformanceMetrics:
    """Aggregated model performance metrics"""
    total_predictions: int
    avg_probability: float
    prediction_distribution: Dict[str, int]
    avg_response_time_ms: float
    error_rate: float
    confidence_distribution: Dict[str, float]
    feature_drift_scores: Dict[str, float]
    performance_trend: str

class AdvancedModelMonitor:
    """Advanced monitoring system with drift detection and performance tracking"""
    
    def __init__(self, window_size: int = 1000, drift_threshold: float = 2.0):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # Storage for metrics
        self.predictions: deque = deque(maxlen=window_size)
        self.feature_baselines: Dict[str, Dict[str, float]] = {}
        self.performance_history: deque = deque(maxlen=100)  # Last 100 time windows
        
        # Real-time counters
        self.total_requests = 0
        self.error_count = 0
        self.response_times = deque(maxlen=1000)
        
        # Feature statistics
        self.feature_stats = defaultdict(lambda: deque(maxlen=window_size))
        
        # Alerts
        self.alerts = deque(maxlen=50)
        
        # Initialize baseline
        self._initialize_baselines()
    
    def _initialize_baselines(self):
        """Initialize feature baselines from historical data or defaults"""
        # Default baselines (would typically come from training data)
        self.feature_baselines = {
            'age': {'mean': 45.0, 'std': 15.0, 'min': 18, 'max': 100},
            'income': {'mean': 50000.0, 'std': 25000.0, 'min': -50000, 'max': 500000},
            'balance': {'mean': 2000.0, 'std': 3000.0, 'min': -10000, 'max': 100000},
            'city': {'A': 0.33, 'B': 0.33, 'C': 0.34},
            'has_credit_card': {'0': 0.4, '1': 0.6}
        }
    
    def log_prediction(self, metrics: PredictionMetrics):
        """Log a new prediction with comprehensive metrics"""
        try:
            self.predictions.append(metrics)
            self.total_requests += 1
            self.response_times.append(metrics.response_time_ms)
            
            # Update feature statistics
            for feature, value in metrics.input_features.items():
                if isinstance(value, (int, float)):
                    self.feature_stats[feature].append(value)
                else:
                    # For categorical features, track distribution
                    self.feature_stats[f"{feature}_{value}"].append(1)
            
            # Check for drift and anomalies
            self._check_drift(metrics)
            self._check_anomalies(metrics)
            
            # Update performance history every 100 predictions
            if len(self.predictions) % 100 == 0:
                self._update_performance_history()
            
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
            self.log_error()
    
    def log_error(self):
        """Log an error occurrence"""
        self.error_count += 1
    
    def _check_drift(self, metrics: PredictionMetrics):
        """Check for feature drift using statistical tests"""
        try:
            if len(self.predictions) < 100:  # Need sufficient data
                return
            
            recent_window = 50
            baseline_window = 100
            
            if len(self.predictions) < baseline_window + recent_window:
                return
            
            # Get recent and baseline data
            recent_predictions = list(self.predictions)[-recent_window:]
            baseline_predictions = list(self.predictions)[-(baseline_window + recent_window):-recent_window]
            
            drift_detected = False
            
            for feature in ['age', 'income', 'balance']:
                if feature in metrics.input_features:
                    recent_values = [p.input_features.get(feature, 0) for p in recent_predictions if feature in p.input_features]
                    baseline_values = [p.input_features.get(feature, 0) for p in baseline_predictions if feature in p.input_features]
                    
                    if len(recent_values) > 10 and len(baseline_values) > 10:
                        # Kolmogorov-Smirnov test for distribution drift
                        ks_stat, p_value = stats.ks_2samp(recent_values, baseline_values)
                        
                        if p_value < 0.05:  # Significant drift detected
                            drift_score = ks_stat * 10  # Scale for interpretability
                            
                            if drift_score > self.drift_threshold:
                                self._create_alert(
                                    "feature_drift",
                                    f"Significant drift detected in {feature}",
                                    {"feature": feature, "drift_score": drift_score, "p_value": p_value}
                                )
                                drift_detected = True
            
            # Check categorical drift
            for feature in ['city', 'has_credit_card']:
                if feature in metrics.input_features:
                    recent_dist = self._get_categorical_distribution(recent_predictions, feature)
                    baseline_dist = self._get_categorical_distribution(baseline_predictions, feature)
                    
                    # Chi-square test for categorical drift
                    if recent_dist and baseline_dist:
                        chi2_stat = self._calculate_chi2_drift(recent_dist, baseline_dist)
                        if chi2_stat > self.drift_threshold:
                            self._create_alert(
                                "categorical_drift",
                                f"Categorical drift detected in {feature}",
                                {"feature": feature, "chi2_stat": chi2_stat}
                            )
                            drift_detected = True
            
        except Exception as e:
            logger.error(f"Error checking drift: {e}")
    
    def _check_anomalies(self, metrics: PredictionMetrics):
        """Check for prediction anomalies"""
        try:
            # Check for unusual probability values
            if metrics.probability > 0.95 or metrics.probability < 0.05:
                self._create_alert(
                    "extreme_probability",
                    f"Extreme probability detected: {metrics.probability:.3f}",
                    {"probability": metrics.probability, "user": metrics.user_id}
                )
            
            # Check for unusual response times
            if len(self.response_times) > 10:
                avg_response_time = np.mean(list(self.response_times)[-10:])
                if metrics.response_time_ms > avg_response_time * 3:
                    self._create_alert(
                        "slow_response",
                        f"Slow response detected: {metrics.response_time_ms:.2f}ms",
                        {"response_time": metrics.response_time_ms, "avg_time": avg_response_time}
                    )
            
            # Check for unusual feature combinations
            self._check_feature_anomalies(metrics.input_features)
            
        except Exception as e:
            logger.error(f"Error checking anomalies: {e}")
    
    def _check_feature_anomalies(self, features: Dict[str, Any]):
        """Check for anomalous feature combinations"""
        try:
            # High income with negative balance
            if features.get('income', 0) > 100000 and features.get('balance', 0) < -5000:
                self._create_alert(
                    "feature_anomaly",
                    "High income with large negative balance",
                    {"income": features.get('income'), "balance": features.get('balance')}
                )
            
            # Very young with very high income
            if features.get('age', 0) < 25 and features.get('income', 0) > 200000:
                self._create_alert(
                    "feature_anomaly",
                    "Very young age with very high income",
                    {"age": features.get('age'), "income": features.get('income')}
                )
            
        except Exception as e:
            logger.error(f"Error checking feature anomalies: {e}")
    
    def _get_categorical_distribution(self, predictions: List[PredictionMetrics], feature: str) -> Dict[str, int]:
        """Get distribution of categorical feature"""
        distribution = defaultdict(int)
        for pred in predictions:
            if feature in pred.input_features:
                value = str(pred.input_features[feature])
                distribution[value] += 1
        return dict(distribution)
    
    def _calculate_chi2_drift(self, recent_dist: Dict[str, int], baseline_dist: Dict[str, int]) -> float:
        """Calculate chi-square statistic for categorical drift"""
        try:
            all_categories = set(recent_dist.keys()) | set(baseline_dist.keys())
            
            recent_counts = [recent_dist.get(cat, 0) for cat in all_categories]
            baseline_counts = [baseline_dist.get(cat, 0) for cat in all_categories]
            
            # Avoid division by zero
            baseline_counts = [max(1, count) for count in baseline_counts]
            
            chi2_stat = sum((r - b) ** 2 / b for r, b in zip(recent_counts, baseline_counts))
            return chi2_stat
            
        except Exception as e:
            logger.error(f"Error calculating chi2 drift: {e}")
            return 0.0
    
    def _create_alert(self, alert_type: str, message: str, details: Dict[str, Any]):
        """Create a new alert"""
        alert = {
            "timestamp": datetime.now(),
            "type": alert_type,
            "message": message,
            "details": details,
            "severity": self._get_alert_severity(alert_type)
        }
        self.alerts.append(alert)
        logger.warning(f"Alert created: {alert_type} - {message}")
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Determine alert severity"""
        severity_map = {
            "feature_drift": "medium",
            "categorical_drift": "medium",
            "extreme_probability": "high",
            "slow_response": "low",
            "feature_anomaly": "medium",
            "performance_degradation": "high"
        }
        return severity_map.get(alert_type, "low")
    
    def _update_performance_history(self):
        """Update performance history for trend analysis"""
        try:
            if len(self.predictions) < 50:
                return
            
            recent_predictions = list(self.predictions)[-50:]
            
            # Calculate metrics for this window
            probabilities = [p.probability for p in recent_predictions]
            response_times = [p.response_time_ms for p in recent_predictions]
            confidences = [p.confidence for p in recent_predictions]
            
            window_metrics = {
                "timestamp": datetime.now(),
                "avg_probability": np.mean(probabilities),
                "avg_response_time": np.mean(response_times),
                "avg_confidence": np.mean(confidences),
                "prediction_count": len(recent_predictions)
            }
            
            self.performance_history.append(window_metrics)
            
            # Check for performance degradation
            if len(self.performance_history) >= 5:
                self._check_performance_trend()
            
        except Exception as e:
            logger.error(f"Error updating performance history: {e}")
    
    def _check_performance_trend(self):
        """Check for performance degradation trends"""
        try:
            recent_windows = list(self.performance_history)[-5:]
            
            # Check response time trend
            response_times = [w["avg_response_time"] for w in recent_windows]
            if len(response_times) >= 3:
                # Simple trend detection
                if all(response_times[i] < response_times[i+1] for i in range(len(response_times)-1)):
                    if response_times[-1] > response_times[0] * 1.5:
                        self._create_alert(
                            "performance_degradation",
                            "Response time degradation detected",
                            {"trend": response_times}
                        )
            
            # Check confidence trend
            confidences = [w["avg_confidence"] for w in recent_windows]
            if len(confidences) >= 3:
                if all(confidences[i] > confidences[i+1] for i in range(len(confidences)-1)):
                    if confidences[0] - confidences[-1] > 0.1:
                        self._create_alert(
                            "performance_degradation",
                            "Model confidence degradation detected",
                            {"trend": confidences}
                        )
            
        except Exception as e:
            logger.error(f"Error checking performance trend: {e}")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring metrics"""
        try:
            if not self.predictions:
                return {"status": "no_data"}
            
            recent_predictions = list(self.predictions)[-100:] if len(self.predictions) >= 100 else list(self.predictions)
            
            # Basic metrics
            probabilities = [p.probability for p in recent_predictions]
            response_times = [p.response_time_ms for p in recent_predictions]
            confidences = [p.confidence for p in recent_predictions]
            predictions = [p.prediction for p in recent_predictions]
            
            # Feature drift scores
            drift_scores = self._calculate_current_drift_scores()
            
            # Performance trend
            trend = self._get_performance_trend()
            
            return {
                "total_requests": self.total_requests,
                "error_rate": self.error_count / max(self.total_requests, 1),
                "avg_response_time_ms": np.mean(response_times) if response_times else 0,
                "avg_probability": np.mean(probabilities) if probabilities else 0,
                "avg_confidence": np.mean(confidences) if confidences else 0,
                "prediction_distribution": {
                    "positive": sum(predictions),
                    "negative": len(predictions) - sum(predictions)
                },
                "confidence_distribution": {
                    "high": sum(1 for c in confidences if c > 0.8),
                    "medium": sum(1 for c in confidences if 0.5 <= c <= 0.8),
                    "low": sum(1 for c in confidences if c < 0.5)
                },
                "feature_drift_scores": drift_scores,
                "performance_trend": trend,
                "active_alerts": len([a for a in self.alerts if a["timestamp"] > datetime.now() - timedelta(hours=1)]),
                "recent_alerts": [
                    {
                        "type": a["type"],
                        "message": a["message"],
                        "timestamp": a["timestamp"].isoformat(),
                        "severity": a["severity"]
                    }
                    for a in list(self.alerts)[-5:]
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_current_drift_scores(self) -> Dict[str, float]:
        """Calculate current drift scores for all features"""
        drift_scores = {}
        
        try:
            if len(self.predictions) < 100:
                return drift_scores
            
            recent_window = 50
            baseline_window = 100
            
            if len(self.predictions) < baseline_window + recent_window:
                return drift_scores
            
            recent_predictions = list(self.predictions)[-recent_window:]
            baseline_predictions = list(self.predictions)[-(baseline_window + recent_window):-recent_window]
            
            for feature in ['age', 'income', 'balance']:
                recent_values = [p.input_features.get(feature, 0) for p in recent_predictions if feature in p.input_features]
                baseline_values = [p.input_features.get(feature, 0) for p in baseline_predictions if feature in p.input_features]
                
                if len(recent_values) > 10 and len(baseline_values) > 10:
                    ks_stat, _ = stats.ks_2samp(recent_values, baseline_values)
                    drift_scores[feature] = ks_stat * 10
            
            return drift_scores
            
        except Exception as e:
            logger.error(f"Error calculating drift scores: {e}")
            return {}
    
    def _get_performance_trend(self) -> str:
        """Get overall performance trend"""
        try:
            if len(self.performance_history) < 3:
                return "insufficient_data"
            
            recent_windows = list(self.performance_history)[-3:]
            response_times = [w["avg_response_time"] for w in recent_windows]
            confidences = [w["avg_confidence"] for w in recent_windows]
            
            # Simple trend analysis
            rt_trend = "stable"
            if response_times[-1] > response_times[0] * 1.2:
                rt_trend = "degrading"
            elif response_times[-1] < response_times[0] * 0.8:
                rt_trend = "improving"
            
            conf_trend = "stable"
            if confidences[-1] < confidences[0] - 0.05:
                conf_trend = "degrading"
            elif confidences[-1] > confidences[0] + 0.05:
                conf_trend = "improving"
            
            if rt_trend == "degrading" or conf_trend == "degrading":
                return "degrading"
            elif rt_trend == "improving" and conf_trend == "improving":
                return "improving"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error getting performance trend: {e}")
            return "unknown"
    
    def get_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            {
                "type": a["type"],
                "message": a["message"],
                "timestamp": a["timestamp"].isoformat(),
                "severity": a["severity"],
                "details": a["details"]
            }
            for a in self.alerts
            if a["timestamp"] > cutoff_time
        ]

# Global monitor instance
advanced_monitor = AdvancedModelMonitor()