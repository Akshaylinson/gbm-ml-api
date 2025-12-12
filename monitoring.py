import time
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Any
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import logging
from scipy import stats

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('ml_api_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('ml_api_request_duration_seconds', 'Request duration')
PREDICTION_COUNT = Counter('ml_api_predictions_total', 'Total predictions', ['model_version', 'prediction'])
MODEL_DRIFT = Gauge('ml_api_model_drift', 'Model drift score', ['feature'])
ERROR_RATE = Gauge('ml_api_error_rate', 'Error rate')

class AdvancedMonitor:
    def __init__(self):
        self.predictions = []
        self.response_times = []
        self.error_count = 0
        self.request_count = 0
        self.feature_stats = defaultdict(list)
        self.baseline_stats = {}
        self.drift_threshold = 2.0  # Standard deviations
        
    def log_prediction(self, features: dict, prediction: int, probability: float, 
                      response_time: float, model_version: str = "v1"):
        """Log prediction for monitoring"""
        timestamp = datetime.now()
        
        # Store prediction data
        pred_data = {
            "prediction": prediction,
            "probability": probability,
            "timestamp": timestamp,
            "response_time": response_time,
            "model_version": model_version
        }
        self.predictions.append(pred_data)
        
        # Update metrics
        self.response_times.append(response_time)
        self.request_count += 1
        
        # Update feature statistics
        for key, value in features.items():
            if isinstance(value, (int, float)):
                self.feature_stats[key].append(value)
        
        # Update Prometheus metrics
        PREDICTION_COUNT.labels(model_version=model_version, prediction=str(prediction)).inc()
        REQUEST_DURATION.observe(response_time)
        
        # Keep only recent data (last 24 hours)
        cutoff = timestamp - timedelta(hours=24)
        self.predictions = [p for p in self.predictions if p["timestamp"] > cutoff]
        
        # Limit feature stats size
        for key in self.feature_stats:
            if len(self.feature_stats[key]) > 10000:
                self.feature_stats[key] = self.feature_stats[key][-5000:]
    
    def log_error(self, error_type: str = "general"):
        """Log error occurrence"""
        self.error_count += 1
        ERROR_RATE.set(self.error_count / max(self.request_count, 1))
    
    def set_baseline(self, baseline_data: Dict[str, List[float]]):
        """Set baseline statistics for drift detection"""
        self.baseline_stats = {}
        for feature, values in baseline_data.items():
            if values:
                self.baseline_stats[feature] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
    
    def detect_drift(self) -> Dict[str, float]:
        """Detect feature drift using statistical tests"""
        drift_scores = {}
        
        for feature, current_values in self.feature_stats.items():
            if feature not in self.baseline_stats or len(current_values) < 30:
                continue
                
            baseline = self.baseline_stats[feature]
            recent_values = current_values[-100:]  # Last 100 values
            
            if len(recent_values) < 10:
                continue
            
            # Statistical tests for drift
            current_mean = np.mean(recent_values)
            current_std = np.std(recent_values)
            
            # Z-score for mean shift
            mean_drift = abs(current_mean - baseline["mean"]) / (baseline["std"] + 1e-8)
            
            # Kolmogorov-Smirnov test for distribution shift
            try:
                # Generate baseline sample for comparison
                baseline_sample = np.random.normal(
                    baseline["mean"], baseline["std"], len(recent_values)
                )
                ks_stat, p_value = stats.ks_2samp(recent_values, baseline_sample)
                distribution_drift = ks_stat
            except:
                distribution_drift = 0
            
            # Combined drift score
            drift_score = max(mean_drift, distribution_drift * 10)
            drift_scores[feature] = drift_score
            
            # Update Prometheus metric
            MODEL_DRIFT.labels(feature=feature).set(drift_score)
        
        return drift_scores
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.predictions:
            return {"status": "no_data"}
        
        # Recent predictions (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_preds = [p for p in self.predictions if p["timestamp"] > recent_cutoff]
        
        # Calculate metrics
        metrics = {
            "total_requests": self.request_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "avg_response_time_ms": np.mean(self.response_times[-1000:]) * 1000 if self.response_times else 0,
            "p95_response_time_ms": np.percentile(self.response_times[-1000:], 95) * 1000 if len(self.response_times) > 10 else 0,
            "requests_per_hour": len(recent_preds),
            "prediction_distribution": self._get_prediction_distribution(recent_preds),
            "probability_stats": self._get_probability_stats(recent_preds),
            "drift_scores": self.detect_drift(),
            "model_health": self._assess_model_health()
        }
        
        return metrics
    
    def _get_prediction_distribution(self, predictions: List[Dict]) -> Dict[str, int]:
        """Get distribution of predictions"""
        dist = {"positive": 0, "negative": 0}
        for pred in predictions:
            if pred["prediction"] == 1:
                dist["positive"] += 1
            else:
                dist["negative"] += 1
        return dist
    
    def _get_probability_stats(self, predictions: List[Dict]) -> Dict[str, float]:
        """Get probability statistics"""
        if not predictions:
            return {}
        
        probs = [p["probability"] for p in predictions]
        return {
            "mean": float(np.mean(probs)),
            "std": float(np.std(probs)),
            "min": float(np.min(probs)),
            "max": float(np.max(probs)),
            "median": float(np.median(probs))
        }
    
    def _assess_model_health(self) -> Dict[str, Any]:
        """Assess overall model health"""
        health_score = 100.0
        issues = []
        
        # Check error rate
        error_rate = self.error_count / max(self.request_count, 1)
        if error_rate > 0.05:  # 5% error rate threshold
            health_score -= 30
            issues.append(f"High error rate: {error_rate:.2%}")
        
        # Check response time
        if self.response_times:
            avg_response = np.mean(self.response_times[-100:])
            if avg_response > 1.0:  # 1 second threshold
                health_score -= 20
                issues.append(f"Slow response time: {avg_response:.2f}s")
        
        # Check drift
        drift_scores = self.detect_drift()
        high_drift_features = [f for f, score in drift_scores.items() if score > self.drift_threshold]
        if high_drift_features:
            health_score -= len(high_drift_features) * 10
            issues.append(f"High drift in features: {high_drift_features}")
        
        # Check prediction distribution
        recent_preds = [p for p in self.predictions if p["timestamp"] > datetime.now() - timedelta(hours=1)]
        if recent_preds:
            pos_rate = sum(1 for p in recent_preds if p["prediction"] == 1) / len(recent_preds)
            if pos_rate > 0.9 or pos_rate < 0.1:  # Extreme prediction bias
                health_score -= 15
                issues.append(f"Extreme prediction bias: {pos_rate:.2%} positive")
        
        return {
            "score": max(0, health_score),
            "status": "healthy" if health_score > 80 else "warning" if health_score > 50 else "critical",
            "issues": issues
        }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics"""
        return generate_latest()

# Global monitor instance
monitor = AdvancedMonitor()