from celery import Celery
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
from config import settings
from model_manager import model_manager
from database import db_manager
from monitoring import monitor
import json

# Configure Celery
celery_app = Celery(
    "ml_api",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["celery_app"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "celery_app.process_batch_predictions": {"queue": "predictions"},
        "celery_app.retrain_model": {"queue": "training"},
        "celery_app.calculate_model_metrics": {"queue": "metrics"}
    }
)

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, max_retries=3)
def process_batch_predictions(self, batch_data: List[Dict], model_version: str = "v1", 
                            user_id: str = "system") -> Dict[str, Any]:
    """Process batch predictions asynchronously"""
    try:
        # Load model if not already loaded
        if model_version not in model_manager.models:
            model_manager.load_model(model_version)
        
        model = model_manager.get_model(model_version)
        if not model:
            raise ValueError(f"Model {model_version} not available")
        
        # Convert to DataFrame
        df = pd.DataFrame(batch_data)
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                "row_index": i,
                "prediction": int(pred),
                "probability": float(prob),
                "input_data": batch_data[i]
            }
            results.append(result)
            
            # Log to database
            log_data = {
                "request_id": f"batch_{self.request.id}_{i}",
                "user_id": user_id,
                "model_version": model_version,
                "input_data": batch_data[i],
                "prediction": int(pred),
                "probability": float(prob),
                "processing_time_ms": 0  # Will be calculated at the end
            }
            db_manager.log_prediction(log_data)
        
        return {
            "status": "completed",
            "results": results,
            "total_processed": len(results),
            "model_version": model_version
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))

@celery_app.task(bind=True)
def retrain_model(self, training_data_path: str, model_version: str, 
                 hyperparameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Retrain model asynchronously"""
    try:
        from train import train_model  # Import training function
        
        # Train new model
        model_path, metrics = train_model(
            data_path=training_data_path,
            output_path=f"models/gbm_pipeline_{model_version}.pkl",
            hyperparameters=hyperparameters or {}
        )
        
        # Load the new model
        model_manager.load_model(model_version, model_path)
        
        return {
            "status": "completed",
            "model_version": model_version,
            "model_path": model_path,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

@celery_app.task
def calculate_model_metrics(model_version: str = "v1", hours: int = 24) -> Dict[str, Any]:
    """Calculate model performance metrics asynchronously"""
    try:
        # Get recent predictions from database
        metrics_data = db_manager.get_model_metrics(model_version, hours)
        
        if not metrics_data:
            return {"status": "no_data"}
        
        # Calculate performance metrics
        predictions = [m.prediction for m in metrics_data if hasattr(m, 'prediction')]
        probabilities = [m.probability for m in metrics_data if hasattr(m, 'probability')]
        
        if not predictions:
            return {"status": "no_predictions"}
        
        metrics = {
            "model_version": model_version,
            "time_period_hours": hours,
            "total_predictions": len(predictions),
            "positive_rate": sum(predictions) / len(predictions),
            "avg_probability": np.mean(probabilities),
            "probability_std": np.std(probabilities),
            "probability_distribution": {
                "min": float(np.min(probabilities)),
                "max": float(np.max(probabilities)),
                "median": float(np.median(probabilities)),
                "q25": float(np.percentile(probabilities, 25)),
                "q75": float(np.percentile(probabilities, 75))
            }
        }
        
        return {
            "status": "completed",
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

@celery_app.task
def cleanup_old_data(days: int = 30) -> Dict[str, Any]:
    """Clean up old prediction logs"""
    try:
        from datetime import datetime, timedelta
        from database import PredictionLog, SessionLocal
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        db = SessionLocal()
        deleted_count = db.query(PredictionLog).filter(
            PredictionLog.timestamp < cutoff_date
        ).delete()
        db.commit()
        db.close()
        
        return {
            "status": "completed",
            "deleted_records": deleted_count,
            "cutoff_date": cutoff_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data cleanup failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

@celery_app.task
def generate_model_report(model_version: str = "v1") -> Dict[str, Any]:
    """Generate comprehensive model performance report"""
    try:
        # Get model metrics
        performance_metrics = monitor.get_performance_metrics()
        
        # Get feature importance
        feature_importance = model_manager.get_feature_importance(model_version)
        
        # Get recent predictions for analysis
        recent_predictions = db_manager.get_user_history("system", limit=1000)
        
        report = {
            "model_version": model_version,
            "generated_at": pd.Timestamp.now().isoformat(),
            "performance_metrics": performance_metrics,
            "feature_importance": feature_importance,
            "data_summary": {
                "total_predictions": len(recent_predictions),
                "time_range": "last_1000_predictions"
            }
        }
        
        return {
            "status": "completed",
            "report": report
        }
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

# Periodic tasks
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    "calculate-hourly-metrics": {
        "task": "celery_app.calculate_model_metrics",
        "schedule": crontab(minute=0),  # Every hour
    },
    "cleanup-old-data": {
        "task": "celery_app.cleanup_old_data",
        "schedule": crontab(hour=2, minute=0),  # Daily at 2 AM
        "kwargs": {"days": 30}
    },
    "generate-daily-report": {
        "task": "celery_app.generate_model_report",
        "schedule": crontab(hour=6, minute=0),  # Daily at 6 AM
    }
}

if __name__ == "__main__":
    celery_app.start()