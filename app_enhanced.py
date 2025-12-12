# Enhanced app.py with all features
import os
import time
import hashlib
import logging
import json
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from pathlib import Path
from collections import defaultdict, deque
from typing import List, Optional, Dict, Any
import asyncio

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
import uvicorn

from database import get_db, PredictionLog
from cache import cache
from config import settings
from ab_testing import ab_manager
from model_registry import registry

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Rate Limiting
rate_limiter = defaultdict(lambda: deque())

# Monitoring
class ModelMonitor:
    def __init__(self):
        self.predictions = []
        self.response_times = []
        self.error_count = 0
        self.request_count = 0
        self.feature_stats = defaultdict(list)
    
    def log_prediction(self, features: dict, prediction: int, probability: float, response_time: float):
        self.predictions.append({"pred": prediction, "prob": probability, "timestamp": datetime.now()})
        self.response_times.append(response_time)
        self.request_count += 1
        for key, value in features.items():
            if isinstance(value, (int, float)):
                self.feature_stats[key].append(value)
    
    def log_error(self):
        self.error_count += 1
    
    def get_metrics(self):
        if not self.predictions:
            return {"status": "no_data"}
        
        recent_preds = [p for p in self.predictions if p["timestamp"] > datetime.now() - timedelta(hours=1)]
        probs = [p["prob"] for p in recent_preds]
        
        return {
            "total_requests": self.request_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "avg_response_time_ms": np.mean(self.response_times) * 1000 if self.response_times else 0,
            "avg_probability": np.mean(probs) if probs else 0,
            "prediction_distribution": {
                "positive": sum(1 for p in recent_preds if p["pred"] == 1),
                "negative": sum(1 for p in recent_preds if p["pred"] == 0)
            },
            "feature_drift": self._detect_drift()
        }
    
    def _detect_drift(self):
        drift_scores = {}
        for feature, values in self.feature_stats.items():
            if len(values) > 100:
                recent = values[-50:]
                baseline = values[-100:-50]
                drift_scores[feature] = abs(np.mean(recent) - np.mean(baseline)) / (np.std(baseline) + 1e-8)
        return drift_scores

monitor = ModelMonitor()

# Input Validation
class SingleRow(BaseModel):
    age: float = Field(..., ge=18, le=100, description="Age between 18-100")
    income: float = Field(..., ge=-50000, le=500000, description="Income between -50k to 500k")
    balance: float = Field(..., ge=-10000, le=100000, description="Balance between -10k to 100k")
    city: str = Field(..., regex="^[ABC]$", description="City must be A, B, or C")
    has_credit_card: int = Field(..., ge=0, le=1, description="0 or 1")
    
    @validator('*', pre=True)
    def sanitize_input(cls, v):
        if isinstance(v, str):
            return v.strip()[:50]
        return v

class BatchRequest(BaseModel):
    rows: List[SingleRow] = Field(..., max_items=settings.max_batch_size)
    model_version: Optional[str] = Field("v1")
    explain: Optional[bool] = Field(False)

class Prediction(BaseModel):
    prediction: int
    probability: float
    confidence_interval: Optional[List[float]] = None
    risk_score: Optional[str] = None
    explanation: Optional[Dict[str, float]] = None

class BatchResponse(BaseModel):
    results: List[Prediction]
    model_version: str
    processing_time_ms: float
    request_id: str

# Model Management
class ModelManager:
    def __init__(self):
        self.models = {}
        self.current_version = "v1"
        self.feature_names = ['age', 'income', 'balance', 'city', 'has_credit_card']
    
    def load_model(self, version="v1"):
        model_path = Path(settings.model_path) if version == "v1" else Path(f"./models/gbm_pipeline_{version}.pkl")
        if not model_path.exists():
            raise RuntimeError(f"Model file not found at {model_path}")
        
        self.models[version] = joblib.load(model_path)
        registry.register_model(version, model_path, {"features": self.feature_names})
        logger.info(f"Model {version} loaded successfully")
    
    def get_model(self, version="v1"):
        return self.models.get(version)
    
    def get_feature_importance(self, version="v1"):
        model = self.get_model(version)
        if model and hasattr(model.named_steps['gbm'], 'feature_importances_'):
            importances = model.named_steps['gbm'].feature_importances_
            preprocessor = model.named_steps['preprocessor']
            feature_names = ['age', 'income', 'balance']
            
            try:
                cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['city', 'has_credit_card'])
                feature_names.extend(cat_features)
            except:
                feature_names.extend(['city_A', 'city_B', 'city_C', 'has_credit_card_0', 'has_credit_card_1'])
            
            return dict(zip(feature_names[:len(importances)], importances))
        return {}

model_manager = ModelManager()

# Authentication
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_key = credentials.credentials
    if api_key not in settings.api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return settings.api_keys[api_key]

# Rate Limiting
def check_rate_limit(user: str = Depends(verify_api_key)):
    now = time.time()
    user_requests = rate_limiter[user]
    
    while user_requests and user_requests[0] < now - 60:
        user_requests.popleft()
    
    if len(user_requests) >= settings.rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    user_requests.append(now)
    return user

# Business Logic
def calculate_risk_score(probability: float) -> str:
    if probability < settings.risk_thresholds["low"]:
        return "LOW"
    elif probability < settings.risk_thresholds["medium"]:
        return "MEDIUM"
    else:
        return "HIGH"

def calculate_confidence_interval(probability: float, n_samples: int = 1000) -> List[float]:
    std_error = np.sqrt(probability * (1 - probability) / n_samples)
    margin = 1.96 * std_error
    return [max(0, probability - margin), min(1, probability + margin)]

# Startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.start_time = time.time()
    model_manager.load_model("v1")
    yield

# App
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Production-ready ML API with monitoring, security, and advanced features",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response

@app.get("/")
def root():
    return {
        "message": settings.api_title,
        "version": settings.api_version,
        "features": ["monitoring", "security", "rate_limiting", "caching", "model_versioning", "ab_testing"],
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "models": "/models",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_manager.get_model() is not None,
        "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
        "cache_status": "connected" if cache.client else "disconnected"
    }

@app.get("/metrics")
def get_metrics(user: str = Depends(verify_api_key)):
    return monitor.get_metrics()

@app.get("/models")
def list_models(user: str = Depends(verify_api_key)):
    return {
        "available_models": list(model_manager.models.keys()),
        "current_version": model_manager.current_version,
        "feature_importance": model_manager.get_feature_importance(),
        "registry": registry.list_models()
    }

@app.post("/models/{version}/activate")
def activate_model(version: str, user: str = Depends(verify_api_key)):
    if version not in model_manager.models:
        raise HTTPException(status_code=404, detail="Model version not found")
    
    registry.activate_model(version)
    model_manager.current_version = version
    return {"message": f"Model {version} activated", "active_model": version}

@app.post("/predict", response_model=BatchResponse)
async def predict(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    user: str = Depends(check_rate_limit),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    request_id = hashlib.md5(f"{user}{start_time}".encode()).hexdigest()[:8]
    
    try:
        # A/B Testing for model version
        if not request.model_version or request.model_version == "auto":
            request.model_version = ab_manager.get_variant("model_version", user)
        
        model = model_manager.get_model(request.model_version)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Model version {request.model_version} not found")
        
        rows_data = [r.dict() for r in request.rows]
        X = pd.DataFrame(rows_data)
        
        results = []
        for i, row_data in enumerate(rows_data):
            # Check cache
            cache_key = cache.get_key(row_data)
            cached_result = cache.get(cache_key)
            
            if cached_result:
                results.append(Prediction(**cached_result))
                continue
            
            # Make prediction
            row_df = pd.DataFrame([row_data])
            prob = model.predict_proba(row_df)[0, 1]
            pred = model.predict(row_df)[0]
            
            # Calculate additional metrics
            confidence_interval = calculate_confidence_interval(prob)
            risk_score = calculate_risk_score(prob)
            
            # Feature importance
            explanation = None
            if request.explain:
                feature_importance = model_manager.get_feature_importance(request.model_version)
                explanation = {k: float(v) for k, v in list(feature_importance.items())[:5]}
            
            result_dict = {
                "prediction": int(pred),
                "probability": float(prob),
                "confidence_interval": confidence_interval,
                "risk_score": risk_score,
                "explanation": explanation
            }
            
            result = Prediction(**result_dict)
            
            # Cache result
            cache.set(cache_key, result_dict, settings.cache_ttl)
            results.append(result)
            
            # Log to database
            background_tasks.add_task(
                log_prediction_to_db,
                db, request_id, user, request.model_version,
                row_data, int(pred), float(prob), risk_score
            )
            
            # Monitor
            background_tasks.add_task(
                monitor.log_prediction,
                row_data, int(pred), float(prob), time.time() - start_time
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchResponse(
            results=results,
            model_version=request.model_version,
            processing_time_ms=processing_time,
            request_id=request_id
        )
        
    except Exception as e:
        monitor.log_error()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def log_prediction_to_db(db: Session, request_id: str, user: str, model_version: str, 
                        input_data: dict, prediction: int, probability: float, risk_score: str):
    try:
        log_entry = PredictionLog(
            request_id=request_id,
            user_id=user,
            model_version=model_version,
            input_data=json.dumps(input_data),
            prediction=prediction,
            probability=probability,
            risk_score=risk_score,
            processing_time_ms=0  # Would be calculated properly
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        logger.error(f"Database logging error: {e}")

@app.get("/predictions/history")
def get_prediction_history(
    limit: int = 100,
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    predictions = db.query(PredictionLog).filter(
        PredictionLog.user_id == user
    ).order_by(PredictionLog.timestamp.desc()).limit(limit).all()
    
    return [
        {
            "request_id": p.request_id,
            "prediction": p.prediction,
            "probability": p.probability,
            "risk_score": p.risk_score,
            "timestamp": p.timestamp.isoformat()
        }
        for p in predictions
    ]

@app.get("/favicon.ico")
def favicon():
    return {"message": "No favicon available"}

if __name__ == "__main__":
    uvicorn.run("app_enhanced:app", host="0.0.0.0", port=8000, reload=settings.debug)