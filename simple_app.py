# Simple version of the enhanced ML API for quick testing
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple input validation
class SingleRow(BaseModel):
    age: float = Field(..., ge=18, le=100)
    income: float = Field(..., ge=-50000, le=500000)
    balance: float = Field(..., ge=-10000, le=100000)
    city: str = Field(..., regex="^[ABC]$")
    has_credit_card: int = Field(..., ge=0, le=1)

class PredictionRequest(BaseModel):
    rows: List[SingleRow] = Field(..., max_items=100)
    model_version: Optional[str] = Field("v1")

class Prediction(BaseModel):
    prediction: int
    probability: float
    risk_score: str

class PredictionResponse(BaseModel):
    results: List[Prediction]
    processing_time_ms: float

# Simple model manager
class SimpleModelManager:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        try:
            import joblib
            model_path = Path("models/gbm_pipeline.pkl")
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info("Model loaded successfully")
            else:
                logger.warning("Model file not found, using dummy model")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def predict(self, data: pd.DataFrame):
        if self.model:
            try:
                predictions = self.model.predict(data)
                probabilities = self.model.predict_proba(data)[:, 1]
                return predictions, probabilities
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                # Fallback to dummy predictions
                return self._dummy_predict(data)
        else:
            return self._dummy_predict(data)
    
    def _dummy_predict(self, data: pd.DataFrame):
        # Simple dummy predictions for testing
        n_samples = len(data)
        predictions = np.random.choice([0, 1], n_samples)
        probabilities = np.random.uniform(0.1, 0.9, n_samples)
        return predictions, probabilities

model_manager = SimpleModelManager()

# Create FastAPI app
app = FastAPI(
    title="Simple ML API",
    version="1.0",
    description="Simplified ML API for testing"
)

@app.get("/")
def root():
    return {
        "message": "Simple ML API",
        "version": "1.0",
        "status": "running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_manager.model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start_time = time.time()
    
    try:
        # Convert to DataFrame
        rows_data = [r.dict() for r in request.rows]
        df = pd.DataFrame(rows_data)
        
        # Make predictions
        predictions, probabilities = model_manager.predict(df)
        
        # Format results
        results = []
        for pred, prob in zip(predictions, probabilities):
            # Simple risk scoring
            if prob < 0.3:
                risk_score = "LOW"
            elif prob < 0.7:
                risk_score = "MEDIUM"
            else:
                risk_score = "HIGH"
            
            results.append(Prediction(
                prediction=int(pred),
                probability=float(prob),
                risk_score=risk_score
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            results=results,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("simple_app:app", host="0.0.0.0", port=8000, reload=True)