# Simple database module
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, index=True)
    user_id = Column(String, index=True)
    model_version = Column(String)
    input_data = Column(Text)
    prediction = Column(Integer)
    probability = Column(Float)
    risk_score = Column(String)
    processing_time_ms = Column(Float)
    timestamp = Column(DateTime, default=datetime.now)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()