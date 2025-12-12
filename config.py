import os
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    api_title: str = "Advanced Gradient Boosting ML API"
    api_version: str = "2.0"
    debug: bool = False
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    api_keys: dict = {"demo-key-123": "demo-user"}
    
    # Rate Limiting
    rate_limit_per_minute: int = 100
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")
    
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    cache_ttl: int = 300
    
    # Model Configuration
    model_path: str = "./models/gbm_pipeline.pkl"
    max_batch_size: int = 100
    
    # Monitoring
    enable_monitoring: bool = True
    log_level: str = "INFO"
    
    # Business Rules
    risk_thresholds: dict = {
        "low": 0.3,
        "medium": 0.7,
        "high": 1.0
    }
    
    class Config:
        env_file = ".env"

settings = Settings()