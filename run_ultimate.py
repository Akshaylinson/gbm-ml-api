# Ultimate ML API Runner with Enhanced Features
import os
import sys
import time
import logging
import asyncio
from pathlib import Path
import uvicorn
from contextlib import asynccontextmanager

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Create logs directory first
Path('logs').mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/api.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment and create necessary directories"""
    try:
        # Create necessary directories
        directories = ['logs', 'models', 'data']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.info(f"Created/verified directory: {directory}")
        
        # Set default environment variables if not set
        env_defaults = {
            'SECRET_KEY': 'your-secret-key-change-in-production',
            'DATABASE_URL': 'sqlite:///./predictions.db',
            'REDIS_URL': 'redis://localhost:6379',
            'RATE_LIMIT_PER_MINUTE': '100',
            'DEBUG': 'false',
            'LOG_LEVEL': 'INFO'
        }
        
        for key, default_value in env_defaults.items():
            if key not in os.environ:
                os.environ[key] = default_value
                logger.info(f"Set default environment variable: {key}")
        
        logger.info("Environment setup completed")
        
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")
        raise

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        # Check if model file exists
        model_path = Path("models/gbm_pipeline.pkl")
        if not model_path.exists():
            logger.warning(f"Model file not found at {model_path}")
            logger.info("You may need to train the model first using train.py")
        
        # Check Redis connection (optional)
        try:
            import redis
            redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
            redis_client.ping()
            logger.info("Redis connection successful")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            logger.info("API will work without Redis but with limited caching")
        
        # Check database connection
        try:
            from database import engine
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info("Database connection successful")
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            logger.info("API will work with SQLite fallback")
        
        logger.info("Dependency check completed")
        
    except Exception as e:
        logger.error(f"Error checking dependencies: {e}")

def run_development_server():
    """Run development server with hot reload"""
    logger.info("Starting development server...")
    
    uvicorn.run(
        "app_ultimate:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        reload_dirs=["./"],
        reload_excludes=["logs/*", "*.log", "__pycache__/*"]
    )

def run_production_server():
    """Run production server with multiple workers"""
    logger.info("Starting production server...")
    
    # Calculate optimal number of workers
    import multiprocessing
    workers = min(multiprocessing.cpu_count(), 4)
    
    uvicorn.run(
        "app_ultimate:app",
        host="0.0.0.0",
        port=8000,
        workers=workers,
        log_level="info",
        access_log=True,
        loop="uvloop",  # Use uvloop for better performance
        http="httptools"  # Use httptools for better performance
    )

def run_with_gunicorn():
    """Run with Gunicorn for production deployment"""
    logger.info("Starting with Gunicorn...")
    
    import subprocess
    import multiprocessing
    
    workers = min(multiprocessing.cpu_count(), 4)
    
    cmd = [
        "gunicorn",
        "app_ultimate:app",
        "-w", str(workers),
        "-k", "uvicorn.workers.UvicornWorker",
        "--bind", "0.0.0.0:8000",
        "--log-level", "info",
        "--access-logfile", "-",
        "--error-logfile", "-",
        "--timeout", "120",
        "--keep-alive", "5",
        "--max-requests", "1000",
        "--max-requests-jitter", "100"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Gunicorn failed to start: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")

def run_celery_worker():
    """Run Celery worker for async processing"""
    logger.info("Starting Celery worker...")
    
    import subprocess
    
    cmd = [
        "celery",
        "-A", "async_processor.celery_app",
        "worker",
        "--loglevel=info",
        "--concurrency=4",
        "--max-tasks-per-child=1000"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Celery worker failed to start: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down Celery worker...")

def run_celery_beat():
    """Run Celery beat for scheduled tasks"""
    logger.info("Starting Celery beat...")
    
    import subprocess
    
    cmd = [
        "celery",
        "-A", "async_processor.celery_app",
        "beat",
        "--loglevel=info"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Celery beat failed to start: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down Celery beat...")

def run_dashboard():
    """Run Streamlit dashboard"""
    logger.info("Starting Streamlit dashboard...")
    
    import subprocess
    
    cmd = [
        "streamlit",
        "run",
        "dashboard.py",
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Dashboard failed to start: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down dashboard...")

def run_simple_server():
    """Run simple server without complex dependencies"""
    logger.info("Starting simple development server...")
    
    uvicorn.run(
        "simple_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )

def show_help():
    """Show help message"""
    help_text = """
Ultimate ML API Runner

Usage: python run_ultimate.py [command]

Commands:
  simple      Run simple server without complex dependencies
  dev         Run development server with hot reload (default)
  prod        Run production server with multiple workers
  gunicorn    Run with Gunicorn for production deployment
  worker      Run Celery worker for async processing
  beat        Run Celery beat for scheduled tasks
  dashboard   Run Streamlit dashboard
  help        Show this help message

Environment Variables:
  SECRET_KEY              JWT secret key
  DATABASE_URL           Database connection string
  REDIS_URL              Redis connection string
  RATE_LIMIT_PER_MINUTE  Rate limit per user
  DEBUG                  Debug mode (true/false)
  LOG_LEVEL              Logging level (DEBUG/INFO/WARNING/ERROR)

Examples:
  python run_ultimate.py dev
  python run_ultimate.py prod
  python run_ultimate.py worker
  
For Docker deployment:
  docker-compose -f docker-compose-ultimate.yml up -d
"""
    print(help_text)

def main():
    """Main entry point"""
    try:
        # Setup environment
        setup_environment()
        
        # Check dependencies
        check_dependencies()
        
        # Get command from arguments
        command = sys.argv[1] if len(sys.argv) > 1 else "dev"
        
        # Route to appropriate function
        if command == "simple":
            run_simple_server()
        elif command == "dev":
            run_development_server()
        elif command == "prod":
            run_production_server()
        elif command == "gunicorn":
            run_with_gunicorn()
        elif command == "worker":
            run_celery_worker()
        elif command == "beat":
            run_celery_beat()
        elif command == "dashboard":
            run_dashboard()
        elif command == "help":
            show_help()
        else:
            logger.error(f"Unknown command: {command}")
            show_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()