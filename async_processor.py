# Async Batch Processing with Celery
import os
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from celery import Celery
from celery.result import AsyncResult
import redis

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BatchJob:
    """Batch processing job"""
    job_id: str
    user_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_rows: int = 0
    processed_rows: int = 0
    failed_rows: int = 0
    model_version: str = "v1"
    input_data: Optional[List[Dict[str, Any]]] = None
    results: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    progress_percentage: float = 0.0
    estimated_completion: Optional[datetime] = None

class AsyncBatchProcessor:
    """Async batch processing manager"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 celery_broker: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.celery_app = self._create_celery_app(celery_broker)
        self.jobs: Dict[str, BatchJob] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Job configuration
        self.max_batch_size = 10000
        self.chunk_size = 100
        self.job_timeout = timedelta(hours=2)
        
    def _create_celery_app(self, broker_url: str) -> Celery:
        """Create and configure Celery app"""
        app = Celery('ml_api_processor', broker=broker_url)
        
        app.conf.update(
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
            task_track_started=True,
            task_time_limit=7200,  # 2 hours
            task_soft_time_limit=7000,  # 1h 56m
            worker_prefetch_multiplier=1,
            task_acks_late=True,
            worker_disable_rate_limits=False,
            task_compression='gzip',
            result_compression='gzip',
        )
        
        return app
    
    async def submit_batch_job(self, user_id: str, input_data: List[Dict[str, Any]], 
                              model_version: str = "v1", 
                              options: Optional[Dict[str, Any]] = None) -> str:
        """Submit a new batch processing job"""
        try:
            # Validate input
            if len(input_data) > self.max_batch_size:
                raise ValueError(f"Batch size exceeds maximum of {self.max_batch_size}")
            
            if not input_data:
                raise ValueError("Input data cannot be empty")
            
            # Create job
            job_id = str(uuid.uuid4())
            job = BatchJob(
                job_id=job_id,
                user_id=user_id,
                status=JobStatus.PENDING,
                created_at=datetime.now(),
                total_rows=len(input_data),
                model_version=model_version,
                input_data=input_data
            )\n            \n            # Store job\n            self.jobs[job_id] = job\n            await self._store_job_in_redis(job)\n            \n            # Submit to Celery\n            celery_task = self.celery_app.send_task(\n                'process_batch_predictions',\n                args=[job_id, input_data, model_version, options or {}],\n                task_id=job_id\n            )\n            \n            logger.info(f\"Batch job {job_id} submitted for user {user_id}\")\n            return job_id\n            \n        except Exception as e:\n            logger.error(f\"Error submitting batch job: {e}\")\n            raise\n    \n    async def get_job_status(self, job_id: str) -> Optional[BatchJob]:\n        \"\"\"Get job status and details\"\"\"\n        try:\n            # Try to get from memory first\n            if job_id in self.jobs:\n                job = self.jobs[job_id]\n            else:\n                # Try to load from Redis\n                job = await self._load_job_from_redis(job_id)\n                if job:\n                    self.jobs[job_id] = job\n            \n            if not job:\n                return None\n            \n            # Update status from Celery if still processing\n            if job.status in [JobStatus.PENDING, JobStatus.PROCESSING]:\n                await self._update_job_from_celery(job)\n            \n            return job\n            \n        except Exception as e:\n            logger.error(f\"Error getting job status: {e}\")\n            return None\n    \n    async def cancel_job(self, job_id: str, user_id: str) -> bool:\n        \"\"\"Cancel a running job\"\"\"\n        try:\n            job = await self.get_job_status(job_id)\n            if not job or job.user_id != user_id:\n                return False\n            \n            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:\n                return False\n            \n            # Cancel Celery task\n            self.celery_app.control.revoke(job_id, terminate=True)\n            \n            # Update job status\n            job.status = JobStatus.CANCELLED\n            job.completed_at = datetime.now()\n            \n            await self._store_job_in_redis(job)\n            \n            logger.info(f\"Job {job_id} cancelled by user {user_id}\")\n            return True\n            \n        except Exception as e:\n            logger.error(f\"Error cancelling job: {e}\")\n            return False\n    \n    async def get_user_jobs(self, user_id: str, limit: int = 50) -> List[BatchJob]:\n        \"\"\"Get jobs for a specific user\"\"\"\n        try:\n            user_jobs = []\n            \n            # Get from memory\n            for job in self.jobs.values():\n                if job.user_id == user_id:\n                    user_jobs.append(job)\n            \n            # Get from Redis if needed\n            if len(user_jobs) < limit:\n                redis_jobs = await self._get_user_jobs_from_redis(user_id, limit)\n                for job in redis_jobs:\n                    if job.job_id not in self.jobs:\n                        user_jobs.append(job)\n                        self.jobs[job.job_id] = job\n            \n            # Sort by creation time (newest first)\n            user_jobs.sort(key=lambda x: x.created_at, reverse=True)\n            \n            return user_jobs[:limit]\n            \n        except Exception as e:\n            logger.error(f\"Error getting user jobs: {e}\")\n            return []\n    \n    async def cleanup_old_jobs(self, max_age_days: int = 7):\n        \"\"\"Clean up old completed jobs\"\"\"\n        try:\n            cutoff_date = datetime.now() - timedelta(days=max_age_days)\n            \n            # Clean from memory\n            jobs_to_remove = []\n            for job_id, job in self.jobs.items():\n                if (job.completed_at and job.completed_at < cutoff_date and \n                    job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]):\n                    jobs_to_remove.append(job_id)\n            \n            for job_id in jobs_to_remove:\n                del self.jobs[job_id]\n            \n            # Clean from Redis\n            await self._cleanup_redis_jobs(cutoff_date)\n            \n            logger.info(f\"Cleaned up {len(jobs_to_remove)} old jobs\")\n            \n        except Exception as e:\n            logger.error(f\"Error cleaning up old jobs: {e}\")\n    \n    async def _store_job_in_redis(self, job: BatchJob):\n        \"\"\"Store job in Redis\"\"\"\n        try:\n            job_data = asdict(job)\n            # Convert datetime objects to ISO strings\n            for key, value in job_data.items():\n                if isinstance(value, datetime):\n                    job_data[key] = value.isoformat()\n                elif isinstance(value, JobStatus):\n                    job_data[key] = value.value\n            \n            self.redis_client.setex(\n                f\"job:{job.job_id}\",\n                timedelta(days=7),  # TTL\n                json.dumps(job_data)\n            )\n            \n            # Add to user index\n            self.redis_client.zadd(\n                f\"user_jobs:{job.user_id}\",\n                {job.job_id: job.created_at.timestamp()}\n            )\n            \n        except Exception as e:\n            logger.error(f\"Error storing job in Redis: {e}\")\n    \n    async def _load_job_from_redis(self, job_id: str) -> Optional[BatchJob]:\n        \"\"\"Load job from Redis\"\"\"\n        try:\n            job_data = self.redis_client.get(f\"job:{job_id}\")\n            if not job_data:\n                return None\n            \n            data = json.loads(job_data)\n            \n            # Convert ISO strings back to datetime objects\n            for key in ['created_at', 'started_at', 'completed_at', 'estimated_completion']:\n                if data.get(key):\n                    data[key] = datetime.fromisoformat(data[key])\n            \n            # Convert status back to enum\n            if 'status' in data:\n                data['status'] = JobStatus(data['status'])\n            \n            return BatchJob(**data)\n            \n        except Exception as e:\n            logger.error(f\"Error loading job from Redis: {e}\")\n            return None\n    \n    async def _get_user_jobs_from_redis(self, user_id: str, limit: int) -> List[BatchJob]:\n        \"\"\"Get user jobs from Redis\"\"\"\n        try:\n            # Get job IDs from sorted set (newest first)\n            job_ids = self.redis_client.zrevrange(\n                f\"user_jobs:{user_id}\", 0, limit - 1\n            )\n            \n            jobs = []\n            for job_id in job_ids:\n                job = await self._load_job_from_redis(job_id.decode())\n                if job:\n                    jobs.append(job)\n            \n            return jobs\n            \n        except Exception as e:\n            logger.error(f\"Error getting user jobs from Redis: {e}\")\n            return []\n    \n    async def _cleanup_redis_jobs(self, cutoff_date: datetime):\n        \"\"\"Clean up old jobs from Redis\"\"\"\n        try:\n            # This would require scanning all job keys in a production system\n            # For now, we'll rely on TTL for cleanup\n            pass\n            \n        except Exception as e:\n            logger.error(f\"Error cleaning up Redis jobs: {e}\")\n    \n    async def _update_job_from_celery(self, job: BatchJob):\n        \"\"\"Update job status from Celery\"\"\"\n        try:\n            result = AsyncResult(job.job_id, app=self.celery_app)\n            \n            if result.state == 'PENDING':\n                job.status = JobStatus.PENDING\n            elif result.state == 'STARTED':\n                job.status = JobStatus.PROCESSING\n                if not job.started_at:\n                    job.started_at = datetime.now()\n            elif result.state == 'SUCCESS':\n                job.status = JobStatus.COMPLETED\n                job.completed_at = datetime.now()\n                job.progress_percentage = 100.0\n                \n                # Get results\n                if result.result:\n                    job.results = result.result.get('results', [])\n                    job.processed_rows = result.result.get('processed_rows', 0)\n                    job.failed_rows = result.result.get('failed_rows', 0)\n            elif result.state == 'FAILURE':\n                job.status = JobStatus.FAILED\n                job.completed_at = datetime.now()\n                job.error_message = str(result.result)\n            elif result.state == 'REVOKED':\n                job.status = JobStatus.CANCELLED\n                job.completed_at = datetime.now()\n            \n            # Update progress if available\n            if hasattr(result, 'info') and isinstance(result.info, dict):\n                progress = result.info.get('progress', 0)\n                job.progress_percentage = progress\n                job.processed_rows = result.info.get('processed_rows', 0)\n                \n                # Estimate completion time\n                if progress > 0 and job.started_at:\n                    elapsed = datetime.now() - job.started_at\n                    estimated_total = elapsed * (100 / progress)\n                    job.estimated_completion = job.started_at + estimated_total\n            \n            await self._store_job_in_redis(job)\n            \n        except Exception as e:\n            logger.error(f\"Error updating job from Celery: {e}\")\n    \n    def get_processing_stats(self) -> Dict[str, Any]:\n        \"\"\"Get processing statistics\"\"\"\n        try:\n            now = datetime.now()\n            last_24h = now - timedelta(hours=24)\n            \n            # Count jobs by status\n            status_counts = {status.value: 0 for status in JobStatus}\n            recent_jobs = 0\n            total_processed_rows = 0\n            avg_processing_time = 0\n            processing_times = []\n            \n            for job in self.jobs.values():\n                status_counts[job.status.value] += 1\n                \n                if job.created_at > last_24h:\n                    recent_jobs += 1\n                \n                if job.status == JobStatus.COMPLETED:\n                    total_processed_rows += job.processed_rows\n                    \n                    if job.started_at and job.completed_at:\n                        processing_time = (job.completed_at - job.started_at).total_seconds()\n                        processing_times.append(processing_time)\n            \n            if processing_times:\n                avg_processing_time = sum(processing_times) / len(processing_times)\n            \n            return {\n                \"total_jobs\": len(self.jobs),\n                \"recent_jobs_24h\": recent_jobs,\n                \"status_distribution\": status_counts,\n                \"total_processed_rows\": total_processed_rows,\n                \"avg_processing_time_seconds\": avg_processing_time,\n                \"active_jobs\": status_counts[JobStatus.PROCESSING.value] + status_counts[JobStatus.PENDING.value]\n            }\n            \n        except Exception as e:\n            logger.error(f\"Error getting processing stats: {e}\")\n            return {\"error\": str(e)}\n\n# Celery task definition (would be in a separate file in production)\n@celery_app.task(bind=True)\ndef process_batch_predictions(self, job_id: str, input_data: List[Dict[str, Any]], \n                            model_version: str, options: Dict[str, Any]):\n    \"\"\"Celery task for batch prediction processing\"\"\"\n    try:\n        from model_manager import ModelManager\n        import joblib\n        \n        # Load model\n        model_manager = ModelManager()\n        model = model_manager.get_model(model_version)\n        \n        if not model:\n            raise ValueError(f\"Model {model_version} not found\")\n        \n        results = []\n        processed_rows = 0\n        failed_rows = 0\n        \n        total_rows = len(input_data)\n        chunk_size = options.get('chunk_size', 100)\n        \n        # Process in chunks\n        for i in range(0, total_rows, chunk_size):\n            chunk = input_data[i:i + chunk_size]\n            \n            try:\n                # Convert to DataFrame\n                df = pd.DataFrame(chunk)\n                \n                # Make predictions\n                predictions = model.predict(df)\n                probabilities = model.predict_proba(df)[:, 1]\n                \n                # Process results\n                for j, (pred, prob) in enumerate(zip(predictions, probabilities)):\n                    results.append({\n                        \"row_index\": i + j,\n                        \"prediction\": int(pred),\n                        \"probability\": float(prob),\n                        \"risk_score\": \"HIGH\" if prob > 0.7 else \"MEDIUM\" if prob > 0.3 else \"LOW\"\n                    })\n                \n                processed_rows += len(chunk)\n                \n            except Exception as e:\n                logger.error(f\"Error processing chunk {i}-{i+len(chunk)}: {e}\")\n                failed_rows += len(chunk)\n            \n            # Update progress\n            progress = (processed_rows + failed_rows) / total_rows * 100\n            self.update_state(\n                state='PROGRESS',\n                meta={\n                    'progress': progress,\n                    'processed_rows': processed_rows,\n                    'failed_rows': failed_rows\n                }\n            )\n        \n        return {\n            'results': results,\n            'processed_rows': processed_rows,\n            'failed_rows': failed_rows,\n            'total_rows': total_rows\n        }\n        \n    except Exception as e:\n        logger.error(f\"Batch processing error: {e}\")\n        raise\n\n# Global processor instance\nbatch_processor = None\n\ndef get_batch_processor(redis_url: str = \"redis://localhost:6379\") -> AsyncBatchProcessor:\n    \"\"\"Get or create batch processor instance\"\"\"\n    global batch_processor\n    if batch_processor is None:\n        batch_processor = AsyncBatchProcessor(redis_url)\n    return batch_processor