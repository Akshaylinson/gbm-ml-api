import time
from collections import defaultdict, deque
from typing import Dict, Optional
from fastapi import HTTPException, status
from config import settings
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self):
        self.user_requests = defaultdict(lambda: deque())
        self.user_limits = defaultdict(lambda: settings.rate_limit_per_minute)
        self.blocked_users = {}
        
    def check_rate_limit(self, user_id: str, custom_limit: Optional[int] = None) -> bool:
        """Check if user is within rate limit"""
        now = time.time()
        limit = custom_limit or self.user_limits[user_id]
        
        # Check if user is temporarily blocked
        if user_id in self.blocked_users:
            if now < self.blocked_users[user_id]:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="User temporarily blocked due to rate limit violations"
                )
            else:
                del self.blocked_users[user_id]
        
        user_requests = self.user_requests[user_id]
        
        # Remove old requests (older than 1 minute)
        while user_requests and user_requests[0] < now - 60:
            user_requests.popleft()
        
        # Check if limit exceeded
        if len(user_requests) >= limit:
            # Block user for 5 minutes after multiple violations
            violation_count = getattr(self, f"_violations_{user_id}", 0)
            violation_count += 1
            setattr(self, f"_violations_{user_id}", violation_count)
            
            if violation_count >= 3:
                self.blocked_users[user_id] = now + 300  # 5 minutes
                logger.warning(f"User {user_id} blocked for repeated rate limit violations")
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Limit: {limit} requests per minute"
            )
        
        # Add current request
        user_requests.append(now)
        return True
    
    def set_user_limit(self, user_id: str, limit: int):
        """Set custom rate limit for a user"""
        self.user_limits[user_id] = limit
        logger.info(f"Rate limit for user {user_id} set to {limit} requests/minute")
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get rate limiting stats for a user"""
        now = time.time()
        user_requests = self.user_requests[user_id]
        
        # Clean old requests
        while user_requests and user_requests[0] < now - 60:
            user_requests.popleft()
        
        return {
            "user_id": user_id,
            "current_requests": len(user_requests),
            "limit": self.user_limits[user_id],
            "remaining": max(0, self.user_limits[user_id] - len(user_requests)),
            "reset_time": int(now + 60) if user_requests else int(now),
            "is_blocked": user_id in self.blocked_users
        }
    
    def reset_user(self, user_id: str):
        """Reset rate limiting for a user"""
        if user_id in self.user_requests:
            self.user_requests[user_id].clear()
        if user_id in self.blocked_users:
            del self.blocked_users[user_id]
        if hasattr(self, f"_violations_{user_id}"):
            delattr(self, f"_violations_{user_id}")
        logger.info(f"Rate limiting reset for user {user_id}")

# Global rate limiter instance
rate_limiter = RateLimiter()