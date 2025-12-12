# Enhanced Security Module with JWT and Advanced Features
import os
import jwt
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
from passlib.context import CryptContext
from fastapi import HTTPException, Request
import re

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@dataclass
class User:
    """User model with enhanced security features"""
    user_id: str
    username: str
    email: str
    hashed_password: str
    roles: List[str]
    api_keys: List[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    rate_limit_override: Optional[int] = None

@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    timestamp: datetime
    event_type: str
    user_id: str
    ip_address: str
    user_agent: str
    details: Dict[str, Any]
    severity: str

class EnhancedSecurityManager:
    """Enhanced security manager with JWT, rate limiting, and audit logging"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        
        # User storage (in production, use a proper database)
        self.users: Dict[str, User] = {}
        self.api_key_to_user: Dict[str, str] = {}
        
        # Security tracking
        self.security_events: deque = deque(maxlen=10000)
        self.failed_attempts: defaultdict = defaultdict(lambda: deque(maxlen=10))
        self.rate_limits: defaultdict = defaultdict(lambda: deque())
        self.blocked_ips: Dict[str, datetime] = {}
        
        # Security rules
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.rate_limit_window = timedelta(minutes=1)
        self.default_rate_limit = 100
        
        # Initialize default users
        self._initialize_default_users()
    
    def _initialize_default_users(self):
        """Initialize default users for testing"""
        try:
            # Demo user
            demo_user = User(
                user_id="demo-user-001",
                username="demo",
                email="demo@example.com",
                hashed_password=self.hash_password("demo123"),
                roles=["user"],
                api_keys=["demo-key-123"],
                is_active=True,
                created_at=datetime.now()
            )
            
            # Admin user
            admin_user = User(
                user_id="admin-user-001",
                username="admin",
                email="admin@example.com",
                hashed_password=self.hash_password("admin123"),
                roles=["admin", "user"],
                api_keys=["admin-key-456"],
                is_active=True,
                created_at=datetime.now(),
                rate_limit_override=1000
            )
            
            self.users[demo_user.user_id] = demo_user
            self.users[admin_user.user_id] = admin_user
            
            # Map API keys to users
            self.api_key_to_user["demo-key-123"] = demo_user.user_id
            self.api_key_to_user["admin-key-456"] = admin_user.user_id
            
            logger.info("Default users initialized")
            
        except Exception as e:
            logger.error(f"Error initializing default users: {e}")
    
    def hash_password(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        try:
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(hours=24)
            
            user = self.users.get(user_id)
            if not user:
                raise ValueError("User not found")
            
            to_encode = {
                "sub": user_id,
                "username": user.username,
                "roles": user.roles,
                "exp": expire,
                "iat": datetime.utcnow(),
                "type": "access_token"
            }
            
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Error creating access token: {e}")
            raise
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            user_id = payload.get("sub")
            if user_id is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            user = self.users.get(user_id)
            if not user or not user.is_active:
                raise HTTPException(status_code=401, detail="User not found or inactive")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def authenticate_user(self, username: str, password: str, request: Request) -> Optional[User]:
        """Authenticate user with enhanced security checks"""
        try:
            ip_address = self._get_client_ip(request)
            
            # Check if IP is blocked
            if self._is_ip_blocked(ip_address):
                self._log_security_event(
                    "blocked_ip_attempt", "", ip_address, request,
                    {"username": username}, "high"
                )
                raise HTTPException(status_code=429, detail="IP temporarily blocked")
            
            # Find user by username
            user = None
            for u in self.users.values():\n                if u.username == username:\n                    user = u\n                    break\n            \n            if not user:\n                self._handle_failed_login(username, ip_address, request, \"user_not_found\")\n                raise HTTPException(status_code=401, detail=\"Invalid credentials\")\n            \n            # Check if user is locked\n            if user.locked_until and user.locked_until > datetime.now():\n                self._log_security_event(\n                    \"locked_user_attempt\", user.user_id, ip_address, request,\n                    {\"locked_until\": user.locked_until.isoformat()}, \"medium\"\n                )\n                raise HTTPException(status_code=423, detail=\"Account temporarily locked\")\n            \n            # Verify password\n            if not self.verify_password(password, user.hashed_password):\n                self._handle_failed_login(username, ip_address, request, \"invalid_password\")\n                raise HTTPException(status_code=401, detail=\"Invalid credentials\")\n            \n            # Successful login\n            user.last_login = datetime.now()\n            user.failed_login_attempts = 0\n            user.locked_until = None\n            \n            self._log_security_event(\n                \"successful_login\", user.user_id, ip_address, request,\n                {\"username\": username}, \"info\"\n            )\n            \n            return user\n            \n        except HTTPException:\n            raise\n        except Exception as e:\n            logger.error(f\"Error authenticating user: {e}\")\n            raise HTTPException(status_code=500, detail=\"Authentication error\")\n    \n    def verify_api_key(self, api_key: str, request: Request) -> User:\n        \"\"\"Verify API key with enhanced security\"\"\"\n        try:\n            ip_address = self._get_client_ip(request)\n            \n            # Check if IP is blocked\n            if self._is_ip_blocked(ip_address):\n                self._log_security_event(\n                    \"blocked_ip_api_attempt\", \"\", ip_address, request,\n                    {\"api_key_prefix\": api_key[:8] if api_key else \"\"}, \"high\"\n                )\n                raise HTTPException(status_code=429, detail=\"IP temporarily blocked\")\n            \n            # Validate API key format\n            if not self._is_valid_api_key_format(api_key):\n                self._log_security_event(\n                    \"invalid_api_key_format\", \"\", ip_address, request,\n                    {\"api_key_prefix\": api_key[:8] if api_key else \"\"}, \"medium\"\n                )\n                raise HTTPException(status_code=401, detail=\"Invalid API key format\")\n            \n            # Find user by API key\n            user_id = self.api_key_to_user.get(api_key)\n            if not user_id:\n                self._log_security_event(\n                    \"invalid_api_key\", \"\", ip_address, request,\n                    {\"api_key_prefix\": api_key[:8]}, \"medium\"\n                )\n                raise HTTPException(status_code=401, detail=\"Invalid API key\")\n            \n            user = self.users.get(user_id)\n            if not user or not user.is_active:\n                self._log_security_event(\n                    \"inactive_user_api_attempt\", user_id, ip_address, request,\n                    {\"api_key_prefix\": api_key[:8]}, \"medium\"\n                )\n                raise HTTPException(status_code=401, detail=\"User inactive\")\n            \n            # Check rate limiting\n            self._check_rate_limit(user, ip_address, request)\n            \n            return user\n            \n        except HTTPException:\n            raise\n        except Exception as e:\n            logger.error(f\"Error verifying API key: {e}\")\n            raise HTTPException(status_code=500, detail=\"API key verification error\")\n    \n    def _handle_failed_login(self, username: str, ip_address: str, request: Request, reason: str):\n        \"\"\"Handle failed login attempts\"\"\"\n        try:\n            # Log failed attempt\n            self._log_security_event(\n                \"failed_login\", \"\", ip_address, request,\n                {\"username\": username, \"reason\": reason}, \"medium\"\n            )\n            \n            # Track failed attempts by IP\n            now = datetime.now()\n            self.failed_attempts[ip_address].append(now)\n            \n            # Clean old attempts\n            cutoff = now - timedelta(minutes=15)\n            while (self.failed_attempts[ip_address] and \n                   self.failed_attempts[ip_address][0] < cutoff):\n                self.failed_attempts[ip_address].popleft()\n            \n            # Check if IP should be blocked\n            if len(self.failed_attempts[ip_address]) >= self.max_failed_attempts:\n                self.blocked_ips[ip_address] = now + self.lockout_duration\n                self._log_security_event(\n                    \"ip_blocked\", \"\", ip_address, request,\n                    {\"failed_attempts\": len(self.failed_attempts[ip_address])}, \"high\"\n                )\n            \n            # Update user failed attempts if user exists\n            for user in self.users.values():\n                if user.username == username:\n                    user.failed_login_attempts += 1\n                    if user.failed_login_attempts >= self.max_failed_attempts:\n                        user.locked_until = now + self.lockout_duration\n                        self._log_security_event(\n                            \"user_locked\", user.user_id, ip_address, request,\n                            {\"failed_attempts\": user.failed_login_attempts}, \"high\"\n                        )\n                    break\n            \n        except Exception as e:\n            logger.error(f\"Error handling failed login: {e}\")\n    \n    def _check_rate_limit(self, user: User, ip_address: str, request: Request):\n        \"\"\"Check rate limiting for user\"\"\"\n        try:\n            now = datetime.now()\n            rate_limit = user.rate_limit_override or self.default_rate_limit\n            \n            # Clean old requests\n            cutoff = now - self.rate_limit_window\n            user_requests = self.rate_limits[user.user_id]\n            \n            while user_requests and user_requests[0] < cutoff:\n                user_requests.popleft()\n            \n            # Check rate limit\n            if len(user_requests) >= rate_limit:\n                self._log_security_event(\n                    \"rate_limit_exceeded\", user.user_id, ip_address, request,\n                    {\"requests_count\": len(user_requests), \"limit\": rate_limit}, \"medium\"\n                )\n                raise HTTPException(status_code=429, detail=\"Rate limit exceeded\")\n            \n            # Add current request\n            user_requests.append(now)\n            \n        except HTTPException:\n            raise\n        except Exception as e:\n            logger.error(f\"Error checking rate limit: {e}\")\n    \n    def _is_ip_blocked(self, ip_address: str) -> bool:\n        \"\"\"Check if IP address is blocked\"\"\"\n        if ip_address in self.blocked_ips:\n            if self.blocked_ips[ip_address] > datetime.now():\n                return True\n            else:\n                # Remove expired block\n                del self.blocked_ips[ip_address]\n        return False\n    \n    def _is_valid_api_key_format(self, api_key: str) -> bool:\n        \"\"\"Validate API key format\"\"\"\n        if not api_key or len(api_key) < 10:\n            return False\n        \n        # Check for basic format (alphanumeric with hyphens)\n        pattern = r'^[a-zA-Z0-9\\-]+$'\n        return bool(re.match(pattern, api_key))\n    \n    def _get_client_ip(self, request: Request) -> str:\n        \"\"\"Get client IP address from request\"\"\"\n        # Check for forwarded headers\n        forwarded_for = request.headers.get(\"X-Forwarded-For\")\n        if forwarded_for:\n            return forwarded_for.split(\",\")[0].strip()\n        \n        real_ip = request.headers.get(\"X-Real-IP\")\n        if real_ip:\n            return real_ip\n        \n        return request.client.host if request.client else \"unknown\"\n    \n    def _log_security_event(self, event_type: str, user_id: str, ip_address: str, \n                           request: Request, details: Dict[str, Any], severity: str):\n        \"\"\"Log security event\"\"\"\n        try:\n            event = SecurityEvent(\n                timestamp=datetime.now(),\n                event_type=event_type,\n                user_id=user_id,\n                ip_address=ip_address,\n                user_agent=request.headers.get(\"User-Agent\", \"\"),\n                details=details,\n                severity=severity\n            )\n            \n            self.security_events.append(event)\n            \n            # Log to application logger\n            log_message = f\"Security Event: {event_type} - User: {user_id} - IP: {ip_address}\"\n            if severity == \"high\":\n                logger.warning(log_message)\n            else:\n                logger.info(log_message)\n            \n        except Exception as e:\n            logger.error(f\"Error logging security event: {e}\")\n    \n    def generate_api_key(self, user_id: str) -> str:\n        \"\"\"Generate new API key for user\"\"\"\n        try:\n            # Generate secure random key\n            key_bytes = secrets.token_bytes(32)\n            api_key = f\"ak-{hashlib.sha256(key_bytes).hexdigest()[:24]}\"\n            \n            # Add to user\n            user = self.users.get(user_id)\n            if user:\n                user.api_keys.append(api_key)\n                self.api_key_to_user[api_key] = user_id\n            \n            return api_key\n            \n        except Exception as e:\n            logger.error(f\"Error generating API key: {e}\")\n            raise\n    \n    def revoke_api_key(self, api_key: str, user_id: str) -> bool:\n        \"\"\"Revoke API key\"\"\"\n        try:\n            user = self.users.get(user_id)\n            if user and api_key in user.api_keys:\n                user.api_keys.remove(api_key)\n                if api_key in self.api_key_to_user:\n                    del self.api_key_to_user[api_key]\n                return True\n            return False\n            \n        except Exception as e:\n            logger.error(f\"Error revoking API key: {e}\")\n            return False\n    \n    def get_security_metrics(self) -> Dict[str, Any]:\n        \"\"\"Get security metrics and statistics\"\"\"\n        try:\n            now = datetime.now()\n            last_24h = now - timedelta(hours=24)\n            \n            # Filter recent events\n            recent_events = [e for e in self.security_events if e.timestamp > last_24h]\n            \n            # Count events by type\n            event_counts = defaultdict(int)\n            severity_counts = defaultdict(int)\n            \n            for event in recent_events:\n                event_counts[event.event_type] += 1\n                severity_counts[event.severity] += 1\n            \n            return {\n                \"total_events_24h\": len(recent_events),\n                \"event_types\": dict(event_counts),\n                \"severity_distribution\": dict(severity_counts),\n                \"blocked_ips_count\": len([ip for ip, until in self.blocked_ips.items() if until > now]),\n                \"active_users\": len([u for u in self.users.values() if u.is_active]),\n                \"locked_users\": len([u for u in self.users.values() if u.locked_until and u.locked_until > now]),\n                \"recent_high_severity_events\": [\n                    {\n                        \"type\": e.event_type,\n                        \"timestamp\": e.timestamp.isoformat(),\n                        \"user_id\": e.user_id,\n                        \"ip_address\": e.ip_address\n                    }\n                    for e in recent_events\n                    if e.severity == \"high\"\n                ][-10:]  # Last 10 high severity events\n            }\n            \n        except Exception as e:\n            logger.error(f\"Error getting security metrics: {e}\")\n            return {\"error\": str(e)}\n    \n    def check_permission(self, user: User, permission: str) -> bool:\n        \"\"\"Check if user has specific permission\"\"\"\n        permission_map = {\n            \"predict\": [\"user\", \"admin\"],\n            \"admin\": [\"admin\"],\n            \"model_management\": [\"admin\"],\n            \"user_management\": [\"admin\"],\n            \"security_metrics\": [\"admin\"]\n        }\n        \n        required_roles = permission_map.get(permission, [])\n        return any(role in user.roles for role in required_roles)\n\n# Global security manager instance\nsecurity_manager = None\n\ndef get_security_manager(secret_key: str = None) -> EnhancedSecurityManager:\n    \"\"\"Get or create security manager instance\"\"\"\n    global security_manager\n    if security_manager is None:\n        if not secret_key:\n            secret_key = os.getenv(\"SECRET_KEY\", \"your-secret-key-change-in-production\")\n        security_manager = EnhancedSecurityManager(secret_key)\n    return security_manager