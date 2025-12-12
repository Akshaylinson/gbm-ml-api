from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hashlib
import secrets
import re
from config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class SecurityManager:
    def __init__(self):
        self.api_keys = {
            "demo-key-123": {"user": "demo-user", "permissions": ["read", "predict"]},
            "admin-key-456": {"user": "admin", "permissions": ["read", "predict", "admin"]}
        }
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        try:
            payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
    
    def generate_api_key(self) -> str:
        return secrets.token_urlsafe(32)
    
    def sanitize_input(self, value: Any) -> Any:
        """Sanitize input to prevent injection attacks"""
        if isinstance(value, str):
            # Remove potentially dangerous characters
            value = re.sub(r'[<>"\';\\]', '', value)
            return value.strip()[:100]  # Limit length
        return value
    
    def validate_api_key(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        api_key = credentials.credentials
        if api_key not in self.api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        return self.api_keys[api_key]

security_manager = SecurityManager()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return security_manager.validate_api_key(credentials)

def require_permission(permission: str):
    def permission_checker(user_info: dict = Depends(get_current_user)):
        if permission not in user_info.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return user_info
    return permission_checker