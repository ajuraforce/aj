# utils/signal_schema.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

class Signal(BaseModel):
    id: str = Field(description="Unique signal ID")
    symbol: str = Field(description="Trading symbol e.g., BTC/USDT")
    timeframe: str = Field(description="Timeframe e.g., 1h")
    signal_type: str = Field(description="BUY/SELL/SHORT/COVER")
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    reason: List[str] = Field(description="Reasons for signal")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_in: int = Field(default=600, description="Seconds until expiry")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }