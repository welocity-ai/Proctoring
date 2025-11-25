from pydantic import BaseModel, Field
from typing import Optional, Union

class SessionEvent(BaseModel):
    type: str
    event: Optional[Union[str, dict]] = None
    b64: Optional[str] = None
    ts: Optional[int] = None
    duration: Optional[int] = None

    class Config:
        extra = "ignore"
