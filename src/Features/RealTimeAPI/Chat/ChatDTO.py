from typing import Optional
from pydantic import BaseModel


class MessageRequest(BaseModel):
    user_id: str
    content: str
    receiver_id: Optional[str] = None 