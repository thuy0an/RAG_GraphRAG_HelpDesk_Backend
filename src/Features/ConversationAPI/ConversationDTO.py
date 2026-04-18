from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class AddConversationHistoryRequest(BaseModel):
    session_id: str = Field(..., description="Session ID for the conversation")
    role: str = Field(..., description="Role of the message sender (e.g., 'adv-user', 'adv-assistant', 'graph-user', 'graph-assistant')")
    content: str = Field(..., description="Content of the message")


class SearchConversationHistoriesRequest(BaseModel):
    page: int = Field(default=1, ge=1, description="Page number (starts from 1)")
    page_size: int = Field(default=10, ge=1, le=100, description="Number of items per page")
    role: Optional[str] = Field(default=None, description="Filter by role (comma-separated for multiple roles, e.g., 'adv-user,adv-assistant')")

class ConversationHistoryResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    timestamp: datetime


class SessionListResponse(BaseModel):
    sessions: List[str]
    total_count: int
