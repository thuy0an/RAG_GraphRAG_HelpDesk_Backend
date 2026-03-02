from typing import Any, Optional, Union
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Null
from sqlmodel import JSON


class TicketBaseDTO(BaseModel):
    subject: Optional[str] = Field(default=None, example=None)
    description: Optional[str] = Field(default=None, example=None)
    category: Optional[str] = Field(default=None, example=None)

class TicketCreateDTO(TicketBaseDTO):
    customer_id: Optional[str] = Field(default=None, example=None)
    dept_id: Optional[str] = Field(default=None, example=None)

class TicketUpdateDTO(BaseModel):
    subject: Optional[str] = Field(default=None, example=None)
    description: Optional[str] = Field(default=None, example=None)
    status: Optional[str] = Field(default=None, example=None)
    priority: Optional[str] = Field(default=None, example=None)
    category: Optional[str] = Field(default=None, example=None)

    customer_id: Optional[str] = Field(default=None, example=None)
    assigned_agent_id: Optional[str] = Field(default=None, example=None)
    dept_id: Optional[str] = Field(default=None, example=None)
    due_date: Optional[str] = None
    
    attachment_url: Optional[str] = None
    
    satisfaction_rating: Optional[int] = None
    customer_feedback: Optional[str] = None

class TicketSearchRequest(BaseModel):
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(10, ge=1, le=100, description="Page size")
    category: Optional[str] = Field(None, description="Filter by category")
    department_name: Optional[str] = Field(None, description="Filter by department")
    status: Optional[str] = Field(None, description="Filter by status")
    priority: Optional[str] = Field(None, description="Filter by priority")

