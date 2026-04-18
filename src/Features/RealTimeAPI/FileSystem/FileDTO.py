from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class FileSearchRequest(BaseModel):
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(10, ge=1, le=100, description="Page size")