from datetime import datetime
from sqlmodel import CHAR, TIMESTAMP, SQLModel, Field, text
from sqlalchemy import Column, Index
from typing import List, Optional
from pydantic import BaseModel
import uuid6


class RetrievedPassage(BaseModel):
    content: str
    filename: str
    pages: List[int]


class CompareRun(SQLModel, table=True):
    __tablename__ = "compare_runs"
    __table_args__ = (
        Index("idx_compare_session_id", "session_id"),
        Index("idx_compare_created_at", "created_at"),
    )

    id: str = Field(default_factory=lambda: str(uuid6.uuid7()), sa_column=Column(CHAR(36), primary_key=True))
    session_id: str = Field(index=True)
    file_name: str = Field()
    file_type: Optional[str] = Field(default=None)
    file_size: Optional[int] = Field(default=None)
    pac_ingest_json: str = Field()
    graphrag_ingest_json: str = Field()
    pac_query_json: Optional[str] = Field(default=None)
    graphrag_query_json: Optional[str] = Field(default=None)
    query_text: Optional[str] = Field(default=None)
    errors_json: Optional[str] = Field(default=None)
    created_at: Optional[datetime] = Field(default=None, sa_column=Column("created_at", TIMESTAMP, server_default=text("(now())")))
