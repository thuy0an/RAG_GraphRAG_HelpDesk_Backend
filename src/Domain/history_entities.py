from datetime import datetime
from sqlmodel import CHAR, TIMESTAMP, SQLModel, Field, text
from sqlalchemy import Column, Index, Text
from typing import Optional
import uuid6


class ConversationHistory(SQLModel, table=True):
    """
    Schema mới: mỗi row = 1 lượt hỏi-đáp của user.
    - user_content    : câu hỏi của user
    - rag_content     : câu trả lời từ PaCRAG (NULL nếu chưa có)
    - graphrag_content: câu trả lời từ GraphRAG (NULL nếu chưa có)

    Cả hai cột answer được upsert độc lập — bất đồng bộ hoàn toàn.
    """
    __tablename__ = "conversation_history"
    __table_args__ = (
        Index('idx_session_id', 'session_id'),
    )

    id: str = Field(
        default_factory=lambda: str(uuid6.uuid7()),
        sa_column=Column(CHAR(36), primary_key=True)
    )
    session_id: str = Field(index=True)
    user_content: str = Field(sa_column=Column('user_content', Text))
    rag_content: Optional[str] = Field(
        default=None,
        sa_column=Column('rag_content', Text, nullable=True)
    )
    graphrag_content: Optional[str] = Field(
        default=None,
        sa_column=Column('graphrag_content', Text, nullable=True)
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        sa_column=Column('timestamp', TIMESTAMP, server_default=text('(now())'))
    )
