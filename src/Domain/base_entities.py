import uuid6
from typing import Optional
import datetime
import enum

from sqlalchemy import CHAR, Column, Enum, Index, Integer, String, TIMESTAMP, Text, text
from sqlmodel import Field, SQLModel

class AccountsRole(str, enum.Enum):
    ADMIN = 'ADMIN'
    AGENT = 'AGENT'
    CUSTOMER = 'CUSTOMER'


class Accounts(SQLModel, table=True):
    __tablename__ = 'Accounts'
    __table_args__ = (
        Index('email', 'email', unique=True),
        Index('username', 'username', unique=True)
    )

    id: str = Field(default_factory=lambda: str(uuid6.uuid7()), sa_column=Column(CHAR(36), primary_key=True))
    username: Optional[str] = Field(default=None, sa_column=Column('username', String(255)))
    email: Optional[str] = Field(default=None, sa_column=Column('email', String(255)))
    full_name: Optional[str] = Field(default=None, sa_column=Column('full_name', String(255)))
    password: Optional[str] = Field(default=None, sa_column=Column('password', String(128)))
    role: Optional[AccountsRole] = Field(default=None, sa_column=Column('role', Enum(AccountsRole, values_callable=lambda cls: [member.value for member in cls]), server_default=text("'CUSTOMER'")))
    created_at: Optional[datetime.datetime] = Field(default=None, sa_column=Column('created_at', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP')))
    delete_at: Optional[datetime.datetime] = Field(default=None, sa_column=Column('delete_at', TIMESTAMP))


class Attachments(SQLModel, table=True):
    __tablename__ = 'Attachments'

    id: str = Field(default_factory=lambda: str(uuid6.uuid7()), sa_column=Column(CHAR(36), primary_key=True))
    type: Optional[str] = Field(default=None, sa_column=Column('type', String(128)))
    file_name: Optional[str] = Field(default=None, sa_column=Column('file_name', String(255)))
    url: Optional[str] = Field(default=None, sa_column=Column('url', String(255)))
    created_at: Optional[datetime.datetime] = Field(default=None, sa_column=Column('created_at', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP')))
    delete_at: Optional[datetime.datetime] = Field(default=None, sa_column=Column('delete_at', TIMESTAMP))


class CompareRuns(SQLModel, table=True):
    __tablename__ = 'CompareRuns'
    __table_args__ = (
        Index('idx_compare_created_at', 'created_at'),
        Index('idx_compare_session_id', 'session_id'),
        Index('idx_compare_session_time', 'session_id', 'created_at')
    )

    id: str = Field(default_factory=lambda: str(uuid6.uuid7()), sa_column=Column(CHAR(36), primary_key=True))
    session_id: Optional[str] = Field(default=None, sa_column=Column('session_id', String(255)))
    file_name: Optional[str] = Field(default=None, sa_column=Column('file_name', String(255)))
    file_type: Optional[str] = Field(default=None, sa_column=Column('file_type', String(100)))
    file_size: Optional[int] = Field(default=None, sa_column=Column('file_size', Integer))
    pac_ingest_json: Optional[str] = Field(default=None, sa_column=Column('pac_ingest_json', Text))
    graphrag_ingest_json: Optional[str] = Field(default=None, sa_column=Column('graphrag_ingest_json', Text))
    pac_query_json: Optional[str] = Field(default=None, sa_column=Column('pac_query_json', Text))
    graphrag_query_json: Optional[str] = Field(default=None, sa_column=Column('graphrag_query_json', Text))
    errors_json: Optional[str] = Field(default=None, sa_column=Column('errors_json', Text))
    created_at: Optional[datetime.datetime] = Field(default=None, sa_column=Column('created_at', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP')))


class ConversationHistories(SQLModel, table=True):
    __tablename__ = 'ConversationHistories'
    __table_args__ = (
        Index('idx_session_id', 'session_id'),
        Index('idx_session_time', 'session_id', 'timestamp')
    )

    id: str = Field(default_factory=lambda: str(uuid6.uuid7()), sa_column=Column(CHAR(36), primary_key=True))
    session_id: Optional[str] = Field(default=None, sa_column=Column('session_id', String(255)))
    role: Optional[str] = Field(default=None, sa_column=Column('role', String(50)))
    content: Optional[str] = Field(default=None, sa_column=Column('content', Text))
    timestamp: Optional[datetime.datetime] = Field(default=None, sa_column=Column('timestamp', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP')))
