import uuid6
from typing import Optional
import datetime

from sqlalchemy import CHAR, Column, Enum, Index, Integer, JSON, String, TIMESTAMP, Text, text
from sqlalchemy.dialects.mysql import TINYINT, VARCHAR
from sqlmodel import Field, SQLModel

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
    role: Optional[str] = Field(default=None, sa_column=Column('role', Enum('ADMIN', 'AGENT', 'CUSTOMER'), server_default=text("'CUSTOMER'")))
    department_id: Optional[str] = Field(default=None, sa_column=Column('department_id', CHAR(36)))
    created_at: Optional[datetime.datetime] = Field(default=None, sa_column=Column('created_at', TIMESTAMP, server_default=text('(now())')))
    delete_at: Optional[datetime.datetime] = Field(default=None, sa_column=Column('delete_at', TIMESTAMP))


class Attachment(SQLModel, table=True):
    __tablename__ = 'Attachment'

    id: str = Field(default_factory=lambda: str(uuid6.uuid7()), sa_column=Column(CHAR(36), primary_key=True))
    file_name: Optional[str] = Field(default=None, sa_column=Column('file_name', VARCHAR(255)))
    url: Optional[str] = Field(default=None, sa_column=Column('url', VARCHAR(255)))
    created_at: Optional[datetime.datetime] = Field(default=None, sa_column=Column('created_at', TIMESTAMP, server_default=text('(now())')))
    delete_at: Optional[datetime.datetime] = Field(default=None, sa_column=Column('delete_at', TIMESTAMP))


class Departments(SQLModel, table=True):
    __tablename__ = 'Departments'

    id: str = Field(default_factory=lambda: str(uuid6.uuid7()), sa_column=Column(CHAR(36), primary_key=True))
    name: Optional[str] = Field(default=None, sa_column=Column('name', String(255), comment='Phòng ban: Kỹ thuật, Kinh doanh, Bảo hành...'))
    created_at: Optional[datetime.datetime] = Field(default=None, sa_column=Column('created_at', TIMESTAMP, server_default=text('(now())')))
    delete_at: Optional[datetime.datetime] = Field(default=None, sa_column=Column('delete_at', TIMESTAMP))


class Messages(SQLModel, table=True):
    __tablename__ = 'Messages'

    id: str = Field(default_factory=lambda: str(uuid6.uuid7()), sa_column=Column(CHAR(36), primary_key=True))
    conversation_key: Optional[str] = Field(default=None, sa_column=Column('conversation_key', String(128)))
    sender_id: Optional[str] = Field(default=None, sa_column=Column('sender_id', CHAR(36)))
    receiver_id: Optional[str] = Field(default=None, sa_column=Column('receiver_id', CHAR(36)))
    content: Optional[str] = Field(default=None, sa_column=Column('content', Text))
    is_read: Optional[int] = Field(default=None, sa_column=Column('is_read', TINYINT(1), server_default=text("'0'")))
    created_at: Optional[datetime.datetime] = Field(default=None, sa_column=Column('created_at', TIMESTAMP, server_default=text('(now())')))
    delete_at: Optional[datetime.datetime] = Field(default=None, sa_column=Column('delete_at', TIMESTAMP))


class TicketReplies(SQLModel, table=True):
    __tablename__ = 'Ticket_Replies'

    id: str = Field(default_factory=lambda: str(uuid6.uuid7()), sa_column=Column(CHAR(36), primary_key=True))
    ticket_id: Optional[str] = Field(default=None, sa_column=Column('ticket_id', CHAR(36)))
    sender_id: Optional[str] = Field(default=None, sa_column=Column('sender_id', CHAR(36)))
    message: Optional[str] = Field(default=None, sa_column=Column('message', Text))
    is_internal: Optional[int] = Field(default=None, sa_column=Column('is_internal', TINYINT(1), server_default=text("'0'")))
    is_ai_generated: Optional[int] = Field(default=None, sa_column=Column('is_ai_generated', TINYINT(1), server_default=text("'0'")))
    created_at: Optional[datetime.datetime] = Field(default=None, sa_column=Column('created_at', TIMESTAMP, server_default=text('(now())')))
    delete_at: Optional[datetime.datetime] = Field(default=None, sa_column=Column('delete_at', TIMESTAMP))


class Tickets(SQLModel, table=True):
    __tablename__ = 'Tickets'

    id: str = Field(default_factory=lambda: str(uuid6.uuid7()), sa_column=Column(CHAR(36), primary_key=True))
    subject: Optional[str] = Field(default=None, sa_column=Column('subject', String(255)))
    description: Optional[str] = Field(default=None, sa_column=Column('description', Text))
    status: Optional[str] = Field(default=None, sa_column=Column('status', Enum('OPEN', 'IN_PROGRESS', 'RESOLVED', 'CLOSED'), server_default=text("'OPEN'")))
    priority: Optional[str] = Field(default=None, sa_column=Column('priority', Enum('LOW', 'MEDIUM', 'HIGH', 'URGENT'), server_default=text("'MEDIUM'")))
    category: Optional[str] = Field(default=None, sa_column=Column('category', String(100)))
    customer_id: Optional[str] = Field(default=None, sa_column=Column('customer_id', CHAR(36)))
    assigned_agent_id: Optional[str] = Field(default=None, sa_column=Column('assigned_agent_id', CHAR(36)))
    dept_id: Optional[str] = Field(default=None, sa_column=Column('dept_id', CHAR(36)))
    satisfaction_rating: Optional[int] = Field(default=None, sa_column=Column('satisfaction_rating', Integer))
    customer_feedback: Optional[str] = Field(default=None, sa_column=Column('customer_feedback', Text))
    attachment_url: Optional[dict] = Field(default=None, sa_column=Column('attachment_url', JSON))
    due_date: Optional[datetime.datetime] = Field(default=None, sa_column=Column('due_date', TIMESTAMP))
    created_at: Optional[datetime.datetime] = Field(default=None, sa_column=Column('created_at', TIMESTAMP, server_default=text('(now())')))
    delete_at: Optional[datetime.datetime] = Field(default=None, sa_column=Column('delete_at', TIMESTAMP))
