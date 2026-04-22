"""
Chatroom Controller - Xử lý conversation keys và message history
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
import uuid
from datetime import datetime
import json

from src.Features.LangChainAPI.persistence.MemoryRepository import MemoryRepository

router = APIRouter(prefix="/chatroom", tags=["chatroom"])

# In-memory storage cho conversation keys (production nên dùng Redis/DB)
conversation_store = {}
message_store = {}

@router.get("/conversation_key/{user_id}")
async def get_conversation_key(user_id: str):
    """
    Lấy hoặc tạo conversation key cho user
    """
    try:
        if user_id not in conversation_store:
            # Tạo conversation key mới
            conversation_key = f"conv_{user_id}_{uuid.uuid4().hex[:8]}"
            conversation_store[user_id] = conversation_key
            message_store[conversation_key] = []
        
        return {
            "success": True,
            "data": {
                "conversation_key": conversation_store[user_id],
                "user_id": user_id
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversation key: {str(e)}")

@router.get("/messages/{conversation_key}")
async def get_messages(conversation_key: str, limit: Optional[int] = 50):
    """
    Lấy message history cho conversation
    """
    try:
        if conversation_key not in message_store:
            message_store[conversation_key] = []
        
        messages = message_store[conversation_key]
        
        # Lấy messages gần nhất
        recent_messages = messages[-limit:] if limit else messages
        
        return {
            "success": True,
            "data": {
                "messages": recent_messages,
                "conversation_key": conversation_key,
                "total": len(messages)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")

@router.post("/messages/{conversation_key}")
async def add_message(conversation_key: str, message_data: dict):
    """
    Thêm message vào conversation
    """
    try:
        if conversation_key not in message_store:
            message_store[conversation_key] = []
        
        # Tạo message object
        message = {
            "id": str(uuid.uuid4()),
            "conversation_key": conversation_key,
            "sender_id": message_data.get("sender_id"),
            "content": message_data.get("content"),
            "role": message_data.get("role", "user"),
            "timestamp": datetime.now().isoformat(),
            "metadata": message_data.get("metadata", {})
        }
        
        message_store[conversation_key].append(message)
        
        return {
            "success": True,
            "data": message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")

@router.delete("/messages/{conversation_key}")
async def clear_messages(conversation_key: str):
    """
    Xóa tất cả messages trong conversation
    """
    try:
        if conversation_key in message_store:
            message_store[conversation_key] = []
        
        return {
            "success": True,
            "message": f"Cleared messages for conversation {conversation_key}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear messages: {str(e)}")