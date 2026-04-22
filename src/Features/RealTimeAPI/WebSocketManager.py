"""
WebSocket Manager - Xử lý real-time chat connections
"""
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Set
import json
import asyncio
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        # Lưu trữ connections theo user_id và conversation_key
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        self.user_conversations: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str, conversation_key: str):
        """Kết nối WebSocket cho user và conversation"""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = {}
            self.user_conversations[user_id] = set()
        
        self.active_connections[user_id][conversation_key] = websocket
        self.user_conversations[user_id].add(conversation_key)
        
        logger.info(f"User {user_id} connected to conversation {conversation_key}")
    
    def disconnect(self, user_id: str, conversation_key: str):
        """Ngắt kết nối WebSocket"""
        if user_id in self.active_connections:
            if conversation_key in self.active_connections[user_id]:
                del self.active_connections[user_id][conversation_key]
            
            if conversation_key in self.user_conversations.get(user_id, set()):
                self.user_conversations[user_id].discard(conversation_key)
            
            # Xóa user nếu không còn conversation nào
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                if user_id in self.user_conversations:
                    del self.user_conversations[user_id]
        
        logger.info(f"User {user_id} disconnected from conversation {conversation_key}")
    
    async def send_personal_message(self, message: str, user_id: str, conversation_key: str):
        """Gửi message đến specific user trong conversation"""
        if (user_id in self.active_connections and 
            conversation_key in self.active_connections[user_id]):
            websocket = self.active_connections[user_id][conversation_key]
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send message to {user_id}/{conversation_key}: {e}")
                self.disconnect(user_id, conversation_key)
    
    async def broadcast_to_conversation(self, message: str, conversation_key: str, exclude_user: str = None):
        """Broadcast message đến tất cả users trong conversation"""
        for user_id, conversations in self.active_connections.items():
            if exclude_user and user_id == exclude_user:
                continue
            
            if conversation_key in conversations:
                websocket = conversations[conversation_key]
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Failed to broadcast to {user_id}/{conversation_key}: {e}")
                    self.disconnect(user_id, conversation_key)

# Global connection manager instance
manager = ConnectionManager()

async def websocket_endpoint(websocket: WebSocket, user_id: str, conversation_key: str = None):
    """
    WebSocket endpoint handler
    URL: /ws/{user_id} hoặc /ws/{user_id}/{conversation_key}
    """
    if not conversation_key:
        conversation_key = f"default_{user_id}"
    
    await manager.connect(websocket, user_id, conversation_key)
    
    try:
        while True:
            # Nhận message từ client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                message_type = message_data.get("type", "message")
                
                if message_type == "message":
                    # Xử lý message thông thường
                    await handle_chat_message(message_data, user_id, conversation_key)
                
                elif message_type == "typing":
                    # Xử lý typing indicator
                    await handle_typing_indicator(message_data, user_id, conversation_key)
                
                elif message_type == "ping":
                    # Heartbeat
                    await manager.send_personal_message(
                        json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                        user_id, 
                        conversation_key
                    )
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON from {user_id}/{conversation_key}: {data}")
                await manager.send_personal_message(
                    json.dumps({"type": "error", "message": "Invalid JSON format"}),
                    user_id,
                    conversation_key
                )
    
    except WebSocketDisconnect:
        manager.disconnect(user_id, conversation_key)
        logger.info(f"WebSocket disconnected: {user_id}/{conversation_key}")
    
    except Exception as e:
        logger.error(f"WebSocket error for {user_id}/{conversation_key}: {e}")
        manager.disconnect(user_id, conversation_key)

async def handle_chat_message(message_data: dict, user_id: str, conversation_key: str):
    """Xử lý chat message"""
    try:
        # Tạo message object
        message = {
            "id": str(uuid.uuid4()),
            "type": "message",
            "sender_id": user_id,
            "conversation_key": conversation_key,
            "content": message_data.get("content", ""),
            "timestamp": datetime.now().isoformat(),
            "metadata": message_data.get("metadata", {})
        }
        
        # Echo lại message cho sender để confirm
        await manager.send_personal_message(
            json.dumps({
                **message,
                "status": "sent",
                "echo": True
            }),
            user_id,
            conversation_key
        )
        
        # Broadcast đến các users khác trong conversation (nếu có)
        await manager.broadcast_to_conversation(
            json.dumps(message),
            conversation_key,
            exclude_user=user_id
        )
        
        logger.info(f"Message handled: {user_id} -> {conversation_key}")
        
    except Exception as e:
        logger.error(f"Failed to handle chat message: {e}")
        await manager.send_personal_message(
            json.dumps({"type": "error", "message": "Failed to process message"}),
            user_id,
            conversation_key
        )

async def handle_typing_indicator(message_data: dict, user_id: str, conversation_key: str):
    """Xử lý typing indicator"""
    try:
        typing_message = {
            "type": "typing",
            "sender_id": user_id,
            "conversation_key": conversation_key,
            "is_typing": message_data.get("is_typing", False),
            "timestamp": datetime.now().isoformat()
        }
        
        # Broadcast typing status đến các users khác
        await manager.broadcast_to_conversation(
            json.dumps(typing_message),
            conversation_key,
            exclude_user=user_id
        )
        
    except Exception as e:
        logger.error(f"Failed to handle typing indicator: {e}")