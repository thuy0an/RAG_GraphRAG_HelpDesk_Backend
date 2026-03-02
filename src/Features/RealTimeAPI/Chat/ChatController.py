# import json
# from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status
# from typing import Optional
# from pydantic import BaseModel
# from datetime import datetime
# from src.Features.RealTimeAPI.Chat.ChatDTO import MessageRequest
# from src.Features.RealTimeAPI.Chat.ChatService import ChatService
# from src.Features.RealTimeAPI.Chat.ChatRepository import ChatRepository
# from src.Shared.exception import APIException
# from src.Shared.base import get_logger
# from src.Domain.base_entities import Messages
# from src.Shared.base.APIResponse import APIResponse

# service = ChatService()

# logger = get_logger(__name__)

# router = APIRouter(
#     prefix=f"/api/v1/message",
#     tags=["Message"]
# )

# class MessageResponse(BaseModel):
#     id: str
#     room_id: str
#     user_id: str
#     message: str
#     created_at: datetime

# @router.get("/{conversation_key}")
# async def get_messages(
#     conversation_key: str,
#     service: ChatService = Depends()
# ):
#     result = await service.get_messages_by_conversation_key(conversation_key)
#     return APIResponse(
#         message="Message sent successfully",
#         status_code=status.HTTP_200_OK,
#         data=result
#     )

# @router.get("/conversation/{user_id}")
# async def get_conversation_by_user_id(
#     user_id: str,
#     service: ChatService = Depends()
# ):
#     result = await service.get_conversation_by_user_id(user_id)
#     return APIResponse(
#         message="Conversation retrieved successfully",
#         status_code=status.HTTP_200_OK,
#         data=result
#     )

# @router.post("/")
# async def send_message(
#     req: MessageRequest,
#     service: ChatService = Depends()
# ):
#     result = await service.send_message(req)
    
#     return APIResponse(
#         message="Message sent successfully",
#         status_code=status.HTTP_201_CREATED,
#         data=result
#     )

# #
# # Websocket
# #
# @router.websocket("/ws/{user_id}")
# async def websocket_chat_new_conversation(
#     websocket: WebSocket,
#     user_id: str,
#     service: ChatService = Depends()
# ):
#     logger.info(f"Starting new conversation")
#     await service.websocket_chat(websocket, user_id, "None")
#     pass

# @router.websocket("/ws/{user_id}/{conversation_key}")
# async def websocket_chat_with_conversation(
#     websocket: WebSocket,
#     user_id: str,
#     conversation_key: str,
#     service: ChatService = Depends()
# ):
#     logger.info(f"Welcome to conversation {conversation_key}")
#     await service.websocket_chat(websocket, user_id, conversation_key)
#     pass

from fastapi import APIRouter, FastAPI, Request, WebSocket
from lagom import Container
from SharedKernel.socket.SocketManager import SocketManager, manager

class SocketController:
    def __init__(self, app: FastAPI, container: Container) -> None:
        self.app = app

        self.router = APIRouter(
            prefix="/api/v1/socket",
            tags=["Socket"]
        )
        self.manager = container[SocketManager]

        @self.router.websocket("/ws/{room_id}")
        async def websocket_endpoint(websocket: WebSocket, room_id: str):
            await self.manager.connect(websocket, room_id)

        self.app.include_router(self.router)


