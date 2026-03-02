import asyncio
from typing import Dict, List
from fastapi import WebSocket
from SharedKernel.base.Logger import get_logger

logger = get_logger("ConnectionManager")

class SocketManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(
        self,
        websocket: WebSocket, 
        room_id: str, 
    ):
        # await websocket.accept()

        if not self.active_connections.get(room_id):
            self.active_connections[room_id] = []

        self.active_connections[room_id].append(websocket)
        logger.info(f"New Active connections are {self.active_connections}")

    async def disconnect(
        self,
        websocket: WebSocket, 
        room_id: str
    ):
        if room_id in self.active_connections:
            self.active_connections[room_id].remove(websocket)
            if not self.active_connections[room_id]: 
                del self.active_connections[room_id]

        logger.info(f"After disconnect active connections are: {self.active_connections}")

    async def send_personal_message(
        self, 
        websocket: WebSocket, 
        message: str
    ):
        await websocket.send_text(message)
        logger.info(f"Sent a personal msg to: {websocket}")
    
    async def broadcast(
        self, 
        websocket: WebSocket, 
        message: str, 
        room_id: str
    ):
        for connection in self.active_connections[room_id]:
            if connection != websocket:
                await connection.send_text(message)
                logger.info(f"In broadcast: sent msg to {connection}")

manager = SocketManager()