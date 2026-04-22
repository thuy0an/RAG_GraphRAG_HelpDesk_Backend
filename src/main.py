from SharedKernel.base.WebApplication import WebApplication
from src.Features.RealTimeAPI.WebSocketManager import websocket_endpoint
from src.Features.RealTimeAPI.ChatroomController import router as chatroom_router

app = WebApplication()

# Đăng ký WebSocket endpoints
@app.websocket("/ws/{user_id}")
async def websocket_user_endpoint(websocket, user_id: str):
    await websocket_endpoint(websocket, user_id)

@app.websocket("/ws/{user_id}/{conversation_key}")
async def websocket_conversation_endpoint(websocket, user_id: str, conversation_key: str):
    await websocket_endpoint(websocket, user_id, conversation_key)

# Đăng ký Chatroom API routes
app.include_router(chatroom_router, prefix="/api/v1")

app.map_controller()
