import json
from typing import Optional
from src.Domain.base_entities import Messages
from src.Shared.exception import APIException
from src.Features.RealTimeAPI.Chat.ChatRepository import ChatRepository
from fastapi import Depends, WebSocket, WebSocketDisconnect, status
from src.Shared.base import get_logger
from src.Shared.Utils import Utils
from src.Shared.ConnectionManager import manager

logger = get_logger(__name__)

class ChatService:
    def __init__(self, repo: ChatRepository = Depends()):
        self.repo = repo
        pass
    
    async def get_messages_by_conversation_key(self, conversation_key: str):
        return await self.repo.find_message_by_conversation_key(conversation_key)   
    
    async def get_conversation_by_user_id(self, user_id: str):
        return await self.repo.find_conversation_by_user_id(user_id)

    async def websocket_chat(self, 
        websocket: WebSocket, 
        user_id: str, 
        conversation_key: Optional[str] = None, 
    ):
        customer_care_agent_id = None

        # is_agent = await self.repo.fetch_one(
        #     """
        #     SELECT * 
        #     FROM Accounts a
        #     WHERE a.id = :user_id 
        #     AND ROLE = :role
        #     """,
        #     {"role": "AGENT", "user_id": user_id} 
        # )

        is_agent = await self.repo.fetch_one(
            """
            SELECT a.* 
            FROM Accounts a
            WHERE a.id = :user_id
            AND a.role = 'AGENT'
            """,
            {"user_id": user_id}
        )

        if not is_agent:
            customer_care_agent = await self.repo.fetch_one(
                """
                SELECT a.* 
                FROM Accounts a
                JOIN Departments d ON a.department_id = d.id
                AND a.role = 'AGENT'
                AND d.name = :dept_name
                """,
                {"dept_name": "Chăm sóc khách hàng"}
            )

            if customer_care_agent is None:
                return APIException(
                    message="Ko tìm thấy nhân viên cskh",
                    status=status.HTTP_404_NOT_FOUND
                )

            customer_care_agent_id = customer_care_agent['id']

        if conversation_key == "None" and customer_care_agent_id is not None:
            conversation_key = Utils.generate_conversation_key(user_id, customer_care_agent_id)
            logger.info(f"Generated conversation key: {conversation_key}")

        await websocket.accept() 
        try:
            await manager.connect(websocket, conversation_key)

            while True:
                raw_message = await websocket.receive_text()
                print(raw_message)

                ws_data = json.loads(raw_message)
                logger.info(f"Received message: {ws_data}")
                if ws_data.get('type') == "message" or ws_data.get('type') == "file":
                    chat = None
                    if is_agent:
                        user_id_in_conversation = Utils.extract_customer_id_from_conversation_key(conversation_key, customer_care_agent_id)
                        chat = Messages(
                            conversation_key=conversation_key,
                            sender_id=ws_data.get('sender_id'), 
                            content=ws_data.get('content'),
                            receiver_id=user_id_in_conversation
                        )
                    else:
                        chat = Messages(
                            conversation_key=conversation_key,
                            sender_id=ws_data.get('sender_id'), 
                            content=ws_data.get('content'),
                            receiver_id=customer_care_agent_id
                        )
                    
                    if chat:
                        await self.repo.save(chat)
                        logger.info("Save user chat")  

                    response = { "sender_id": ws_data.get('sender_id'), "content": ws_data.get('content') }
                    json_res = json.dumps(response)

                    # if ws_data.get('type') == "file":
                    #     await manager.send_personal_message(websocket, json_res)

                    await manager.broadcast(websocket, json_res, conversation_key) 
                if ws_data.get('type') == "typing":
                    data = json.dumps(ws_data)
                    await manager.broadcast(websocket, data, conversation_key) 
        except WebSocketDisconnect:
            await manager.disconnect(websocket, conversation_key)
            response = { "content": f"User {user_id} left the chat" }
            json_res = json.dumps(response)
            # await manager.broadcast(websocket, json_res, conversation_key)

        