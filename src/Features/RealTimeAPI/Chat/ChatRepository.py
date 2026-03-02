import uuid
from fastapi import Depends
from sqlmodel.ext.asyncio.session import AsyncSession
from src.Shared.Utils import Utils
from src.Domain.base_entities import Messages
from src.Shared.persistence.Engine import get_async_session
from src.Shared.persistence.CrudRepository import CrudRepository
from src.Shared.base import get_logger
from src.Features.RealTimeAPI.Chat.ChatDTO import MessageRequest

logger = get_logger(__name__)

class ChatRepository(CrudRepository[Messages, uuid.UUID]):
    def __init__(self, session: AsyncSession = Depends(get_async_session)):
        super().__init__(Messages, session)
    
    async def find_message_by_conversation_key(self, conversation_key: str):
        return await self.fetch_all(
            "SELECT * FROM Messages m WHERE m.conversation_key = :key",
            {"key": conversation_key}
        )
    
    async def find_conversation_by_user_id(self, user_id: str):
        return await self.fetch_all(
            """
            SELECT DISTINCT conversation_key
            FROM Messages
            WHERE conversation_key LIKE :user_id
            ORDER BY conversation_key;
            """,
            {"user_id": f"%{user_id}%"}
        )

    async def send_message(self, req: MessageRequest):
        message = Messages(
            sender_id=req.user_id, 
            content=req.content,
            receiver_id=req.receiver_id
        ) 
        return await self.save(message)


