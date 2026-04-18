from typing import Any, Dict, Optional, List
from fastapi import Depends
from sqlmodel.ext.asyncio.session import AsyncSession
from src.Domain.base_entities import ConversationHistories
from src.SharedKernel.persistence.CrudRepository import CrudRepository
from src.SharedKernel.persistence.PersistenceManager import get_db_session
from src.Features.ConversationAPI.ConversationDTO import ConversationHistoryResponse, SearchConversationHistoriesRequest

class ConversationRepository(CrudRepository[ConversationHistories, str]):
    def __init__(self, session: AsyncSession = Depends(get_db_session)):
        super().__init__(ConversationHistories, session)
    
    # Conversation history operations (sử dụng CrudRepository pattern như AccountRepository)
    async def add_conversation_history(self, session_id: str, role: str, content: str) -> Dict:
        conversation_history = ConversationHistories(
            session_id=session_id,
            role=role,
            content=content
        )
        return await self.save(conversation_history)
    
    async def fetch_conversation_histories(self, session_id: str, request: SearchConversationHistoriesRequest) -> List[ConversationHistoryResponse]:
        limit = request.page_size
        offset = (request.page - 1) * request.page_size
        
        if request.role:
            roles = [r.strip() for r in request.role.split(",")]
            escaped_roles = [f"'{r}'" for r in roles]
            roles_str = ",".join(escaped_roles)
            query = f"""
            SELECT * FROM ConversationHistories
            WHERE session_id = :session_id
            AND role IN ({roles_str})
            ORDER BY timestamp ASC
            LIMIT :limit OFFSET :offset
            """
            result = await self.fetch_all(query, {
                "session_id": session_id,
                "limit": limit,
                "offset": offset
            })
            return [ConversationHistoryResponse(**item) for item in result]
        else:
            query = """
            SELECT * FROM ConversationHistories
            WHERE session_id = :session_id
            ORDER BY timestamp ASC
            LIMIT :limit OFFSET :offset
            """
            result = await self.fetch_all(query, {
                "session_id": session_id,
                "limit": limit,
                "offset": offset
            })
            return [ConversationHistoryResponse(**item) for item in result]
    
    async def delete_conversation_histories(self, session_id: str) -> int:
        query = """
        DELETE FROM ConversationHistories 
        WHERE session_id = :session_id
        """
        result = await self.execute(query, {"session_id": session_id})
        return result["affected_rows"]
    
    async def get_conversation_history_count(self, session_id: str) -> int:
        query = """
        SELECT COUNT(*) as count FROM ConversationHistories 
        WHERE session_id = :session_id
        """
        result = await self.fetch_one(query, {"session_id": session_id})
        return result["count"] if result else 0
    
    async def get_recent_conversation_histories(self, session_id: str, limit: int = 10) -> List[Dict]:
        query = """
        SELECT * FROM ConversationHistories 
        WHERE session_id = :session_id 
        ORDER BY timestamp DESC
        LIMIT :limit
        """
        return await self.fetch_all(query, {
            "session_id": session_id,
            "limit": limit
        })
    
    async def clear_conversation_histories_by_session(self, session_id: str) -> int:
        query = """
        DELETE FROM ConversationHistories 
        WHERE session_id = :session_id
        """
        result = await self.execute(query, {"session_id": session_id})
        return result["affected_rows"]
    
    async def get_unique_sessions(self) -> List[str]:
        query = """
        SELECT session_id 
        FROM ConversationHistories 
        WHERE session_id IS NOT NULL
        GROUP BY session_id
        ORDER BY MAX(timestamp) DESC
        """
        results = await self.fetch_all(query)
        print(results)
        return [row["session_id"] for row in results if row["session_id"]]
