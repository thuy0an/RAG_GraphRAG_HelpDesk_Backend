from typing import Dict, List
from fastapi import Depends
from src.Features.ConversationAPI.ConversationRepository import ConversationRepository
from src.Features.ConversationAPI.ConversationDTO import ConversationHistoryResponse, SearchConversationHistoriesRequest

class ConversationService:
    def __init__(self, conversation_repo: ConversationRepository = Depends()):
        self.conversation_repo = conversation_repo

    async def add_conversation_history(self, session_id: str, role: str, content: str) -> Dict:
        """Add a message to conversation history"""
        return await self.conversation_repo.add_conversation_history(session_id, role, content)

    async def get_conversation_histories(self, session_id: str, req: SearchConversationHistoriesRequest) -> List[ConversationHistoryResponse]:
        """Get conversation history for a session, optionally filtered by role"""
        return await self.conversation_repo.fetch_conversation_histories(session_id, req)

    async def get_recent_conversation_histories(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversation context for a session"""
        return await self.conversation_repo.get_recent_conversation_histories(session_id, limit)

    async def clear_conversation_histories_by_session(self, session_id: str) -> bool:
        """Clear conversation history for a session"""
        result = await self.conversation_repo.clear_conversation_histories_by_session(session_id)
        return result

    async def get_all_sessions(self) -> List[str]:
        """Get all unique session IDs"""
        return await self.conversation_repo.get_unique_sessions()
