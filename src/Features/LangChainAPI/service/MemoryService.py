# """
# MemoryService - Business logic for chat memory
# KISS: Delegates all DB operations to MemoryRepo
# """

# import asyncio
# from typing import Dict, List
# from pathlib import Path

# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_community.chat_message_histories import SQLChatMessageHistory
# from sqlalchemy.ext.asyncio import create_async_engine
# from Features.LangChainAPI.LangChainDTO import ChatMessageRequest
# from src.Features.LangChainAPI.persistence.MemoryRepository import MemoryRepository

# memory_store: dict[str, InMemoryChatMessageHistory] = {}


# class MemoryService:
#     """
#     Service layer for chat memory business logic.
#     Uses MemoryRepo for all database operations.
#     """

#     def __init__(self, provider) -> None:
#         self.provider = provider
#         self.memory_repo = MemoryRepository(".data/chat_history.db")

#     async def long_chat(self, req: ChatMessageRequest):
#         """Main chat flow with history"""
#         await self.memory_repo.init_db()

#         # Save user message
#         await self.memory_repo.add_message(req.session_id, "user", req.message)

#         # Get recent history for context
#         history_data = await self.memory_repo.get_recent_messages(
#             req.session_id,
#             limit=10
#         )

#         # Format for prompt
#         formatted_history = self._format_history(history_data)

#         # Build prompt with history
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", f"Bạn là một trợ lý ảo hữu ích. Lịch sử trò chuyện:\n{formatted_history}"),
#             ("human", "{input}")
#         ])

#         chain = prompt | self.provider
#         stream = chain.astream({"input": req.message})

#         async def response_generator():
#             ai_response = ""
#             async for event in stream:
#                 ai_response += event.content
#                 yield event.content
#             # Save AI response
#             await self.memory_repo.add_message(req.session_id, "ai", ai_response)

#         return response_generator()

#     def _format_history(self, history: List[Dict]) -> str:
#         """Format history for prompt inclusion"""
#         if not history:
#             return "Không có lịch sử trò chuyện."

#         formatted = []
#         for msg in history:
#             role = "Người dùng" if msg.get('role') == 'user' else "AI"
#             formatted.append(f"{role}: {msg.get('content', '')}")

#         return "\n".join(formatted)

#     # ============================================================
#     # Legacy compatibility methods (using SQLChatMessageHistory)
#     # ============================================================

#     async def _save_messages(self, session_id: str, user_msg: str, ai_msg: str):
#         """Legacy: Save messages using SQLChatMessageHistory"""
#         history = await asyncio.to_thread(self._get_session, session_id)

#         human_msg = HumanMessage(content=user_msg)
#         ai_msg_obj = AIMessage(content=ai_msg)

#         await history.aadd_message(human_msg)
#         await history.aadd_message(ai_msg_obj)

#     async def get_chat_history(self, session_id: str, limit: int = None):
#         """Legacy: Get formatted chat history"""
#         try:
#             chat_history = await asyncio.to_thread(self._get_session, session_id)
#             messages = await chat_history.aget_messages()

#             if limit and limit > 0:
#                 messages = messages[-limit:]

#             formatted_messages = []
#             for msg in messages:
#                 formatted_messages.append({
#                     "role": "user" if msg.type == "human" else "assistant",
#                     "content": msg.content
#                 })

#             return {
#                 "session_id": session_id,
#                 "messages": formatted_messages,
#                 "total_count": len(messages)
#             }

#         except Exception as e:
#             print(f"Error getting chat history: {e}")
#             return {
#                 "session_id": session_id,
#                 "messages": [],
#                 "total_count": 0,
#                 "error": str(e)
#             }

#     def _get_session(self, session_id: str):
#         """Get SQLChatMessageHistory session (legacy)"""
#         return SQLChatMessageHistory(
#             session_id=session_id,
#             connection=self._legacy_engine
#         )

#     # ============================================================
#     # New direct repository methods
#     # ============================================================

#     async def get_history_paginated(self, session_id: str, page: int = 1, size: int = 5):
#         """Get paginated history directly from repo"""
#         return await self.memory_repo.get_history_paginated(session_id, page, size)

#     async def get_history_all(self, session_id: str) -> List[Dict]:
#         """Get all history directly from repo"""
#         return await self.memory_repo.get_history_all(session_id)

#     async def add_message_direct(self, session_id: str, role: str, content: str):
#         """Add message directly through repo"""
#         await self.memory_repo.add_message(session_id, role, content)

#     async def clear_session(self, session_id: str) -> int:
#         """Clear all history for a session"""
#         return await self.memory_repo.delete_session_history(session_id)

#     # ============================================================
#     # In-memory fallback (for quick sessions)
#     # ============================================================

#     def get_inmem_history(self, session_id: str):
#         """Get or create in-memory history"""
#         if session_id not in memory_store:
#             memory_store[session_id] = InMemoryChatMessageHistory()
#         return memory_store[session_id]
