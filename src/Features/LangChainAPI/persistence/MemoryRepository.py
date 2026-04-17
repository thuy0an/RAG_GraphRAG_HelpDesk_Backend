"""
MemoryRepository - Repository cho conversation_history với schema mới.

Schema: mỗi row = 1 lượt hỏi-đáp
  - user_content    : câu hỏi của user
  - rag_content     : trả lời PaCRAG (upsert độc lập, có thể NULL)
  - graphrag_content: trả lời GraphRAG (upsert độc lập, có thể NULL)

Flow:
  1. begin_turn(session_id, user_content) → tạo row mới, trả về turn_id
  2. update_rag(turn_id, answer)          → cập nhật rag_content
  3. update_graphrag(turn_id, answer)     → cập nhật graphrag_content
  Bước 2 và 3 hoàn toàn độc lập, bất đồng bộ.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlmodel import SQLModel
from sqlalchemy import text as sa_text
import uuid6

from src.Domain.history_entities import ConversationHistory


class MemoryRepository:
    def __init__(self, db_path: str = "specs/data/chat_history.db"):
        self.db_path = Path(db_path).resolve()
        self._ensure_dir()
        self._sqlite_engine: AsyncEngine = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def sqlite_engine(self) -> AsyncEngine:
        if self._sqlite_engine is None:
            self._sqlite_engine = create_async_engine(
                f"sqlite+aiosqlite:///{self.db_path}"
            )
        return self._sqlite_engine

    def _ensure_dir(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with self.sqlite_engine.begin() as conn:
                await conn.run_sync(
                    SQLModel.metadata.create_all,
                    tables=[ConversationHistory.__table__]
                )
            self._initialized = True

    # =========================================================
    # CORE API
    # =========================================================

    async def begin_turn(self, session_id: str, user_content: str) -> str:
        """
        Tạo một row mới cho lượt hỏi-đáp.
        Trả về turn_id để dùng cho update_rag / update_graphrag.
        """
        await self._ensure_initialized()
        turn_id = str(uuid6.uuid7())
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        async with self.sqlite_engine.begin() as conn:
            await conn.execute(
                sa_text("""
                    INSERT INTO conversation_history
                        (id, session_id, user_content, rag_content, graphrag_content, timestamp)
                    VALUES
                        (:id, :session_id, :user_content, NULL, NULL, :timestamp)
                """),
                {
                    "id": turn_id,
                    "session_id": session_id,
                    "user_content": user_content,
                    "timestamp": now,
                }
            )
        return turn_id

    async def update_rag(self, turn_id: str, answer: str) -> None:
        """Cập nhật câu trả lời PaCRAG cho một lượt hỏi."""
        await self._ensure_initialized()
        async with self.sqlite_engine.begin() as conn:
            await conn.execute(
                sa_text("""
                    UPDATE conversation_history
                    SET rag_content = :answer
                    WHERE id = :turn_id
                """),
                {"answer": answer, "turn_id": turn_id}
            )

    async def update_graphrag(self, turn_id: str, answer: str) -> None:
        """Cập nhật câu trả lời GraphRAG cho một lượt hỏi."""
        await self._ensure_initialized()
        async with self.sqlite_engine.begin() as conn:
            await conn.execute(
                sa_text("""
                    UPDATE conversation_history
                    SET graphrag_content = :answer
                    WHERE id = :turn_id
                """),
                {"answer": answer, "turn_id": turn_id}
            )

    # =========================================================
    # QUERY API
    # =========================================================

    async def get_history_all(
        self,
        session_id: str,
        role_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Lấy toàn bộ lịch sử của session.

        role_filter (tương thích ngược):
          - None / 'user'           → trả về tất cả các turn
          - 'assistant_rag'         → trả về turn có rag_content
          - 'assistant_graphrag'    → trả về turn có graphrag_content
        """
        await self._ensure_initialized()

        if role_filter == "assistant_rag":
            query = sa_text("""
                SELECT id, session_id, user_content, rag_content, graphrag_content, timestamp
                FROM conversation_history
                WHERE session_id = :session_id
                  AND rag_content IS NOT NULL
                ORDER BY timestamp ASC
            """)
        elif role_filter == "assistant_graphrag":
            query = sa_text("""
                SELECT id, session_id, user_content, rag_content, graphrag_content, timestamp
                FROM conversation_history
                WHERE session_id = :session_id
                  AND graphrag_content IS NOT NULL
                ORDER BY timestamp ASC
            """)
        else:
            query = sa_text("""
                SELECT id, session_id, user_content, rag_content, graphrag_content, timestamp
                FROM conversation_history
                WHERE session_id = :session_id
                ORDER BY timestamp ASC
            """)

        async with self.sqlite_engine.connect() as conn:
            result = await conn.execute(query, {"session_id": session_id})
            rows = [dict(row) for row in result.mappings().all()]

        # Chuyển sang format list message để frontend dễ dùng
        return self._rows_to_messages(rows, role_filter)

    def _rows_to_messages(
        self,
        rows: List[Dict],
        role_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Chuyển mỗi row thành danh sách message objects.
        Mỗi turn sinh ra tối đa 3 message: user, rag, graphrag.
        """
        messages = []
        for row in rows:
            turn_id = row["id"]
            timestamp = row["timestamp"]

            # User message luôn có
            messages.append({
                "id": f"{turn_id}_user",
                "turn_id": turn_id,
                "session_id": row["session_id"],
                "role": "user",
                "content": row["user_content"],
                "timestamp": str(timestamp) if timestamp else None,
            })

            # RAG answer — chỉ thêm nếu có và không filter graphrag
            if row.get("rag_content") and role_filter != "assistant_graphrag":
                messages.append({
                    "id": f"{turn_id}_rag",
                    "turn_id": turn_id,
                    "session_id": row["session_id"],
                    "role": "assistant_rag",
                    "content": row["rag_content"],
                    "timestamp": str(timestamp) if timestamp else None,
                })

            # GraphRAG answer — chỉ thêm nếu có và không filter rag
            if row.get("graphrag_content") and role_filter != "assistant_rag":
                messages.append({
                    "id": f"{turn_id}_graphrag",
                    "turn_id": turn_id,
                    "session_id": row["session_id"],
                    "role": "assistant_graphrag",
                    "content": row["graphrag_content"],
                    "timestamp": str(timestamp) if timestamp else None,
                })

        return messages

    async def get_recent_messages(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """Lấy N turn gần nhất (mỗi turn = 1 row)."""
        await self._ensure_initialized()
        async with self.sqlite_engine.connect() as conn:
            result = await conn.execute(
                sa_text("""
                    SELECT id, session_id, user_content, rag_content, graphrag_content, timestamp
                    FROM conversation_history
                    WHERE session_id = :session_id
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """),
                {"session_id": session_id, "limit": limit}
            )
            rows = [dict(row) for row in result.mappings().all()]
        rows.reverse()
        return self._rows_to_messages(rows)

    async def delete_session_history(self, session_id: str) -> int:
        """Xóa toàn bộ lịch sử của session. Trả về số row đã xóa."""
        await self._ensure_initialized()
        async with self.sqlite_engine.begin() as conn:
            result = await conn.execute(
                sa_text("DELETE FROM conversation_history WHERE session_id = :session_id"),
                {"session_id": session_id}
            )
            return result.rowcount

    async def close(self) -> None:
        await self.sqlite_engine.dispose()
