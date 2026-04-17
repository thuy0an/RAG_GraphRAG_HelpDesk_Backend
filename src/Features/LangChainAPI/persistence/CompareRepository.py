import asyncio
import json
from datetime import datetime, timezone
import uuid6
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel

from src.Domain.compare_entities import CompareRun


class CompareRepository:
    def __init__(self, db_path: str = "specs/data/compare_runs.db"):
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

    def _ensure_dir(self) -> None:
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
                    tables=[CompareRun.__table__],
                )

            self._initialized = True

    async def create_run(
        self,
        session_id: str,
        file_name: str,
        file_type: Optional[str],
        file_size: Optional[int],
        pac_ingest: Dict,
        graphrag_ingest: Dict,
        errors: Optional[Dict] = None,
    ) -> Dict:
        await self._ensure_initialized()

        run_id = str(uuid6.uuid7())
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        query = sa_text(
            """
            INSERT INTO compare_runs
            (id, session_id, file_name, file_type, file_size,
             pac_ingest_json, graphrag_ingest_json, errors_json, created_at)
            VALUES
            (:id, :session_id, :file_name, :file_type, :file_size,
             :pac_ingest, :graphrag_ingest, :errors_json, :created_at)
            """
        )

        async with self.sqlite_engine.begin() as conn:
            await conn.execute(
                query,
                {
                    "id": run_id,
                    "session_id": session_id,
                    "file_name": file_name,
                    "file_type": file_type,
                    "file_size": file_size,
                    "pac_ingest": json.dumps(pac_ingest or {}),
                    "graphrag_ingest": json.dumps(graphrag_ingest or {}),
                    "errors_json": json.dumps(errors or {}) if errors else None,
                    "created_at": created_at,
                },
            )

        return await self.get_run(run_id)

    async def update_query_metrics(
        self,
        run_id: str,
        pac_query: Optional[Dict],
        graphrag_query: Optional[Dict],
    ) -> Optional[Dict]:
        await self._ensure_initialized()
        query = sa_text(
            """
            UPDATE compare_runs
            SET pac_query_json = :pac_query,
                graphrag_query_json = :graphrag_query
            WHERE id = :run_id
            """
        )

        async with self.sqlite_engine.begin() as conn:
            await conn.execute(
                query,
                {
                    "run_id": run_id,
                    "pac_query": json.dumps(pac_query or {}),
                    "graphrag_query": json.dumps(graphrag_query or {}),
                },
            )

        return await self.get_run(run_id)

    async def list_runs(self, session_id: str) -> List[Dict]:
        await self._ensure_initialized()
        query = sa_text(
            """
            SELECT *
            FROM compare_runs
            WHERE session_id = :session_id
            ORDER BY created_at DESC
            """
        )

        async with self.sqlite_engine.connect() as conn:
            result = await conn.execute(query, {"session_id": session_id})
            rows = result.mappings().all()
            return [self._serialize_row(row) for row in rows]

    async def get_run(self, run_id: str) -> Optional[Dict]:
        await self._ensure_initialized()
        query = sa_text(
            """
            SELECT *
            FROM compare_runs
            WHERE id = :run_id
            LIMIT 1
            """
        )

        async with self.sqlite_engine.connect() as conn:
            result = await conn.execute(query, {"run_id": run_id})
            row = result.mappings().first()
            if not row:
                return None
            return self._serialize_row(row)

    async def delete_run(self, run_id: str) -> int:
        await self._ensure_initialized()
        query = sa_text(
            """
            DELETE FROM compare_runs
            WHERE id = :run_id
            """
        )

        async with self.sqlite_engine.begin() as conn:
            result = await conn.execute(query, {"run_id": run_id})
            return result.rowcount

    def _deserialize_query_json(self, raw: Optional[str]) -> Optional[dict]:
        """Deserialize query JSON với backward compatibility cho các trường mới."""
        if not raw:
            return None
        data = json.loads(raw)
        # Backward compatibility: đảm bảo các trường mới có giá trị mặc định
        data.setdefault("relevance_score", None)
        data.setdefault("source_coverage", None)
        data.setdefault("word_count", None)
        data.setdefault("doc_passages", [])
        data.setdefault("retrieved_chunks", [])
        return data

    def _serialize_run(self, run: CompareRun) -> Dict:
        return {
            "id": run.id,
            "session_id": run.session_id,
            "file_name": run.file_name,
            "file_type": run.file_type,
            "file_size": run.file_size,
            "pac_ingest": json.loads(run.pac_ingest_json or "{}"),
            "graphrag_ingest": json.loads(run.graphrag_ingest_json or "{}"),
            "pac_query": self._deserialize_query_json(run.pac_query_json),
            "graphrag_query": self._deserialize_query_json(run.graphrag_query_json),
            "errors": json.loads(run.errors_json or "{}") if run.errors_json else None,
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }

    def _serialize_row(self, row: Dict) -> Dict:
        created_at = row.get("created_at")
        if isinstance(created_at, str):
            created_at_value = created_at
        elif created_at:
            created_at_value = created_at.isoformat()
        else:
            created_at_value = None

        return {
            "id": row.get("id"),
            "session_id": row.get("session_id"),
            "file_name": row.get("file_name"),
            "file_type": row.get("file_type"),
            "file_size": row.get("file_size"),
            "pac_ingest": json.loads(row.get("pac_ingest_json") or "{}"),
            "graphrag_ingest": json.loads(row.get("graphrag_ingest_json") or "{}"),
            "pac_query": self._deserialize_query_json(row.get("pac_query_json")),
            "graphrag_query": self._deserialize_query_json(row.get("graphrag_query_json")),
            "errors": json.loads(row.get("errors_json") or "{}") if row.get("errors_json") else None,
            "created_at": created_at_value,
        }
