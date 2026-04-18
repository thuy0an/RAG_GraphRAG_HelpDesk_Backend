import json
from datetime import datetime, timezone
import uuid6
from typing import Dict, List, Optional
from fastapi import Depends
from sqlmodel.ext.asyncio.session import AsyncSession

from src.Domain.base_entities import CompareRuns
from src.SharedKernel.persistence.CrudRepository import CrudRepository
from src.SharedKernel.persistence.PersistenceManager import get_db_session


class CompareRepository(CrudRepository[CompareRuns, str]):
    def __init__(self, session: AsyncSession = Depends(get_db_session)):
        super().__init__(CompareRuns, session)

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
        run_id = str(uuid6.uuid7())
        
        compare_run = CompareRuns(
            id=run_id,
            session_id=session_id,
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            pac_ingest_json=json.dumps(pac_ingest or {}),
            graphrag_ingest_json=json.dumps(graphrag_ingest or {}),
            errors_json=json.dumps(errors or {}) if errors else None,
        )
        
        result = await self.save(compare_run)
        return await self.get_run(run_id)

    async def update_query_metrics(
        self,
        run_id: str,
        pac_query: Optional[Dict],
        graphrag_query: Optional[Dict],
    ) -> Optional[Dict]:
        query = """
        UPDATE CompareRuns
        SET pac_query_json = :pac_query,
            graphrag_query_json = :graphrag_query
        WHERE id = :run_id
        """
        
        await self.execute(query, {
            "run_id": run_id,
            "pac_query": json.dumps(pac_query or {}),
            "graphrag_query": json.dumps(graphrag_query or {}),
        })
        
        return await self.get_run(run_id)

    async def list_runs(self, session_id: str) -> List[Dict]:
        query = """
        SELECT * FROM CompareRuns
        WHERE session_id = :session_id
        ORDER BY created_at DESC
        """
        
        results = await self.fetch_all(query, {"session_id": session_id})
        return [self._serialize_row(row) for row in results]

    async def get_run(self, run_id: str) -> Optional[Dict]:
        query = """
        SELECT * FROM CompareRuns
        WHERE id = :run_id
        LIMIT 1
        """
        
        result = await self.fetch_one(query, {"run_id": run_id})
        if not result:
            return None
        return self._serialize_row(result)

    async def delete_run(self, run_id: str) -> int:
        query = """
        DELETE FROM CompareRuns
        WHERE id = :run_id
        """

        result = await self.execute(query, {"run_id": run_id})
        return result["affected_rows"]

    def _serialize_row(self, row: Dict) -> Dict:
        created_at = row.get("created_at")
        if isinstance(created_at, str):
            created_at_value = created_at
        elif created_at:
            created_at_value = created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at)
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
            "pac_query": json.loads(row.get("pac_query_json") or "{}") if row.get("pac_query_json") else None,
            "graphrag_query": json.loads(row.get("graphrag_query_json") or "{}") if row.get("graphrag_query_json") else None,
            "errors": json.loads(row.get("errors_json") or "{}") if row.get("errors_json") else None,
            "created_at": created_at_value,
        }
