from typing import Any, Dict, Optional
import uuid
from fastapi import Depends
from sqlmodel.ext.asyncio.session import AsyncSession
from src.Domain.base_entities import Accounts
from src.Features.AuthAPI.AccountDTO import SearchAccountRequest
from src.SharedKernel.base.Page import Page
from src.SharedKernel.persistence.CrudRepository import CrudRepository
from src.SharedKernel.persistence.PersistenceManager import get_db_session


class AccountRepository(CrudRepository[Accounts, uuid.UUID]):
    def __init__(self, session: AsyncSession = Depends(get_db_session)):
        super().__init__(Accounts, session)

    async def search_accounts(self, req: SearchAccountRequest):
        limit = req.page_size
        offset = (req.page - 1) * req.page_size

        query = """
        SELECT a.*, COUNT(*) OVER() as total
        FROM Accounts a
        WHERE 1=1
        ORDER BY a.created_at DESC
        LIMIT :limit OFFSET :offset
        """

        exec_result = await self.fetch_all(
            query, {"limit": limit, "offset": offset}
        )
        total = exec_result[0]["total"] if exec_result else 0

        return Page(
            content=exec_result,
            page_number=req.page,
            page_size=req.page_size,
            total_elements=total,
        )

    async def find_by_email(
        self, email: str, exclude_id: str = None
    ) -> Optional[Dict[str, Any]]:
        base_query = """
        SELECT * 
        FROM Accounts a
        WHERE a.email = :email
        """

        params = {"email": email}

        if exclude_id:
            base_query += " AND a.id != :exclude_id"
            params["exclude_id"] = exclude_id

        exec_result = await self.fetch_one(base_query, params)
        return exec_result

    async def find_by_username(
        self, username: str, exclude_id: str = None
    ) -> Optional[Dict[str, Any]]:
        base_query = """
        SELECT * 
        FROM Accounts a
        WHERE a.username = :username 
        """

        params = {
            "username": username,
        }

        if exclude_id:
            base_query += " AND a.id != :exclude_id"
            params["exclude_id"] = exclude_id

        exec_result = await self.fetch_one(base_query, params)
        return exec_result
