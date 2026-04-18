import uuid
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.Features.RealTimeAPI.FileSystem.FileDTO import FileSearchRequest
from src.Domain.base_entities import Attachments
from src.SharedKernel.base.Page import Page
from src.SharedKernel.persistence.CrudRepository import CrudRepository
from src.SharedKernel.persistence.Decorators import Repository
from src.SharedKernel.persistence.PersistenceManager import get_db_session
from src.SharedKernel.utils.yamlenv import load_env_yaml
from sqlalchemy.ext.asyncio import AsyncSession

class FileRepository(CrudRepository[Attachments, uuid.UUID]):
    def __init__(self, session: AsyncSession = Depends(get_db_session)):
        super().__init__(Attachments, session)

    async def search_files(self, req: FileSearchRequest):
        offset = (req.page - 1) * req.page_size

        query = """
        SELECT att.* FROM Attachments att
        WHERE att.delete_at IS NULL
        ORDER BY att.created_at DESC
        LIMIT :limit OFFSET :offset
        """

        count_query = """
        SELECT COUNT(*) as total FROM Attachments att
        WHERE att.delete_at IS NULL
        """

        exec_result = await self.fetch_all(query, {"limit": req.page_size, "offset": offset})
        count_result = await self.fetch_one(count_query, {})

        total = count_result['total'] if count_result else 0

        return Page(
            content=exec_result,
            page_number=req.page,
            page_size=req.page_size,
            total_elements=count_result['total']
        )

    async def search_files_by_name(self, file_name: str):
        query = """
        SELECT att.* FROM Attachments att
        WHERE att.delete_at IS NULL
        AND LOWER(att.file_name) LIKE LOWER(:file_name)
        ORDER BY att.created_at DESC
        LIMIT 1
        """
        
        exec_result = await self.fetch_one(query, {"file_name": f"%{file_name}%"})
        
        return exec_result

    async def find_by_filename(self, filename: str):
        """Tìm file theo tên chính xác (chưa bị delete)"""
        query = """
        SELECT * FROM Attachments
        WHERE file_name = :filename 
        AND delete_at IS NULL
        ORDER BY created_at DESC
        LIMIT 1
        """
        return await self.fetch_one(query, {"filename": filename})

    async def soft_delete_by_filename(self, filename: str):
        """Soft delete file theo tên"""
        query = """
        UPDATE Attachments
        SET delete_at = NOW() 
        WHERE file_name = :filename 
        AND delete_at IS NULL
        """
        await self.execute(query, {"filename": filename})