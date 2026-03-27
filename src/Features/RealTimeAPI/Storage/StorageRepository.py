import uuid
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.Features.RealTimeAPI.Storage.FileDTO import FileSearchRequest
from src.Domain.base_entities import Attachment
from src.SharedKernel.base.Page import Page
from src.SharedKernel.persistence.CrudRepository import CrudRepository
from src.SharedKernel.persistence.Decorators import Repository
from src.SharedKernel.persistence.PersistenceManager import get_db_session
from src.SharedKernel.persistence.QueryExtension import QueryExtension
from src.SharedKernel.utils.yamlenv import load_env_yaml
from sqlalchemy.ext.asyncio import AsyncSession

@Repository
class FileRepository(CrudRepository[Attachment, uuid.UUID]):
    def __init__(self, session: AsyncSession = Depends(get_db_session)):
        super().__init__(Attachment, session)

    async def search_files(self, req: FileSearchRequest):
        base_query = """
        FROM Attachment att
        WHERE 1=1
        AND att.delete_at IS NULL
        """

        query = (
            QueryExtension(base_query)
            .paginate(req.page, req.page_size)
        )

        exec_query, params = query.build_select("att.*")
        count_query, count_params = query.build_count()

        exec_result = await self.fetch_all(exec_query, params)
        count_result = await self.fetch_all(count_query, count_params)

        return Page(
            content=exec_result,
            page_number=req.page,
            page_size=req.page_size,
            total_elements=count_result[0]['total']
        )

    async def search_files_by_name(self, file_name: str):
        base_query = """
        FROM Attachment att
        WHERE 1=1
        AND att.delete_at IS NULL
        """
        
        query = (
            QueryExtension(base_query)
            .filter(
                file_name, 
                "LOWER(att.file_name) LIKE LOWER(:file_name)", 
                file_name=f"%{file_name}%"
            )
        )
        
        exec_query, params = query.build_select("att.*")
        exec_result = await self.fetch_one(exec_query, params)
        
        return exec_result

    async def find_by_filename(self, filename: str):
        """Tìm file theo tên chính xác (chưa bị delete)"""
        query = """
        SELECT * FROM Attachment 
        WHERE file_name = :filename 
        AND delete_at IS NULL
        ORDER BY created_at DESC
        LIMIT 1
        """
        return await self.fetch_one(query, {"filename": filename})

    async def soft_delete_by_filename(self, filename: str):
        """Soft delete file theo tên"""
        query = """
        UPDATE Attachment 
        SET delete_at = NOW() 
        WHERE file_name = :filename 
        AND delete_at IS NULL
        """
        await self.execute(query, {"filename": filename})
    

