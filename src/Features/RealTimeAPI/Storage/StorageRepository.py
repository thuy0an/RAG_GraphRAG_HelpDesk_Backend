# import uuid
# from fastapi import Depends
# from sqlalchemy.ext.asyncio import AsyncSession
# from src.Features.RealTimeAPI.Storage.FileDTO import FileSearchRequest
# from src.Shared.base.Page import Page
# from src.Shared.persistence.QueryExtension import QueryExtension
# from src.Shared.persistence.Engine import get_async_session
# from src.Domain.base_entities import Attachment
# from src.Shared.persistence.CrudRepository import CrudRepository

# class FileRepository(CrudRepository[Attachment, uuid.UUID]):
#     def __init__(self, session: AsyncSession = Depends(get_async_session)):
#         super().__init__(Attachment, session)

#     async def search_files(self, req: FileSearchRequest):
#         base_query = """
#         FROM Attachment att
#         WHERE 1=1
#         AND att.delete_at IS NULL
#         """

#         query = (
#             QueryExtension(base_query)
#             .paginate(req.page, req.page_size)
#         )

#         exec_query, params = query.build_select("att.*")
#         count_query, count_params = query.build_count()

#         exec_result = await self.fetch_all(exec_query, params)
#         count_result = await self.fetch_all(count_query, count_params)

#         return Page(
#             content=exec_result,
#             page_number=req.page,
#             page_size=req.page_size,
#             total_elements=count_result[0]['total']
#         )