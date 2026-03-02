# from typing import Optional
# import uuid
# from fastapi import Depends
# from sqlmodel.ext.asyncio.session import AsyncSession
# from src.Shared.base import get_logger
# from src.Features.DeparmentAPI.DepartmentDTO import DepartmentSearchRequest
# from src.Shared.base.Page import Page
# from src.Shared.persistence.QueryExtension import QueryExtension
# from src.Domain.base_entities import Departments
# from src.Shared.persistence.CrudRepository import CrudRepository
# from src.Shared.persistence.Engine import get_async_session

# logger = get_logger(__name__)

# class DepartmentRepository(CrudRepository[Departments, uuid.UUID]):
#     def __init__(self, session: AsyncSession = Depends(get_async_session)):
#         super().__init__(Departments, session)

#     async def search_department(self, req: DepartmentSearchRequest):
#         base_query = """
#         FROM Departments d
#         JOIN Accounts a ON d.id = a.department_id
#         WHERE 1=1
#         AND d.name != 'Chăm sóc khách hàng'
#         """

#         query = (
#             QueryExtension(base_query)
#             .paginate(req.page, req.page_size)
#         )

#         exec_query, params = query.build_select("*")
#         count_query, count_params = query.build_count()

#         exec_result = await self.fetch_all(exec_query, params)
#         count_result = await self.fetch_all(count_query, count_params)

#         return Page(
#             content=exec_result,
#             page_number=req.page,
#             page_size=req.page_size,
#             total_elements=count_result[0]['total']
#         )