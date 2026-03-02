# from fastapi import Depends
# from src.Features.DeparmentAPI.DeparmentRepository import DepartmentRepository
# from src.Features.DeparmentAPI.DepartmentDTO import DepartmentSearchRequest
# from src.Shared.exception.APIException import APIException
# from starlette import status
# from src.Domain.base_entities import Departments

# class DepartmentService:
#     def __init__(self, repo: DepartmentRepository = Depends()):
#         self.repo = repo

#     async def search_departments(self, req: DepartmentSearchRequest):
#         return await self.repo.search_department(req)