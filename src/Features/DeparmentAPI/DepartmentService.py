from fastapi import Depends
from src.Features.DeparmentAPI.DeparmentRepository import DepartmentRepository
from src.Features.DeparmentAPI.DepartmentDTO import DepartmentSearchRequest

class DepartmentService:
    def __init__(self, repo: DepartmentRepository = Depends()):
        self.repo = repo

    async def search_departments(self, req: DepartmentSearchRequest):
        return await self.repo.search_department(req)