# from fastapi import APIRouter, Depends, status
# from src.Features.DeparmentAPI.DepartmentService import DepartmentService
# from src.Features.DeparmentAPI.DepartmentDTO import DepartmentSearchRequest
# from src.Shared.base.APIResponse import APIResponse

# router = APIRouter(
#     prefix="/api/v1/departments",
#     tags=["Departments"]
# )

# @router.get("/")
# async def get_departments(
#     req: DepartmentSearchRequest = Depends(),
#     service: DepartmentService = Depends()
# ):
#     result = await service.search_departments(req)
#     return APIResponse(
#         message="Get departments",
#         status_code=status.HTTP_200_OK,
#         data=result
#     )