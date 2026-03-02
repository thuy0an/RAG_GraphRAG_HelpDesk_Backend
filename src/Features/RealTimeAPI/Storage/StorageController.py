# import glob
# import mimetypes
# import os
# from typing import List
# from fastapi import APIRouter, Depends, UploadFile, File, status
# from fastapi.responses import FileResponse
# from src.Features.RealTimeAPI.Storage.StorageService import StorageService
# from src.Features.RealTimeAPI.Storage.FileDTO import DeleteFileRequest, FileSearchRequest
# from src.Shared.base import get_logger
# from src.Shared.exception.APIException import APIException
# from src.Shared.base.APIResponse import APIResponse

# storage_prefix = "/api/v1/storage"

# router = APIRouter(
#     prefix=storage_prefix, 
#     tags=["Storage"]
# )

# logger = get_logger(__name__)

# @router.get("/files")
# async def get_all_files(
#     req: FileSearchRequest = Depends(),
#     file_service: StorageService = Depends()
# ):
#     result = await file_service.get_all_files(req)
    
#     return APIResponse(
#         message="Get all files successfully",
#         status_code=status.HTTP_200_OK,
#         data=result
#     )

# @router.get("/files/{id}")
# async def get_file_by_id(
#     id: str,
#     file_service: StorageService = Depends()
# ):
#     result = await file_service.get_file_by_id(id)

#     response = FileResponse(
#         path=result["file_path"],
#         filename=id,
#         media_type=result["mime_type"] or "application/octet-stream", 
#         content_disposition_type="attachment" 
#     )
    
#     response.headers["Access-Control-Allow-Origin"] = "*"
#     response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
#     response.headers["Access-Control-Allow-Headers"] = "*"
    
#     return response

# @router.post("/files/upload")
# async def upload_file(
#     type_storage: str, 
#     files: List[UploadFile] = File(...), 
#     file_service: StorageService = Depends()
# ):
#     await file_service.save_files(type_storage, files)

#     return APIResponse(
#         message="Upload file successfully",
#         status_code=status.HTTP_200_OK,
#     )

# @router.delete("/files/{id}")
# async def delete_file(
#     id: str,
#     type_storage: str,
#     file_service: StorageService = Depends()):

#     req = DeleteFileRequest(file_id=id, type_storage=type_storage)
#     await file_service.delete_file(req)

#     return APIResponse(
#         message="Delete file successfully",
#         status_code=status.HTTP_200_OK,
#         data=None
#     )

# # @router.get("/uploads/{file_name}")
# # def get_pdf(file_name: str):
# #     return FileResponse(
# #         path=f"static/uploads/{file_name}",
# #         media_type="application/pdf",
# #         content_disposition_type="inline"
# #     )
