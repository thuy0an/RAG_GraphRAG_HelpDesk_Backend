# import datetime
# import glob
# import mimetypes
# import os
# from typing import List
# import uuid6
# from fastapi import Depends, UploadFile, status
# from src.Shared.base import get_logger
# from src.Features.RealTimeAPI.Storage.FileDTO import DeleteFileRequest, FileSearchRequest
# from src.Domain.base_entities import Attachment
# from src.Features.RealTimeAPI.Storage.StorageRepository import FileRepository
# from src.Shared.exception.APIException import APIException
# from dotenv import load_dotenv
# load_dotenv()

# logger = get_logger(__name__)

# class StorageService:
#     USER_STORAGE_DIR = "static/user_storage"
#     SYSTEM_DIR = "static/system_storage"
#     STORAGE_PREFIX = "api/v1/storage/files" 

#     def __init__(self, repo: FileRepository = Depends()):
#         os.makedirs(self.USER_STORAGE_DIR, exist_ok=True)
#         os.makedirs(self.SYSTEM_DIR, exist_ok=True)
#         self.repo = repo

#     async def get_all_files(self, req: FileSearchRequest):
#         return await self.repo.search_files(req)
#         pass

#     async def save_files(self, 
#         type_storage: str, 
#         files: List[UploadFile]
#     ):
#         for file in files:
#             new_id = uuid6.uuid7()
#             file_name = os.path.splitext(file.filename)[0]
#             file_ext = os.path.splitext(file.filename)[1]
#             filename = f"{new_id}_{file_name}{file_ext}"
            
#             file_path = self._get_file_path(type_storage, filename)

#             with open(file_path, "wb") as f:
#                 content = await file.read()
#                 f.write(content)

#             attachment = Attachment()
#             attachment.id = new_id
#             attachment.file_name = filename
#             attachment.url = "http://{}:{}/{}/{}?type_storage={}".format(
#                 os.getenv("HOST_SERVER"), 
#                 os.getenv("PORT_SERVER"), 
#                 f"{self.STORAGE_PREFIX}",
#                 filename,
#                 f"{type_storage}_storage"
#             )
#             await self.repo.save(attachment)
#         pass
            
#     async def get_file_by_id(self, id: str):
#         file = await self.repo.find_by_id(id)

#         if not file: 
#             raise APIException(
#                 "File not found",
#                 status_code=status.HTTP_404_NOT_FOUND
#             )

#         path = file.url.split("/").pop().split("?")
#         file_name = path[0]
#         type_storage = path[1].split("=")[1]

#         file_path = "./static/{}/{}".format(type_storage, file_name)

#         logger.info(file_path)

#         if not os.path.exists(file_path):
#             raise APIException(
#                 "Can't find file",
#                 status_code=status.HTTP_404_NOT_FOUND
#             )

#         mime_type, _ = mimetypes.guess_type(file_path)

#         return {
#             "file_path": file_path,
#             "mime_type": mime_type
#         }

#     async def delete_file(self, req: DeleteFileRequest):
#         # file_path = self._get_file_path(req.type_storage, req.filename)

#         # if not os.path.exists(file_path):
#         #     raise APIException(
#         #         "Can't find file",
#         #         status_code=status.HTTP_400_BAD_REQUEST
#         #     )

#         # os.remove(file_path)
#         file = await self.repo.find_by_id(req.file_id)
#         file.delete_at = datetime.datetime.now()
#         await self.repo.save(file)
#         pass

#     def _get_file_path(self, type_storage: str, filename: str):
#         if type_storage == "user_storage":
#             return os.path.join(self.USER_STORAGE_DIR, filename)
#         elif type_storage == "system_storage":
#             return os.path.join(self.SYSTEM_DIR, filename)
#         else:
#             raise APIException(
#                 "Invalid file type",
#                 status_code=status.HTTP_400_BAD_REQUEST
#             )
