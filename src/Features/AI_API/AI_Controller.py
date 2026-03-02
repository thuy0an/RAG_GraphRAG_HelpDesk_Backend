# import logging
# from dotenv import load_dotenv
# from fastapi import APIRouter, Depends, File, UploadFile, Form
# from pydantic import BaseModel
# from starlette import status
# from src.Features.AI_API.AI_Service import AIService
# from src.Shared.base import get_logger
# from src.Shared.base.APIResponse import APIResponse

# load_dotenv()

# router = APIRouter(
#     tags=["LangChain"]
# )

# prefix="api/v1/langchain",

# logger = get_logger(__name__)

# @router.post(f"${prefix}/ingest")
# async def ingest(
#     file: UploadFile = File(...),
#     service: AIService = Depends()
# ):
#     logger.info(f"File name: {type(file.file)}")
#     result = await service.ingest_doc(file.filename, file.file)
    
#     return APIResponse(
#         message="Query successfully",
#         status_code=status.HTTP_200_OK,
#         data=result
#     )

# class SearchRequest(BaseModel):
#     query: str
# @router.post(f"{prefix}/doc_rag")
# async def doc_rag(
#     req: SearchRequest, 
#     service: AIService = Depends()
# ):
#     result = await service.rag_query(req.query)

#     return APIResponse(
#         message="Query successfully",
#         status_code=status.HTTP_200_OK,
#         data=result
#     )

# @router.post(f"${prefix}/sql_rag")
# async def sql_rag(
#     req: SearchRequest, 
#     service: AIService = Depends()
# ):
#     result = await service.rag_sql(req.query)
    
#     return APIResponse(
#         message="Query successfully",
#         status_code=status.HTTP_200_OK,
#         data=result
#     )
