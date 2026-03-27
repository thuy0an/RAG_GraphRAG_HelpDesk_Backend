from typing import List, Optional

from pydantic import BaseModel
from Features.LangChainAPI.LangChainFacade import LangChainFacade
from Features.LangChainAPI.LangTools import LangTools
from SharedKernel.ai.AIConfig import AIConfigFactory
from fastapi import APIRouter, Body, Depends, FastAPI, File, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from Features.LangChainAPI.LangChainDTO import ChatMessageRequest, ChatRequest, ChatTechniqueRequest, ChatTemplateRequest, PromptType, RagRequest, RagType, TechType, TemplateType
from SharedKernel.persistence.Decorators import Controller
from src.SharedKernel.base.APIResponse import APIResponse

@Controller
class LangChainController:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.router = APIRouter(
            prefix="/api/v1/langchain",
            tags=["LangChain"]
        )
        self.tool_router = APIRouter(
            prefix="/api/v1/tools",
            tags=["Tools"]
        )
        self.tools = LangTools()
        self.register_route()
        self.tool_route()
        self.app.include_router(self.router)
        self.app.include_router(self.tool_router)

    def register_route(self):        
        @self.router.post("/long_chat")
        async def long_chat(
            req: ChatMessageRequest,
            langfacade: LangChainFacade = Depends()
        ):
            """Long chat với streaming responses"""
            return StreamingResponse(
                await langfacade.memory_service.long_chat(req),
                media_type="text/event-stream"
            )
        
        class ChatHistoryRequest(BaseModel):
            page_number: Optional[int] = 1
            page_size: Optional[int] = 10
            
        @self.router.get("/chat_history/{session_id}")
        async def get_chat_history(
            session_id: str,
            langfacade: LangChainFacade = Depends()
        ):
            """Get paginated chat history for a session"""
            response = await langfacade.synthesizer.memory_repo.get_history_all(
                session_id=session_id,
            )
            
            return APIResponse(
                message="Chat history retrieved successfully",
                status_code=status.HTTP_200_OK,
                data=response
            )

        @self.router.post("/load_document_pdf_PaC")
        async def load_document_pdf_PaC(
            files: List[UploadFile] = File(...),
            langfacade: LangChainFacade = Depends()
        ):
            for file in files:
                await langfacade.synthesizer.ingest_pdf_PaC(file)

            return APIResponse(
                message=f"Successfully processed {len(files)} PDF file(s)",
                status_code=status.HTTP_200_OK,
                data=None
            )
        
        class DeleteDocumentRequest(BaseModel):
            filename: str
        @self.router.delete("/delete_document")
        async def delete_document(
            req: DeleteDocumentRequest, 
            langfacade: LangChainFacade = Depends()
        ):
            await langfacade.synthesizer.delete_document_by_file_name(req.filename)
            return APIResponse(
                message=f"Delete successfully",
                status_code=status.HTTP_200_OK,
                data=None
            )
            ...

        class RetrieveDocumentRequest(BaseModel):
            query: str
            session_id: str
        @self.router.post("/retrieve_document")
        async def retrieve_document(
            req: RetrieveDocumentRequest, 
            langfacade: LangChainFacade = Depends()
        ):           
            return StreamingResponse(
                await langfacade.synthesizer.rag_PaC(req.query, req.session_id),
                media_type="text/event-stream"
            )
            ...

        @self.tool_router.post("/crawl")
        def web_crawler(url: str, langfacade: LangChainFacade = Depends()):
            langfacade.crawler.crawl_data(url)
            
    def tool_route(self):
        @self.tool_router.post("/tool_search")
        async def web_search(req: ChatRequest):
            urls = await self.tools.duckduckgo_search(req.message)
            contents = []
            for url in urls:
                content = await self.tools.crawl_web(url)
                contents.append(content)
            return contents

        @self.tool_router.post("/web_fetch")
        async def web_fetch(req: ChatRequest):
            content = await self.tools.crawl_web(req.message)
            return content

        @self.tool_router.post("/fetch")
        def web_asfetch(req: ChatRequest):
            # content = await self.tools.crawl(req.message)
            return StreamingResponse(
                self.tools.ascrawl_web(req.message),
                media_type="text/event-stream"
            )
        ...
    ...