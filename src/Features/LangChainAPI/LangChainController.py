# import os
# from typing import List
# from fastapi import APIRouter, Depends, File, UploadFile
# from fastapi.responses import StreamingResponse
# from Features import get_logger
# from src.Features.LangChainAPI.LangTools import crawl, crawl_tool_stream, duckduckgo_search
# from src.Features.LangChainAPI.LangChainService import LangChainService
# from src.Features.LangChainAPI.LangChainDTO import ChatRequest, ChatTechniqueRequest, ChatTemplateRequest, MemoryType, PromptType, TechType, TemplateType
# from langchain_community.tools import DuckDuckGoSearchRun

# logger = get_logger(__name__)

# router = APIRouter(
#   prefix="/api/v1/langchain",
#   tags=["Lang Chain"]
# ) 

# # ====================
# # CHUONG: PROMPT
# # ====================
# @router.post("/")
# async def prompt(
#   req: ChatRequest,
#   prompt_type: PromptType = PromptType.none,
#   lang_chain_service: LangChainService = Depends() 
# ):
#     if prompt_type == PromptType.none:
#         return await lang_chain_service.prompt(req)
#     elif prompt_type == PromptType.stream:
#         return StreamingResponse(
#             lang_chain_service.stream_prompt(req),
#             media_type="text/event-stream"
#         )

# # ========================================
# # CHUONG: ENGINEERING PROMPT
# # ========================================
# @router.get("/technique")
# async def prompt_technique(
#     tech_type: TechType,
#     lang_chain_service: LangChainService = Depends() 
# ):
#     dto = ChatTechniqueRequest(message="", tech=tech_type)
#     async def event_generator():
#         async for token in lang_chain_service.promt_techniques(dto):
#             yield token
    
#     return StreamingResponse(
#         event_generator(),
#         media_type="text/event-stream"
#     )

# # ====================
# # CHUONG: 
# # ====================
# @router.post("/template")
# async def prompt_template(
#     req: ChatRequest,
#     template_type: TemplateType,
#     lang_chain_service: LangChainService = Depends() 
# ):
#     dto = ChatTemplateRequest(message=req.message, template=template_type)    
#     return StreamingResponse(
#         lang_chain_service.prompt_template(dto),
#         media_type="text/event-stream"
#     )

# @router.post("/search")
# async def search(
#     req: ChatRequest,
#     lang_chain_service: LangChainService = Depends() 
# ):
#     urls = duckduckgo_search(req.message)
#     all_docs = []
#     for url in urls:
#         content = ""
#         async for chunk in crawl_tool_stream(url):
#             content += chunk
#         all_docs.append(content)
    
#     return all_docs
#     # return duckduckgo_search(req.message)

# @router.post("/fetch")
# def fetch_web(
#     req: ChatRequest,
#     lang_chain_service: LangChainService = Depends() 
# ):
#     return StreamingResponse(
#         crawl_tool_stream(req.message),
#         media_type="text/event-stream"
#     )

# @router.post("/tools")
# async def tools(
#     lang_chain_service: LangChainService = Depends() 
# ):
#     return StreamingResponse(
#         lang_chain_service.tools(),
#         media_type="text/plain"
#     )

# @router.post("/tools_stream")
# async def tools_stream(
#     lang_chain_service: LangChainService = Depends() 
# ):
#     # return await lang_chain_service.tools()
#     return StreamingResponse(
#         lang_chain_service.tools_stream(),
#         media_type="text/event-stream"
#     )

# @router.post("/structed_output")
# async def structed_output(
#     lang_chain_service: LangChainService = Depends() 
# ):
#     return await lang_chain_service.structed_output()   

# # ====================
# # CHUONG: MEMORY
# # ====================
# @router.post("/short_memory")
# async def short_chat(
#     user_id: str, 
#     message: str, 
#     service: LangChainService = Depends()
# ):
#     # return await lang_chain_service.short_chat(user_id, message)
#     return await service.short_chat_no_runnable(user_id, message)
#     pass

# @router.post("/long_memory")
# async def long_chat(
#     user_id: str, 
#     message: str, 
#     service: LangChainService = Depends()
# ):
#     return await service.long_chat(user_id, message)
#     pass

# # =======================
# # CHUONG: DOCUMENT LOADER
# # =======================
# @router.post("/document_load")
# async def load_document(
#     files: List[UploadFile] = File(...),
#     service: LangChainService = Depends()
# ):
#     logger.info("test")
#     # service.load_pdf(files)
#     return StreamingResponse(
#         service.load_pdf(files),
#         media_type="text/event-stream"
#     )
#     pass

from socket import has_dualstack_ipv6
from typing import List
from Features.LangChainAPI.LangChainFacade import LangChainFacade
from Features.LangChainAPI.LangTools import LangTools
from Features.LangChainAPI.service.PromptService import PromptService
from SharedKernel.AIConfig import AIConfig, AIConfigFactory
from fastapi import APIRouter, FastAPI, File, Request, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from lagom import Container, Singleton
from Features.LangChainAPI.LangChainDTO import ChatMessageRequest, ChatRequest, ChatTechniqueRequest, ChatTemplateRequest, PromptType, TechType, TemplateType
from SharedKernel.persistence.Decorators import Controller

@Controller
class LangChainController:
    def __init__(self, app: FastAPI, container: Container) -> None:
        self.app = app
        self.container = container 
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
        self.container[AIConfigFactory] = AIConfigFactory()
        langfacade = self.container[LangChainFacade]

        @self.router.post("/")
        async def prompt(
            req: ChatRequest,
            prompt_type: PromptType = PromptType.NONE
        ):
            async def handle_none(req):
                return await langfacade.prompt.aprompt(req)
            async def handle_stream(req):
                result = await langfacade.prompt.asprompt(req)
                return StreamingResponse(
                    result["content"],
                    media_type="text/event-stream"
                )

            prompt_dict = {
                PromptType.NONE: handle_none,
                PromptType.STREAM: handle_stream
            }

            handler = prompt_dict.get(prompt_type)
            if handler is None:
                raise ValueError(f"Invalid prompt type: {prompt_type}")
            return await handler(req)
        ...

        @self.router.get("/technique")
        async def prompt_technique(
            tech_type: TechType
        ):
            dto = ChatTechniqueRequest(message="", tech=tech_type)
            handler = await langfacade.prompt.promt_techniques(dto)
            result = await handler.get(tech_type)

            if result is None:
                raise ValueError(f"Invalid prompt type: {tech_type}")

            return StreamingResponse(
                result["content"],
                media_type="text/event-stream"
            )
        ...

        @self.router.post("/template")
        async def prompt_template(
            req: ChatRequest,
            template_type: TemplateType
        ):
            dto = ChatTemplateRequest(message=req.message, template=template_type)
            handler = await langfacade.prompt.prompt_template(dto)
            result = await handler.get(template_type)
            
            if handler is None:
                raise ValueError(f"Invalid prompt type: {template_type}")   

            return StreamingResponse(
                result["content"],
                media_type="text/event-stream"
            )
        ...

        @self.router.post("/structed_output")
        async def structed_output():
            result = await langfacade.output_parser.structed_output()
            return result
        ...

        @self.router.post("/tools")
        async def tools(
            req: ChatRequest
        ):
            handler = await langfacade.tools.call_search_tools()
            result = handler.get("content")

            return StreamingResponse(
                result,
                media_type="text/event-stream"
            )

        @self.router.post("/short_chat")
        async def short_chat(
            req: ChatMessageRequest
        ):
            return await langfacade.memory.short_chat(req)

        @self.router.post("/long_chat")
        async def long_chat(
            req: ChatMessageRequest
        ):
            print(req.session_id)
            return await langfacade.memory.long_chat(req)

        @self.router.post("/document_webpage")
        def document_webpage(
            req: ChatRequest
        ):
            result = langfacade.LPI.loader.load_webpage(req.message)
 
            return StreamingResponse(
                result,
                media_type="text/event-stream"
            )
        ...

        @self.router.post("/document_pdf")
        async def document_pdf(
            files: List[UploadFile] = File(...),
        ):
            for file in files:
                await langfacade.LPI.pdf_handler(file)

        @self.router.post("/search_document")
        async def search_document(query: str):
            return await langfacade.LPI.vector_store_repo.search(query)
            ...
            
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