from typing import List, Optional
from pydantic import BaseModel
from Features.LangChainAPI.LangChainFacade import LangChainFacade
from Features.LangChainAPI.LangTools import LangTools
from SharedKernel.config.LLMConfig import LLMFactory
import asyncio
import io
from fastapi import APIRouter, Depends, FastAPI, File, UploadFile, status, Form, Query
from starlette.datastructures import Headers
from fastapi.responses import StreamingResponse
from Features.LangChainAPI.LangChainDTO import ChatRequest
from SharedKernel.persistence.Decorators import Controller
from src.SharedKernel.base.APIResponse import APIResponse
from src.Features.LangChainAPI.persistence.CompareRepository import CompareRepository
from src.Features.RealTimeAPI.FileSystem.StorageService import StorageService


@Controller
class LangChainController:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.adv_router = APIRouter(
            prefix="/api/v1/langchain/naive", tags=["Naive RAG"]
        )
        self.adv_router = APIRouter(
            prefix="/api/v1/langchain/adv", tags=["Advanced RAG"]
        )
        self.graph_router = APIRouter(
            prefix="/api/v1/langchain/graph", tags=["Graph RAG"]
        )
        self.tool_router = APIRouter(prefix="/api/v1/langchain/tools", tags=["Tools"])
        self.compare_router = APIRouter(
            prefix="/api/v1/langchain/compare", tags=["Compare"]
        )

        self.tools = LangTools()
        self.register_route()

        self.app.include_router(self.adv_router)
        self.app.include_router(self.graph_router)
        self.app.include_router(self.tool_router)
        self.app.include_router(self.compare_router)

    def register_adv_routes(self):
        """Register PaCRAG routes"""

        @self.adv_router.post("/load_document")
        async def load_document(
            files: List[UploadFile] = File(...),
            parent_chunk_size: int | None = Form(None),
            parent_chunk_overlap: int | None = Form(None),
            child_chunk_size: int | None = Form(None),
            child_chunk_overlap: int | None = Form(None),
            langfacade: LangChainFacade = Depends(),
        ):
            for file in files:
                await langfacade.PaCRAG.index(
                    file,
                    parent_chunk_size=parent_chunk_size,
                    parent_chunk_overlap=parent_chunk_overlap,
                    child_chunk_size=child_chunk_size,
                    child_chunk_overlap=child_chunk_overlap,
                )

            return APIResponse(
                message=f"Successfully indexing {len(files)} PDF file(s)",
                status_code=status.HTTP_200_OK,
                result=None,
            )

        class DeleteDocumentRequest(BaseModel):
            filename: str

        @self.adv_router.delete("/delete_document")
        async def delete_document(
            req: DeleteDocumentRequest, langfacade: LangChainFacade = Depends()
        ):
            await langfacade.PaCRAG.delete(req.filename)
            return APIResponse(
                message=f"Delete successfully",
                status_code=status.HTTP_200_OK,
                result=None,
            )

        @self.adv_router.delete("/clear_vector_store")
        async def clear_vector_store(
            source: str | None = Query(default=None),
            langfacade: LangChainFacade = Depends(),
        ):
            await langfacade.PaCRAG.clear_vector_store(source)
            return APIResponse(
                message="Vector store cleared",
                status_code=status.HTTP_200_OK,
                result={"source": source},
            )

        class RetrieveDocumentRequest(BaseModel):
            query: str
            session_id: str

        @self.adv_router.post("/retrieve_document")
        async def retrieve_document(
            req: RetrieveDocumentRequest, 
            langfacade: LangChainFacade = Depends()
        ):
            return StreamingResponse(
                langfacade.PaCRAG.retrieve(req.query, req.session_id),
                media_type="text/event-stream",
            )

    def register_graph_routes(self):
        """Register GraphRAG routes"""

        @self.graph_router.post("/build")
        async def build_graph(
            file: UploadFile = File(...), langfacade: 
            LangChainFacade = Depends()
        ):
            source = file.filename
            result = await langfacade.GraphRAG.ingest(file, source)

            return APIResponse(
                message="Graph built successfully",
                status_code=status.HTTP_200_OK,
                result=result,
            )

        @self.graph_router.get("/{source}/stats")
        async def get_graph_stats(source: str, langfacade: LangChainFacade = Depends()):
            stats = langfacade.GraphRAG.internal.get_graph_stats(source)

            return APIResponse(
                message="Graph stats retrieved",
                status_code=status.HTTP_200_OK,
                result=stats,
            )

        class GraphSearchRequest(BaseModel):
            query: str
            top_k: int = 5

        @self.graph_router.post("/search")
        async def search_graph(
            req: GraphSearchRequest, langfacade: LangChainFacade = Depends()
        ):
            results = await langfacade.GraphRAG.neo4j_store.search_by_embedding(
                req.query, req.top_k
            )
            return APIResponse(
                message="Search completed",
                status_code=status.HTTP_200_OK,
                result={"results": results},
            )

        class GraphQueryRequest(BaseModel):
            query: str
            session_id: str
            source: Optional[str] = None

        @self.graph_router.post("/query")
        async def query_graph(
            req: GraphQueryRequest, langfacade: LangChainFacade = Depends()
        ):
            result = await langfacade.GraphRAG.retrieve(req.query, session_id=req.session_id, source=req.source)
            return APIResponse(
                message="Query completed", status_code=status.HTTP_200_OK, result=result
            )

        @self.graph_router.delete("/{source}")
        async def delete_graph(source: str, langfacade: LangChainFacade = Depends()):
            await langfacade.GraphRAG.delete(source)
            return APIResponse(
                message="Graph deleted successfully",
                status_code=status.HTTP_200_OK,
                result=None,
            )

    def register_compare_routes(self):
        """Register Compare (PaCRAG vs GraphRAG) routes"""

        @self.compare_router.post("/upload")
        async def compare_upload(
            files: List[UploadFile] = File(...),
            session_id: str = Form(...),
            parent_chunk_size: int | None = Form(None),
            parent_chunk_overlap: int | None = Form(None),
            child_chunk_size: int | None = Form(None),
            child_chunk_overlap: int | None = Form(None),
            storage_service: StorageService = Depends(),
            langfacade: LangChainFacade = Depends(),
            compare_repo: CompareRepository = Depends(),
        ):
            results = []

            # Step 1: Save files and index with PaCRAG using chunking via StorageService
            chunked_response = await storage_service.save_files_with_chunking(
                files=files,
                parent_chunk_size=parent_chunk_size,
                parent_chunk_overlap=parent_chunk_overlap,
                child_chunk_size=child_chunk_size,
                child_chunk_overlap=child_chunk_overlap
            )

            # Step 2: Index with GraphRAG for each file
            for i, file in enumerate(files):
                content_type = file.content_type or "application/octet-stream"
                pac_metrics = chunked_response.indexing_results[i] if i < len(chunked_response.indexing_results) else {}
                uploaded_file_info = chunked_response.uploaded_files[i] if i < len(chunked_response.uploaded_files) else {}
                file_size = uploaded_file_info.get("size", 0) if isinstance(uploaded_file_info, dict) else 0

                # Reset file pointer for GraphRAG
                await file.seek(0)

                try:
                    graph_result = await langfacade.GraphRAG.ingest(file, file.filename)
                    graph_metrics = graph_result
                    errors = None
                except Exception as e:
                    graph_metrics = {}
                    errors = {"graphrag": str(e)}

                run = await compare_repo.create_run(
                    session_id=session_id,
                    file_name=file.filename,
                    file_type=content_type,
                    file_size=file_size,
                    pac_ingest=pac_metrics,
                    graphrag_ingest=graph_metrics,
                    errors=errors,
                )

                results.append(run)

            return APIResponse(
                message="Comparison ingest completed",
                status_code=status.HTTP_200_OK,
                result={"runs": results},
            )

        class CompareQueryRequest(BaseModel):
            session_id: str
            run_id: str
            query: str

        @self.compare_router.post("/query")
        async def compare_query(
            req: CompareQueryRequest,
            langfacade: LangChainFacade = Depends(),
            compare_repo: CompareRepository = Depends(),
        ):
            pac_result = await langfacade.PaCRAG.retrieve_with_metrics(
                req.query,
                session_id=req.session_id
            )
            graph_result = await langfacade.GraphRAG.retrieve_with_metrics(
                req.query,
                session_id=req.session_id,
                source=None
            )

            errors = {}
            pac_metrics = pac_result or {}
            graph_metrics = graph_result or {}

            run = await compare_repo.update_query_metrics(
                req.run_id,
                pac_query=pac_metrics,
                graphrag_query=graph_metrics,
            )

            return APIResponse(
                message="Comparison query completed",
                status_code=status.HTTP_200_OK,
                result={
                    "run": run,
                    "errors": errors or None,
                },
            )

        @self.compare_router.get("/history/{session_id}")
        async def compare_history(
            session_id: str,
            compare_repo: CompareRepository = Depends(),
        ):
            runs = await compare_repo.list_runs(session_id)
            return APIResponse(
                message="Comparison history retrieved",
                status_code=status.HTTP_200_OK,
                result={"runs": runs},
            )

        @self.compare_router.delete("/history/{run_id}")
        async def compare_history_delete(
            run_id: str,
            compare_repo: CompareRepository = Depends(),
        ):
            deleted = await compare_repo.delete_run(run_id)
            return APIResponse(
                message="Comparison run deleted",
                status_code=status.HTTP_200_OK,
                result={"deleted": deleted},
            )

    def register_tool_routes(self):
        """Register Tool routes"""

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
            return StreamingResponse(
                self.tools.ascrawl_web(req.message), media_type="text/event-stream"
            )

    def register_route(self):
        """Register all routes"""
        self.register_adv_routes()
        self.register_compare_routes()
        self.register_graph_routes()
        self.register_tool_routes()
