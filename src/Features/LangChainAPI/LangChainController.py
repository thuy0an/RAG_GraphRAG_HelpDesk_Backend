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


@Controller
class LangChainController:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.router = APIRouter(prefix="/api/v1/langchain", tags=["LangChain"])
        self.tool_router = APIRouter(prefix="/api/v1/tools", tags=["Tools"])
        self.tools = LangTools()
        self.register_route()
        self.app.include_router(self.router)

    def register_route(self):
        compare_repo = CompareRepository()
        #
        # RAG
        #
        @self.router.get("/chat_history/{session_id}")
        async def get_chat_history(
            session_id: str, langfacade: LangChainFacade = Depends()
        ):
            """Get paginated chat history for a session"""
            response = await langfacade.PaCRAG.get_chat_history(
                session_id=session_id,
            )

            return APIResponse(
                message="Chat history retrieved successfully",
                status_code=status.HTTP_200_OK,
                data=response,
            )

        @self.router.post("/load_document_pdf_PaC")
        async def load_document_pdf_PaC(
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
                data=None,
            )

        class DeleteDocumentRequest(BaseModel):
            filename: str

        @self.router.delete("/delete_document")
        async def delete_document(
            req: DeleteDocumentRequest, 
            langfacade: LangChainFacade = Depends()
        ):
            await langfacade.PaCRAG.delete(req.filename)
            return APIResponse(
                message=f"Delete successfully",
                status_code=status.HTTP_200_OK,
                data=None,
            )
            ...

        @self.router.delete("/clear_history/{session_id}")
        async def clear_chat_history(
            session_id: str,
            langfacade: LangChainFacade = Depends(),
        ):
            deleted = await langfacade.PaCRAG.clear_history(session_id)
            return APIResponse(
                message="Chat history cleared",
                status_code=status.HTTP_200_OK,
                data={"deleted": deleted},
            )

        @self.router.delete("/clear_vector_store")
        async def clear_vector_store(
            source: str | None = Query(default=None),
            langfacade: LangChainFacade = Depends(),
        ):
            await langfacade.PaCRAG.clear_vector_store(source)
            return APIResponse(
                message="Vector store cleared",
                status_code=status.HTTP_200_OK,
                data={"source": source},
            )

        #
        # COMPARE (PaCRAG vs GraphRAG)
        #
        @self.router.post("/compare/upload")
        async def compare_upload(
            files: List[UploadFile] = File(...),
            session_id: str = Form(...),
            parent_chunk_size: int | None = Form(None),
            parent_chunk_overlap: int | None = Form(None),
            child_chunk_size: int | None = Form(None),
            child_chunk_overlap: int | None = Form(None),
            langfacade: LangChainFacade = Depends(),
        ):
            results = []

            for file in files:
                content = await file.read()
                content_type = file.content_type or "application/octet-stream"
                upload_headers = Headers({"content-type": content_type})
                pac_file = UploadFile(
                    filename=file.filename,
                    file=io.BytesIO(content),
                    headers=upload_headers,
                )
                graph_file = UploadFile(
                    filename=file.filename,
                    file=io.BytesIO(content),
                    headers=upload_headers,
                )

                pac_task = langfacade.PaCRAG.index_with_metrics(
                    pac_file,
                    parent_chunk_size=parent_chunk_size,
                    parent_chunk_overlap=parent_chunk_overlap,
                    child_chunk_size=child_chunk_size,
                    child_chunk_overlap=child_chunk_overlap,
                )
                graph_task = langfacade.GraphRAG.ingest(
                    graph_file,
                    file.filename,
                    chunk_size=child_chunk_size,
                    chunk_overlap=child_chunk_overlap,
                )

                pac_result, graph_result = await asyncio.gather(
                    pac_task, graph_task, return_exceptions=True
                )

                errors = {}
                if isinstance(pac_result, Exception):
                    errors["pac"] = str(pac_result)
                    pac_metrics = {}
                else:
                    pac_metrics = pac_result

                if isinstance(graph_result, Exception):
                    errors["graphrag"] = str(graph_result)
                    graph_metrics = {}
                else:
                    graph_metrics = graph_result

                run = await compare_repo.create_run(
                    session_id=session_id,
                    file_name=file.filename,
                    file_type=content_type,
                    file_size=len(content),
                    pac_ingest=pac_metrics,
                    graphrag_ingest=graph_metrics,
                    errors=errors or None,
                )

                results.append(run)

            return APIResponse(
                message="Comparison ingest completed",
                status_code=status.HTTP_200_OK,
                data={"runs": results},
            )

        class CompareQueryRequest(BaseModel):
            session_id: str
            run_id: str
            query: str

        @self.router.post("/compare/query")
        async def compare_query(
            req: CompareQueryRequest,
            langfacade: LangChainFacade = Depends(),
        ):
            pac_task = langfacade.PaCRAG.retrieve_full(req.query, session_id=req.session_id)
            graph_task = langfacade.GraphRAG.retrieve_with_metrics(req.query, source=None, session_id=req.session_id)

            pac_result, graph_result = await asyncio.gather(
                pac_task, graph_task, return_exceptions=True
            )

            errors = {}
            pac_metrics = None
            graph_metrics = None

            if isinstance(pac_result, Exception):
                errors["pac"] = str(pac_result)
            else:
                pac_metrics = pac_result

            if isinstance(graph_result, Exception):
                errors["graphrag"] = str(graph_result)
            else:
                graph_metrics = graph_result

            run = await compare_repo.update_query_metrics(
                req.run_id,
                pac_query=pac_metrics,
                graphrag_query=graph_metrics,
            )

            return APIResponse(
                message="Comparison query completed",
                status_code=status.HTTP_200_OK,
                data={
                    "run": run,
                    "errors": errors or None,
                },
            )

        @self.router.get("/compare/history/{session_id}")
        async def compare_history(
            session_id: str,
        ):
            runs = await compare_repo.list_runs(session_id)
            return APIResponse(
                message="Comparison history retrieved",
                status_code=status.HTTP_200_OK,
                data={"runs": runs},
            )

        @self.router.delete("/compare/history/{run_id}")
        async def compare_history_delete(
            run_id: str,
        ):
            deleted = await compare_repo.delete_run(run_id)
            return APIResponse(
                message="Comparison run deleted",
                status_code=status.HTTP_200_OK,
                data={"deleted": deleted},
            )

        class RetrieveDocumentRequest(BaseModel):
            query: str
            session_id: str

        @self.router.post("/retrieve_document")
        async def retrieve_document(
            req: RetrieveDocumentRequest, 
            langfacade: LangChainFacade = Depends()
        ):
            async def generate():
                async for chunk in langfacade.PaCRAG.retrieve(req.query, req.session_id):
                    if chunk:
                        yield chunk

            return StreamingResponse(
                generate(),
                media_type="text/plain; charset=utf-8",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
            ...

        #
        # GRAPH
        # 
        @self.router.post("/build-graph")
        async def build_graph(
            file: UploadFile = File(...),
            langfacade: LangChainFacade = Depends()
        ):
            source = file.filename
            result = await langfacade.GraphRAG.ingest(file, source)
            return APIResponse(
                message="Graph built successfully",
                status_code=status.HTTP_200_OK,
                data=result,
            )

        @self.router.get("/graph/{source}/stats")
        async def get_graph_stats(
            source: str,
            langfacade: LangChainFacade = Depends()
        ):
            stats = langfacade.GraphRAG.internal.get_graph_stats(source)
            return APIResponse(
                message="Graph stats retrieved",
                status_code=status.HTTP_200_OK,
                data=stats,
            )

        class GraphSearchRequest(BaseModel):
            query: str
            top_k: int = 5

        @self.router.post("/graph/search")
        async def search_graph(
            req: GraphSearchRequest,
            langfacade: LangChainFacade = Depends()
        ):
            results = await langfacade.GraphRAG.neo4j_store.search_by_embedding(
                req.query, req.top_k
            )
            return APIResponse(
                message="Search completed",
                status_code=status.HTTP_200_OK,
                data={"results": results},
            )

        class GraphQueryRequest(BaseModel):
            query: str
            source: Optional[str] = None
            session_id: Optional[str] = None

        @self.router.post("/graph/query")
        async def query_graph(
            req: GraphQueryRequest,
            langfacade: LangChainFacade = Depends()
        ):
            result = await langfacade.GraphRAG.retrieve(req.query, source=req.source, session_id=req.session_id)
            return APIResponse(
                message="Query completed", status_code=status.HTTP_200_OK, data=result
            )

        @self.router.delete("/graph/{source}")
        async def delete_graph(
            source: str,
            langfacade: LangChainFacade = Depends()
        ):
            await langfacade.GraphRAG.delete(source)
            return APIResponse(
                message="Graph deleted successfully",
                status_code=status.HTTP_200_OK,
                data=None,
            )

        @self.router.get("/graph/history/{session_id}")
        async def get_graph_history(
            session_id: str,
            langfacade: LangChainFacade = Depends()
        ):
            """Get GraphRAG chat history for a session"""
            response = await langfacade.GraphRAG.get_chat_history(session_id=session_id)
            return APIResponse(
                message="GraphRAG chat history retrieved",
                status_code=status.HTTP_200_OK,
                data=response,
            )

        @self.router.delete("/graph/history/{session_id}")
        async def clear_graph_history(
            session_id: str,
            langfacade: LangChainFacade = Depends()
        ):
            """Clear GraphRAG chat history for a session"""
            deleted = await langfacade.GraphRAG.clear_history(session_id)
            return APIResponse(
                message="GraphRAG chat history cleared",
                status_code=status.HTTP_200_OK,
                data={"deleted": deleted},
            )

        @self.router.delete("/graph")
        async def delete_all_graph(
            langfacade: LangChainFacade = Depends()
        ):
            """Xóa toàn bộ Neo4j graph (không có source cụ thể)"""
            await langfacade.GraphRAG.neo4j_store.delete_graph(source=None)
            return APIResponse(
                message="All graph data deleted successfully",
                status_code=status.HTTP_200_OK,
                data=None,
            )

        ...

        #
        # TOOL 
        #
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
                self.tools.ascrawl_web(req.message), media_type="text/event-stream"
            )

    ...
