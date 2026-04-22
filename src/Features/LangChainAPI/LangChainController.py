import logging
import numpy as np
from typing import List, Optional
from pydantic import BaseModel
from Features.LangChainAPI.LangChainFacade import LangChainFacade
import asyncio
import io
from fastapi import APIRouter, Depends, FastAPI, File, UploadFile, status, Form, Query
from starlette.datastructures import Headers
from fastapi.responses import StreamingResponse
from SharedKernel.persistence.Decorators import Controller
from src.SharedKernel.base.APIResponse import APIResponse
from src.Features.LangChainAPI.persistence.CompareRepository import CompareRepository

log = logging.getLogger(__name__)


def _compute_relevance_score(embedding_model, query: str, answer: str) -> Optional[float]:
    """Tính cosine similarity giữa embedding của query và answer."""
    try:
        q_vec = np.array(embedding_model.embed_query(query))
        a_vec = np.array(embedding_model.embed_query(answer))
        norm = np.linalg.norm(q_vec) * np.linalg.norm(a_vec)
        if norm == 0:
            return None
        cosine = float(np.dot(q_vec, a_vec) / norm)
        return round(max(0.0, min(1.0, cosine)), 4)
    except Exception as e:
        log.error(f"Failed to compute relevance score: {e}")
        return None


def _compute_source_coverage(sources: list, retrieved_chunk_count: int) -> Optional[float]:
    """Tính tỉ lệ unique sources / retrieved_chunk_count, clamped to [0.0, 1.0]."""
    if not retrieved_chunk_count:
        return None
    unique_sources = len({s.get("filename", "") for s in sources if s.get("filename")})
    return round(min(1.0, unique_sources / retrieved_chunk_count), 4)


def _summarize_scores(scores: object) -> Optional[dict]:
    if not isinstance(scores, list):
        return None

    numeric_scores = [float(score) for score in scores if isinstance(score, (int, float))]
    if not numeric_scores:
        return None

    avg_score = sum(numeric_scores) / len(numeric_scores)
    return {
        "count": len(numeric_scores),
        "avg": round(avg_score, 4),
        "min": round(min(numeric_scores), 4),
        "max": round(max(numeric_scores), 4),
    }


def _build_metric_groups(metrics: Optional[dict], query: str) -> Optional[dict]:
    if not metrics:
        return None

    retrieved_chunk_count = int(metrics.get("retrieved_chunk_count") or 0)
    retrieved_source_count = int(metrics.get("retrieved_source_count") or 0)
    answer = metrics.get("answer", "")
    latency_breakdown = metrics.get("latency_breakdown") or {}

    system_metrics = dict(metrics.get("system_metrics") or {})
    system_metrics.setdefault("time_total_s", metrics.get("time_total_s"))
    system_metrics.setdefault("answer_tokens", metrics.get("answer_tokens"))
    system_metrics.setdefault("word_count", metrics.get("word_count"))

    relevance_score = metrics.get("relevance_score")
    source_coverage = metrics.get("source_coverage")
    confidence_score = metrics.get("confidence_score")

    generation_metrics = {
        "answer_length_tokens": metrics.get("answer_tokens"),
        "word_count": metrics.get("word_count"),
        "confidence_score": confidence_score,
        "answer_relevance_proxy": relevance_score,
        "faithfulness_proxy": confidence_score,
        # Normalized proxy metrics for generation quality (0..1)
        "faithfulness_groundedness": confidence_score,
        "answer_relevancy": relevance_score,
    }

    if confidence_score is not None and relevance_score is not None:
        generation_metrics["answer_correctness"] = round(
            (float(confidence_score) + float(relevance_score)) / 2.0,
            4,
        )
    else:
        generation_metrics["answer_correctness"] = confidence_score if confidence_score is not None else relevance_score

    retrieval_metrics = dict(metrics.get("retrieval_metrics") or {})
    retrieval_metrics.setdefault("retrieved_chunk_count", retrieved_chunk_count)
    retrieval_metrics.setdefault("retrieved_source_count", retrieved_source_count)
    retrieval_metrics.setdefault("source_coverage", source_coverage)
    retrieval_metrics.setdefault("source_diversity", (dict(metrics.get("retrieval_metrics") or {}).get("source_diversity")))
    reranking_summary = _summarize_scores(metrics.get("reranking_scores"))
    retrieval_metrics["reranking_summary"] = reranking_summary

    # Standardized retriever metrics (0..1 when available)
    retrieval_metrics["context_relevance"] = relevance_score
    retrieval_metrics["context_recall"] = source_coverage
    if reranking_summary and reranking_summary.get("avg") is not None:
        avg_score = float(reranking_summary["avg"])
        retrieval_metrics["context_precision"] = round(avg_score / 10.0, 4) if avg_score > 1 else round(avg_score, 4)
    else:
        retrieval_metrics["context_precision"] = source_coverage

    graph_metrics = dict(metrics.get("graph_metrics") or {})
    graph_metrics.setdefault("entity_count", len(metrics.get("entities") or []))
    graph_metrics.setdefault("source_count", len(metrics.get("sources") or []))
    graph_metrics.setdefault("doc_passage_count", len(metrics.get("doc_passages") or []))
    graph_metrics.setdefault("graph_fact_count", len(metrics.get("graph_facts") or []))
    graph_metrics["graph_density_proxy"] = round(
        graph_metrics["graph_fact_count"] / max(graph_metrics["entity_count"], 1),
        4,
    )

    return {
        "answer": answer,
        "system_metrics": system_metrics,
        "generation_metrics": generation_metrics,
        "retrieval_metrics": retrieval_metrics,
        "graph_metrics": graph_metrics,
        "latency_breakdown": latency_breakdown,
        "system": system_metrics,
    }


@Controller
class LangChainController:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.router = APIRouter(prefix="/api/v1/langchain", tags=["LangChain"])
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

        class BeginTurnRequest(BaseModel):
            session_id: str
            user_content: str

        @self.router.post("/begin_turn")
        async def begin_turn(
            req: BeginTurnRequest,
            langfacade: LangChainFacade = Depends(),
        ):
            """
            Tạo một turn mới trong conversation_history.
            Trả về turn_id để frontend truyền cho cả PaCRAG và GraphRAG.
            """
            turn_id = await langfacade.PaCRAG.memory_repo.begin_turn(
                session_id=req.session_id,
                user_content=req.user_content,
            )
            return APIResponse(
                message="Turn created",
                status_code=status.HTTP_200_OK,
                data={"turn_id": turn_id},
            )

        class SaveTurnRequest(BaseModel):
            session_id: str
            user_content: str
            rag_content: Optional[str] = None
            graphrag_content: Optional[str] = None

        @self.router.post("/save_turn")
        async def save_turn(
            req: SaveTurnRequest,
            langfacade: LangChainFacade = Depends(),
        ):
            """
            Lưu 1 lượt hỏi-đáp hoàn chỉnh vào history.
            Frontend gọi sau khi đã có đủ cả 2 kết quả RAG.
            Đảm bảo chỉ 1 row duy nhất được tạo.
            """
            turn_id = await langfacade.PaCRAG.memory_repo.begin_turn(
                session_id=req.session_id,
                user_content=req.user_content,
            )
            if req.rag_content:
                await langfacade.PaCRAG.memory_repo.update_rag(
                    turn_id=turn_id, answer=req.rag_content
                )
            if req.graphrag_content:
                await langfacade.PaCRAG.memory_repo.update_graphrag(
                    turn_id=turn_id, answer=req.graphrag_content
                )
            return APIResponse(
                message="Turn saved",
                status_code=status.HTTP_200_OK,
                data={"turn_id": turn_id},
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
            graph_chunk_size: int | None = Form(None),
            graph_chunk_overlap: int | None = Form(None),
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
                    chunk_size=graph_chunk_size if graph_chunk_size is not None else child_chunk_size,
                    chunk_overlap=graph_chunk_overlap if graph_chunk_overlap is not None else child_chunk_overlap,
                )

                pac_result, graph_result = await asyncio.gather(
                    pac_task, graph_task, return_exceptions=True
                )

                errors = {}
                if isinstance(pac_result, Exception):
                    log.error(f"PaCRAG ingest failed for {file.filename}: {pac_result}", exc_info=pac_result)
                    errors["pac"] = str(pac_result)
                    pac_metrics = {}
                else:
                    pac_metrics = pac_result

                if isinstance(graph_result, Exception):
                    log.error(f"GraphRAG ingest failed for {file.filename}: {graph_result}", exc_info=graph_result)
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

                results.append({
                    **run,
                    "_upload_errors": errors or None,
                })

            return APIResponse(
                message="Comparison ingest completed",
                status_code=status.HTTP_200_OK,
                data={"runs": results},
            )

        class CompareQueryRequest(BaseModel):
            session_id: str
            run_id: str
            query: str
            source_filter: Optional[str] = None
            source_filters: Optional[List[str]] = None
            reranking_enabled: bool = False

        @self.router.post("/compare/query")
        async def compare_query(
            req: CompareQueryRequest,
            langfacade: LangChainFacade = Depends(),
        ):
            # Chạy song song cả 2 RAG — không lưu history, frontend sẽ save sau
            pac_task = langfacade.PaCRAG.retrieve_full(req.query, enable_reranking=req.reranking_enabled, source_filter=req.source_filter, source_filters=req.source_filters)
            graph_task = langfacade.GraphRAG.retrieve_with_metrics(
                req.query,
                source=req.source_filter,
                source_filters=req.source_filters,
                enable_reranking=req.reranking_enabled,
            )

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

            if pac_metrics is not None:
                pac_answer = pac_metrics.get("answer", "")
                pac_metrics["relevance_score"] = _compute_relevance_score(langfacade.embedding, req.query, pac_answer)
                pac_metrics["source_coverage"] = _compute_source_coverage(
                    pac_metrics.get("retrieved_chunks", []),
                    pac_metrics.get("retrieved_chunk_count", 0)
                )
                pac_metrics["metric_groups"] = _build_metric_groups(pac_metrics, req.query)

            if graph_metrics is not None:
                graph_answer = graph_metrics.get("answer", "")
                graph_metrics["relevance_score"] = _compute_relevance_score(langfacade.embedding, req.query, graph_answer)
                graph_metrics["source_coverage"] = _compute_source_coverage(
                    graph_metrics.get("sources", []),
                    graph_metrics.get("retrieved_chunk_count", 0)
                )
                graph_metrics["metric_groups"] = _build_metric_groups(graph_metrics, req.query)

            run = await compare_repo.create_query_run(
                req.run_id,
                pac_query=pac_metrics,
                graphrag_query=graph_metrics,
                query_text=req.query,
            )

            if run is None:
                run = await compare_repo.update_query_metrics(
                    req.run_id,
                    pac_query=pac_metrics,
                    graphrag_query=graph_metrics,
                    query_text=req.query,
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
            turn_id: Optional[str] = None
            save_history: bool = True
            source_filter: Optional[str] = None
            source_filters: Optional[List[str]] = None

        @self.router.post("/retrieve_document")
        async def retrieve_document(
            req: RetrieveDocumentRequest,
            langfacade: LangChainFacade = Depends()
        ):
            async def generate():
                try:
                    async for chunk in langfacade.PaCRAG.retrieve(
                        req.query,
                        session_id=req.session_id,
                        turn_id=req.turn_id,
                        save_user_message=req.save_history,
                        source_filter=req.source_filter,
                        source_filters=req.source_filters,
                    ):
                        if chunk:
                            yield chunk
                except Exception as e:
                    log.error(f"PaCRAG stream error: {e}", exc_info=True)
                    yield f"Lỗi hệ thống: {str(e)}"

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
            source_filters: Optional[List[str]] = None
            session_id: Optional[str] = None
            turn_id: Optional[str] = None
            save_history: bool = True

        @self.router.post("/graph/query")
        async def query_graph(
            req: GraphQueryRequest,
            langfacade: LangChainFacade = Depends()
        ):
            result = await langfacade.GraphRAG.retrieve(
                req.query,
                source=req.source,
                source_filters=req.source_filters,
                session_id=req.session_id,
                turn_id=req.turn_id,
                save_user_message=req.save_history,
            )
            return APIResponse(
                message="Query completed", status_code=status.HTTP_200_OK, data=result
            )

        class GraphMultiHopRequest(BaseModel):
            query: str
            source: Optional[str] = None
            source_filters: Optional[List[str]] = None
            max_hops: int = 2

        @self.router.post("/graph/multi-hop-query")
        async def multi_hop_query_graph(
            req: GraphMultiHopRequest,
            langfacade: LangChainFacade = Depends()
        ):
            result = await langfacade.GraphRAG.multi_hop_retrieve(
                req.query,
                max_hops=req.max_hops,
                source=req.source,
                source_filters=req.source_filters,
            )
            return APIResponse(
                message="Multi-hop query completed",
                status_code=status.HTTP_200_OK,
                data=result,
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

