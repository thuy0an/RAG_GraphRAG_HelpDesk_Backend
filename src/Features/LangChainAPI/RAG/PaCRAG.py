import logging
import time
from typing import Any, Dict, List, Optional, AsyncGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.embeddings import Embeddings
from fastapi import UploadFile
from SharedKernel.base.Metrics import Metrics
from Features.LangChainAPI.RAG.BaseRAG import BaseRAG
from Features.LangChainAPI.RAG.LLMReranker import LLMReranker
from Features.LangChainAPI.RAG.ConfidenceScorer import ConfidenceScorer
from Features.LangChainAPI.persistence.RedisVSRepository import RedisVSRepository
from SharedKernel.config.LLMConfig import EmbeddingFactory
from Features.LangChainAPI.prompt import PaC_template, PaC_template_with_history, format_history_block

log = logging.getLogger(__name__)

class PaCRAG(BaseRAG):
    def __init__(
        self,
        provider: BaseChatModel,
        embedding: Embeddings,
        enable_reranking: bool = False,
    ) -> None:
        super().__init__(provider, embedding)
        self._redis_vs_repo = None
        self.enable_reranking = enable_reranking
        self._reranker = LLMReranker(provider) if enable_reranking else None

    @property
    def redis_vs_repo(self):
        if self._redis_vs_repo is None:
            self._redis_vs_repo = RedisVSRepository(EmbeddingFactory)
        return self._redis_vs_repo

    async def index(self, file: UploadFile, **kwargs) -> None:
        """Ingest PDF/DOCX file vào Redis vector store với Page-aware Chunking"""
        await self.index_with_metrics(file, **kwargs)

    async def index_with_metrics(self, file: UploadFile, **kwargs) -> Dict:
        metrics = Metrics("Index PDF")
        start = time.perf_counter()
        try:
            with metrics.stage("delete_existing"):
                await self.redis_vs_repo.delete_documents_by_metadata(
                    {"source": file.filename}
                )

            with metrics.stage("load_document"):
                docs = await self.loader.load_file(file)

            if not docs:
                return {
                    "time_total_s": 0,
                    "parent_chunks": 0,
                    "child_chunks": 0,
                }

            with metrics.stage("split_pac"):
                chunks = await self.process.split_PaC(
                    docs,
                    parent_chunk_size=kwargs.get("parent_chunk_size"),
                    parent_chunk_overlap=kwargs.get("parent_chunk_overlap"),
                    child_chunk_size=kwargs.get("child_chunk_size"),
                    child_chunk_overlap=kwargs.get("child_chunk_overlap"),
                )

            with metrics.stage("add_documents"):
                await self.redis_vs_repo.add_PaC_documents(chunks)

            metrics.log_summary()
            total_time = time.perf_counter() - start

            parent_count = len(chunks.get("parent", [])) if isinstance(chunks, dict) else 0
            child_count = len(chunks.get("children", [])) if isinstance(chunks, dict) else 0

            return {
                "time_total_s": round(total_time, 2),
                "parent_chunks": parent_count,
                "child_chunks": child_count,
            }
        except Exception:
            log.error(
                "PaCRAG ingest failed for %s",
                file.filename or "<unknown>",
                exc_info=True,
            )
            metrics.increment("error_count", 1)
            metrics.log_summary()
            raise

    async def retrieve(
        self,
        query: str,
        session_id: str = None,
        turn_id: str = None,
        save_user_message: bool = True,
        source_filter: Optional[str] = None,
        source_filters: Optional[List[str]] = None,
        enable_reranking: Optional[bool] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Retrieve documents và generate response với streaming. Không tự lưu history."""
        metrics = Metrics("Retriever")

        # Lấy conversation history để inject vào prompt (1 lần duy nhất)
        history_block = ""
        limit = self._get_history_limit()
        if session_id and limit > 0:
            try:
                turns = await self.memory_repo.get_recent_messages(session_id, limit=limit)
                history_block = format_history_block(turns, role_key="rag_content")
                log.debug(f"Injecting {len(turns)} conversation turns for session {session_id}")
            except Exception as e:
                log.warning(f"Failed to get history for session {session_id}: {e}")
                history_block = ""
        elif limit == 0:
            log.debug("Conversation history disabled (limit=0)")

        with metrics.stage("hybrid_retrieval"):
            hybrid_docs = await self.redis_vs_repo.hybrid_retriver(query=query, k=5, source_filter=source_filter, source_filters=source_filters)
            print(hybrid_docs)
        metrics.increment("retrieved_docs", len(hybrid_docs))

        # Re-ranking (optional)
        _do_rerank = enable_reranking if enable_reranking is not None else self.enable_reranking
        if _do_rerank and hybrid_docs:
            if self._reranker is None:
                self._reranker = LLMReranker(self.provider)
            hybrid_docs, _, _ = await self._reranker.rerank(query, hybrid_docs, top_k=5)

        if not hybrid_docs:
            answer = "Không tìm thấy tài liệu liên quan. Vui lòng upload tài liệu trước hoặc đặt câu hỏi cụ thể hơn."
            metrics.log_summary()
            yield answer
            return

        with metrics.stage("context_formatting"):
            context = self._format_context_PaC(hybrid_docs)
            print(context)
        metrics.increment("context_length", len(context))

        with metrics.stage("prompt_building"):
            template = PaC_template_with_history(context, history_block)
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.provider

        with metrics.stage("llm_generation"):
            answer_parts = []
            try:
                async for chunk in chain.astream({"query": query}):
                    if hasattr(chunk, "content"):
                        answer_parts.append(chunk.content)
                        yield chunk.content
            except Exception as e:
                log.error(f"LLM streaming failed: {e}")
                yield "Lỗi khi sinh câu trả lời. Vui lòng thử lại."
                return
        metrics.increment("answer_tokens", len("".join(answer_parts).split()))
        metrics.log_summary()

    async def retrieve_full(
        self,
        query: str,
        session_id: str = None,
        turn_id: str = None,
        save_user_message: bool = True,
        source_filter: Optional[str] = None,
        source_filters: Optional[List[str]] = None,
        enable_reranking: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Non-streaming version. Không tự lưu history."""
        metrics = Metrics("RetrieverFull")
        start = time.perf_counter()

        with metrics.stage("hybrid_retrieval"):
            hybrid_docs = await self.redis_vs_repo.hybrid_retriver(query=query, k=5, source_filter=source_filter, source_filters=source_filters)
        metrics.increment("retrieved_docs", len(hybrid_docs))

        # Re-ranking (optional)
        reranking_scores = None
        reranking_time_s = None
        _do_rerank = enable_reranking if enable_reranking is not None else self.enable_reranking
        if _do_rerank and hybrid_docs:
            if self._reranker is None:
                self._reranker = LLMReranker(self.provider)
            hybrid_docs, reranking_scores, reranking_time_s = await self._reranker.rerank(
                query, hybrid_docs, top_k=5
            )

        if not hybrid_docs:
            answer = "Không tìm thấy tài liệu liên quan. Vui lòng upload tài liệu trước hoặc đặt câu hỏi cụ thể hơn."
            metrics.log_summary()
            total_time = time.perf_counter() - start
            return {
                "answer": answer,
                "time_total_s": round(total_time, 2),
                "answer_tokens": len(answer.split()),
                "word_count": 0,
                "retrieved_chunk_count": 0,
                "retrieved_chunks": [],
                "confidence_score": None,
            }

        with metrics.stage("context_formatting"):
            context = self._format_context_PaC(hybrid_docs)
        metrics.increment("context_length", len(context))

        with metrics.stage("prompt_building"):
            template = PaC_template(context)
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.provider

        with metrics.stage("llm_generation"):
            response = await chain.ainvoke({"query": query})
            answer = response.content if hasattr(response, "content") else str(response)

        # Confidence scoring
        confidence_score = None
        try:
            scorer = ConfidenceScorer(self.provider)
            confidence_score = await scorer.score(query, context, answer)
        except Exception as e:
            log.warning(f"Confidence scoring failed: {e}")

        # Trích xuất retrieved_chunks (tối đa 10)
        retrieved_chunks = []
        for doc in hybrid_docs[:10]:
            metadata = doc.get("metadata", {})
            pages = metadata.get("pages") or []
            if not pages and metadata.get("page_number"):
                pages = [metadata["page_number"]]
            retrieved_chunks.append({
                "content": doc.get("content", ""),
                "filename": metadata.get("source", ""),
                "pages": pages,
            })

        metrics.log_summary()
        total_time = time.perf_counter() - start
        return {
            "answer": answer,
            "time_total_s": round(total_time, 2),
            "answer_tokens": len(answer.split()),
            "word_count": len(answer.split()),
            "retrieved_chunk_count": len(hybrid_docs),
            "retrieved_chunks": retrieved_chunks,
            "confidence_score": confidence_score,
            **({"reranking_scores": reranking_scores, "reranking_time_s": reranking_time_s} if reranking_scores is not None else {}),
        }

    async def delete(self, identifier: str, **kwargs) -> None:
        """Delete documents theo file_name"""
        await self.redis_vs_repo.delete_documents_by_metadata({"source": identifier})

    async def clear_vector_store(self, source: str | None = None) -> None:
        """Clear vector store data for a source or all documents."""
        if source:
            await self.redis_vs_repo.delete_documents_by_metadata({"source": source})
        else:
            await self.redis_vs_repo.delete_all_documents()

    async def clear_history(self, session_id: str) -> int:
        """Clear chat history for a session. Returns deleted count."""
        return await self.memory_repo.delete_session_history(session_id)

    async def get_chat_history(self, session_id: str, **kwargs) -> List[Dict]:
        """Get chat history for a session - chỉ lấy user + assistant_rag messages"""
        return await self.memory_repo.get_history_all(session_id, role_filter="assistant_rag")

    def _format_context_PaC(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results thành context string cho PaC"""
        if not search_results:
            return ""

        context_parts = []
        seen_parents = set()

        for idx, result in enumerate(search_results):
            parent_id = result.get("id")
            if not parent_id:
                continue

            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            parent_content = result.get("content", "")
            parent_metadata = result.get("metadata", {})

            file_name = parent_metadata.get("source", "")
            pages = parent_metadata.get("pages", [])
            page_number = parent_metadata.get("page_number")
            page_span = parent_metadata.get("page_span")

            if pages:
                pages_str = ", ".join([str(p) for p in pages])
            elif page_span:
                pages_str = str(page_span)
            elif page_number:
                pages_str = str(page_number)
            else:
                pages_str = "không xác định"

            doc_content = parent_content.replace("\n", " ").strip()

            metadata_info = []
            if file_name:
                metadata_info.append(f"Source: {file_name}")
            metadata_info.append(f"Page: {pages_str}")

            doc_content = doc_content + "\n" + (" | ".join(metadata_info))
            context_parts.append(doc_content)

        formatted_context = "\n\n".join(context_parts)
        formatted_context = formatted_context.replace("{", "{{").replace("}", "}}")
        return formatted_context
