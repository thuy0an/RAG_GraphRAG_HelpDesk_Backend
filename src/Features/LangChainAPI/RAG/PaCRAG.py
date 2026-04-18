import logging
import time
from typing import Any, Dict, List, AsyncGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.embeddings import Embeddings
from fastapi import UploadFile, Depends, HTTPException
from SharedKernel.base.Metrics import Metrics
from Features.LangChainAPI.RAG.BaseRAG import BaseRAG
from Features.LangChainAPI.persistence.RedisVSRepository import RedisVSRepository
from Features.ConversationAPI.ConversationService import ConversationService
from Features.ConversationAPI.ConversationRepository import ConversationRepository
from SharedKernel.config.LLMConfig import EmbeddingFactory
from sqlmodel.ext.asyncio.session import AsyncSession
from SharedKernel.persistence.PersistenceManager import get_db_session
from src.SharedKernel.exception.APIException import APIException

log = logging.getLogger(__name__)

class PaCRAG(BaseRAG):
    def __init__(
        self,
        provider: BaseChatModel,
        embedding: Embeddings,
        conversation_service: ConversationService = None
    ) -> None:
        super().__init__(provider, embedding)
        self._redis_vs_repo = None
        self.conversation_service = conversation_service

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

    async def index_with_kwargs(self, 
        file: UploadFile, 
        parent_chunk_size=None, 
        parent_chunk_overlap=None, 
        child_chunk_size=None, 
        child_chunk_overlap=None
    ) -> Dict:
        """Index file with custom kwargs - similar to index_with_metrics"""
        
        metrics = Metrics("Index with kwargs")
        start = time.perf_counter()

        with metrics.stage("delete_existing"):
            await self.redis_vs_repo.delete_documents_by_metadata(
                {"source": file.filename}
            )

        with metrics.stage("load_document"):
            docs = await self.loader.load_file(file)
            log.info(docs)

        if not docs:
            return {
                "time_total_s": 0,
                "parent_chunks": 0,
                "child_chunks": 0,
            }

        with metrics.stage("split_pac"):
            chunks = await self.process.split_PaC(
                docs,
                parent_chunk_size=parent_chunk_size,
                parent_chunk_overlap=parent_chunk_overlap,
                child_chunk_size=child_chunk_size,
                child_chunk_overlap=child_chunk_overlap,
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

    async def retrieve(
        self,
        query: str,
        session_id: str = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Retrieve documents và generate response với streaming"""
        # Call internal logic with streaming
        metrics = Metrics("Retriever")
        start = time.perf_counter()

        if not session_id:
            raise APIException(messsage="session_id is required for PaCRAG query", status_code=400)

        with metrics.stage("memory_add_user"):
            log.info(f"PaCRAG DEBUG: session_id={session_id}, conversation_service={self.conversation_service}")
            if self.conversation_service:
                await self.conversation_service.add_conversation_history(
                    session_id=session_id,
                    role="adv-user",
                    content=query
                )
                log.info(f"PaCRAG DEBUG: User message added successfully")
            else:
                log.warning(f"PaCRAG DEBUG: conversation_service is None, cannot add user message")

        with metrics.stage("hybrid_retrieval"):
            hybrid_docs = await self.redis_vs_repo.hybrid_retriver(query=query, k=5)
            print(hybrid_docs)
        metrics.increment("retrieved_docs", len(hybrid_docs))

        if not hybrid_docs:
            answer = "Không tìm thấy tài liệu liên quan. Vui lòng upload tài liệu trước hoặc đặt câu hỏi cụ thể hơn."
            if session_id:
                with metrics.stage("memory_add_assistant"):
                    if self.conversation_service:
                        await self.conversation_service.add_conversation_history(
                            session_id=session_id,
                            role="adv-assistant",
                            content=answer
                        )
            metrics.log_summary()
            yield answer
            return

        with metrics.stage("context_formatting"):
            context = self._format_context_PaC(hybrid_docs)
            print(context)
        metrics.increment("context_length", len(context))

        with metrics.stage("prompt_building"):
            system_prompt = """
            Bạn là trợ lý AI chuyên nghiệp

            ## Quy tắc xử lý

            1. Trường hợp người dùng chỉ chào
            Nếu người dùng chỉ gửi lời chào (ví dụ: "xin chào", "hello", "hi", ...):

            - Chỉ chào lại một cách lịch sự.
            - **KHÔNG trả lời nội dung.**
            - **KHÔNG sử dụng ngữ cảnh.**
            - **KHÔNG hiển thị nguồn.**

            2. Trong trường hợp người dùng gửi câu hỏi thì:
            Hãy trả lời câu hỏi của người dùng dựa trên context

            YÊU CẦU BẮT BUỘC:
            1. Tuân theo quy tắc xử lý
            2. Trả lời câu hỏi dựa trên ngữ cảnh
            3. KẾT THÚC câu trả lời với 3 dòng thông tin nguồn:

            Trong ngữ cảnh có metadata ở cuối mỗi tài liệu với định dạng:
            Source: <tên file>, Page: <trang>

            Hãy trích xuất thông tin từ metadata này và trình bày lại theo định dạng sau:

            - Nguồn: <tên file>
            - Trang: <không xác định nếu không có thông tin>

            QUAN TRỌNG:
            - Chỉ sử dụng thông tin từ metadata.
            - Nếu không có thông tin trang thì ghi: "không xác định".
            - Không sử dụng định dạng khác.

            Ví dụ output:

            - Nguồn: example.pdf
            - Trang: không xác định
            """

            template = f"""{system_prompt}

            Ngữ cảnh: {context}

            Câu hỏi: {query}

            Hãy trả lời câu hỏi dựa trên ngữ cảnh

            Lưu ý nếu không tìm thấy thông tin thì output: tôi không có thông tin vui lòng liên hệ bộ phận hỗ trợ
            """
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.provider

        with metrics.stage("llm_generation"):
            answer_parts = []
            async for chunk in chain.astream({"query": query}):
                if hasattr(chunk, "content"):
                    answer_parts.append(chunk.content)
                    yield chunk.content
            answer = "".join(answer_parts)
            metrics.increment("answer_tokens", len(answer.split()))

        with metrics.stage("memory_add_assistant"):
            if self.conversation_service:
                await self.conversation_service.add_conversation_history(
                    session_id=session_id,
                    role="adv-assistant",
                    content=answer
                )

        metrics.log_summary()
        log.info("GraphRAG query completed")

    async def retrieve_with_metrics(self,
        query: str,
        session_id: str = None
    ) -> Dict[str, Any]:
        """Retrieve documents và generate response với dict metrics"""
        start = time.perf_counter()
        answer_parts = []
        async for chunk in self.retrieve(query, session_id=session_id):
            answer_parts.append(chunk)
        answer = "".join(answer_parts)
        total_time = time.perf_counter() - start
        token_count = len(answer.split())
        return {
            "answer": answer,
            "time_total_s": round(total_time, 2),
            "answer_tokens": token_count,
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

    async def clear_history(self, session_id: str) -> bool:
        """Clear chat history for a session. Returns success status."""
        return await self.conversation_service.clear_conversation_history(session_id)

    async def get_chat_history(self, session_id: str, **kwargs) -> List[Dict]:
        """Get chat history for a session"""
        return await self.conversation_service.get_conversation_history(session_id)

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
