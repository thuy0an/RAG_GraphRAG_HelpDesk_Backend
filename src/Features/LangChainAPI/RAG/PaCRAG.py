import logging
import time
from typing import Any, Dict, List, AsyncGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.embeddings import Embeddings
from fastapi import UploadFile
from SharedKernel.base.Metrics import Metrics
from Features.LangChainAPI.RAG.BaseRAG import BaseRAG
from Features.LangChainAPI.persistence.RedisVSRepository import RedisVSRepository
from SharedKernel.config.LLMConfig import EmbeddingFactory

log = logging.getLogger(__name__)

class PaCRAG(BaseRAG):
    def __init__(
        self,
        provider: BaseChatModel,
        embedding: Embeddings
    ) -> None:
        super().__init__(provider, embedding)
        self._redis_vs_repo = None

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

    async def retrieve(
        self,
        query: str,
        session_id: str = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Retrieve documents và generate response với streaming"""
        metrics = Metrics("Retriever")

        if session_id:
            with metrics.stage("memory_add_user"):
                await self.memory_repo.add_message(
                    session_id=session_id, role="user", content=query
                )

        with metrics.stage("hybrid_retrieval"):
            hybrid_docs = await self.redis_vs_repo.hybrid_retriver(query=query, k=5)
            print(hybrid_docs)
        metrics.increment("retrieved_docs", len(hybrid_docs))

        if not hybrid_docs:
            answer = "Không tìm thấy tài liệu liên quan. Vui lòng upload tài liệu trước hoặc đặt câu hỏi cụ thể hơn."
            if session_id:
                with metrics.stage("memory_add_assistant"):
                    await self.memory_repo.add_message(
                        session_id=session_id, role="assistant", content=answer
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
        metrics.increment("answer_tokens", len("".join(answer_parts).split()))

        answer = "".join(answer_parts)

        if session_id:
            with metrics.stage("memory_add_assistant"):
                await self.memory_repo.add_message(
                    session_id=session_id, role="assistant", content=answer
                )

        metrics.log_summary()

    async def retrieve_full(self, query: str, session_id: str = None) -> Dict[str, Any]:
        metrics = Metrics("RetrieverFull")
        start = time.perf_counter()

        if session_id:
            with metrics.stage("memory_add_user"):
                await self.memory_repo.add_message(
                    session_id=session_id, role="user", content=query
                )

        with metrics.stage("hybrid_retrieval"):
            hybrid_docs = await self.redis_vs_repo.hybrid_retriver(query=query, k=5)
        metrics.increment("retrieved_docs", len(hybrid_docs))

        if not hybrid_docs:
            answer = "Không tìm thấy tài liệu liên quan. Vui lòng upload tài liệu trước hoặc đặt câu hỏi cụ thể hơn."
            if session_id:
                with metrics.stage("memory_add_assistant"):
                    await self.memory_repo.add_message(
                        session_id=session_id, role="assistant", content=answer
                    )
            metrics.log_summary()
            total_time = time.perf_counter() - start
            return {
                "answer": answer,
                "time_total_s": round(total_time, 2),
                "answer_tokens": len(answer.split()),
            }

        with metrics.stage("context_formatting"):
            context = self._format_context_PaC(hybrid_docs)
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
            response = chain.invoke({"query": query})
            answer = response.content if hasattr(response, "content") else str(response)

        if session_id:
            with metrics.stage("memory_add_assistant"):
                await self.memory_repo.add_message(
                    session_id=session_id, role="assistant", content=answer
                )

        metrics.log_summary()

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

    async def clear_history(self, session_id: str) -> int:
        """Clear chat history for a session. Returns deleted count."""
        return await self.memory_repo.delete_session_history(session_id)

    async def get_chat_history(self, session_id: str, **kwargs) -> List[Dict]:
        """Get chat history for a session"""
        return await self.memory_repo.get_history_all(session_id)

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
