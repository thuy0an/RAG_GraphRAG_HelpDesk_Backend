import logging
import time
from typing import List

from fastapi import UploadFile
from langchain.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src.SharedKernel.base.Metrics import Metrics
from src.SharedKernel.utils.yamlenv import load_env_yaml
from src.Features.LangChainAPI.RAG.BaseRAG import BaseRAG
from src.Features.LangChainAPI.persistence.Neo4JStore import Neo4JStore
from src.Features.LangChainAPI.RAG.GraphRAGInternal import GraphRAGInternal
from SharedKernel.threading.ThreadPoolManager import ThreadPoolManager

log = logging.getLogger(__name__)


class GraphRAG(BaseRAG):
    def __init__(
        self,
        provider: BaseChatModel,
        embedding: Embeddings
    ) -> None:
        super().__init__(provider, embedding)
        self._neo4j_store = None
        self.thread_pool = ThreadPoolManager()
        self._config = load_env_yaml()
        self._internal = None

    @property
    def neo4j_store(self):
        if self._neo4j_store is None:
            self._neo4j_store = Neo4JStore(embedding_model=self.embedding)
        return self._neo4j_store

    @property
    def internal(self) -> GraphRAGInternal:
        if self._internal is None:
            self._internal = GraphRAGInternal(
                provider=self.provider,
                embedding=self.embedding,
                neo4j_store=self.neo4j_store,
                config=self._config,
            )
        return self._internal

    async def ingest(self, file: UploadFile, source: str = None, chunk_size: int = None, chunk_overlap: int = None, **kwargs) -> dict:
        """Build Lexical Graph from file using ProjectGraphRAG pipeline."""
        if not source:
            source = file.filename

        metrics = Metrics("BuildLexicalGraph")
        log.info(f"======= Starting GraphRAG Pipeline: {source} =======")

        try:
            with metrics.stage("load_documents"):
                chunks = await self.thread_pool.run_in_executor(
                    lambda f: self.internal.load_and_chunk_file(f, chunk_size=chunk_size, chunk_overlap=chunk_overlap),
                    file
                )

            if not chunks:
                raise ValueError("No text extracted from file")

            metrics.increment("chunks_count", len(chunks))
            log.info(f"✓ Loaded {len(chunks)} chunks")

            with metrics.stage("build_graph"):
                result = await self.thread_pool.run_in_executor(
                    self.internal.build_lexical_graph, chunks, source
                )

            metrics.increment("sections_count", result.get("sections", 0))
            metrics.increment("entities_count", result.get("entities", 0))
            metrics.increment("relations_count", result.get("relations", 0))

            with metrics.stage("faiss_index"):
                self.internal.upsert_faiss_index(chunks)

            metrics.log_summary()
            log.info(f"======= Completed GraphRAG Pipeline: {source} =======")

            return result

        except Exception as e:
            log.error(f"GraphRAG pipeline failed: {str(e)}")
            metrics.increment("error_count", 1)
            metrics.log_summary()
            raise

    async def index(self, file: UploadFile, **kwargs) -> None:
        """Alias for ingest to satisfy BaseRAG contract."""
        await self.ingest(file, **kwargs)

    async def retrieve(
        self,
        query: str,
        session_id: str = None,
        source: str = None,
        **kwargs
    ) -> dict:
        """Graph RAG query pipeline using ProjectGraphRAG query chain."""
        metrics = Metrics("GraphRAGQuery")
        log.info(f"Starting GraphRAG query: {query}")

        try:
            if session_id:
                with metrics.stage("memory_add_user"):
                    await self.memory_repo.add_message(
                        session_id=session_id, role="user", content=query
                    )

            with metrics.stage("embed_query"):
                query_embedding = self.embedding.embed_query(query)

            doc_ids = [self.internal._uid(source)] if source else None
            with metrics.stage("vector_search"):
                hits = self.internal.vector_search_chunks(
                    query_embedding, k=self.internal.top_k, doc_ids=doc_ids
                )

            if not hits:
                metrics.increment("empty_hits", 1)
                metrics.log_summary()
                return {
                    "answer": "Chưa có dữ liệu GraphRAG. Vui lòng upload tài liệu trước.",
                    "sources": [],
                    "entities": [],
                }

            doc_passages, section_ids, doc_ids = self.internal.collect_context(hits)

            with metrics.stage("extract_entities"):
                entities = self.internal.extract_query_entities(query)
            metrics.increment("entities_count", len(entities))

            with metrics.stage("graph_traversal"):
                graph_facts = self.internal.get_entity_subgraph(
                    entities, depth=self.internal._graph_depth
                )

            with metrics.stage("hierarchical_context"):
                section_summaries = self.internal.get_section_summaries(list(section_ids))
                doc_summary_parts = self.internal.get_document_summaries(list(doc_ids))

            with metrics.stage("llm_generation"):
                prompt = self.internal.build_answer_prompt(
                    doc_summary_parts,
                    section_summaries,
                    graph_facts,
                    doc_passages,
                    query,
                )
                response = self.provider.invoke(prompt)
                answer = response.content.strip()

            if session_id:
                with metrics.stage("memory_add_assistant"):
                    await self.memory_repo.add_message(
                        session_id=session_id, role="assistant_graphrag", content=answer
                    )

            metrics.log_summary()
            log.info("GraphRAG query completed")

            doc_id_list = list(doc_ids)
            sources = self.internal.collect_source_pages(hits, doc_id_list)
            return {"answer": answer, "sources": sources, "entities": entities}

        except Exception as e:
            log.error(f"GraphRAG query failed: {str(e)}")
            metrics.increment("error_count", 1)
            metrics.log_summary()
            raise

    async def retrieve_with_metrics(
        self,
        query: str,
        session_id: str = None,
        source: str = None,
        **kwargs
    ) -> dict:
        start = time.perf_counter()
        result = await self.retrieve(query, session_id=session_id, source=source, **kwargs)
        total_time = time.perf_counter() - start
        answer = result.get("answer", "")
        return {
            "answer": answer,
            "sources": result.get("sources", []),
            "entities": result.get("entities", []),
            "time_total_s": round(total_time, 2),
            "answer_tokens": len(answer.split()),
        }

    async def delete(self, identifier: str, **kwargs) -> None:
        """Delete document graph by filename."""
        if not identifier:
            return

        doc_id = self.internal._uid(identifier)
        self.internal.delete_document(doc_id)

    async def get_chat_history(self, session_id: str, **kwargs):
        """Get GraphRAG chat history - chỉ lấy user + assistant_graphrag messages"""
        return await self.memory_repo.get_history_all(session_id, role_filter="assistant_graphrag")

    async def clear_history(self, session_id: str) -> int:
        """Clear GraphRAG chat history for a session. Returns deleted count."""
        return await self.memory_repo.delete_session_history(session_id)
