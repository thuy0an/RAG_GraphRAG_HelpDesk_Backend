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
from src.Features.LangChainAPI.prompt import format_history_block
from src.Features.LangChainAPI.RAG.ConfidenceScorer import ConfidenceScorer
from src.Features.LangChainAPI.RAG.LLMReranker import LLMReranker
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
        source_filters: List[str] | None = None,
        turn_id: str = None,
        save_user_message: bool = True,
        enable_reranking: bool | None = None,
        **kwargs
    ) -> dict:
        """Graph RAG query pipeline. Không tự lưu history."""
        metrics = Metrics("GraphRAGQuery")
        log.info(f"Starting GraphRAG query: {query}")

        try:
            # Lấy conversation history để inject vào prompt
            history_block = ""
            limit = self._get_history_limit()
            if session_id and limit > 0:
                try:
                    turns = await self.memory_repo.get_recent_messages(session_id, limit=limit)
                    history_block = format_history_block(turns, role_key="graphrag_content")
                    log.debug(f"Injecting {len(turns)} conversation turns for session {session_id}")
                except Exception as e:
                    log.warning(f"Failed to get history for session {session_id}: {e}")
                    history_block = ""
            elif limit == 0:
                log.debug("Conversation history disabled (limit=0)")

            with metrics.stage("embed_query"):
                query_embedding = self.embedding.embed_query(query)

            doc_ids = None
            if source_filters:
                doc_ids = [self.internal._uid(s) for s in source_filters if s]
                doc_ids = list(dict.fromkeys(doc_ids)) or None
            elif source:
                doc_ids = [self.internal._uid(source)]
            with metrics.stage("vector_search"):
                hits = self.internal.vector_search_chunks(
                    query_embedding, k=self.internal.top_k, doc_ids=doc_ids
                )

            if not hits:
                metrics.increment("empty_hits", 1)
                metrics.log_summary()
                # Kiểm tra xem có document nào trong Neo4j không
                chunk_label = self.internal._label("Chunk")
                try:
                    count_result = self.neo4j_store.execute_query(
                        f"MATCH (c:{chunk_label}) RETURN COUNT(c) AS cnt"
                    )
                    chunk_count = count_result[0].get("cnt", 0) if count_result else 0
                except Exception:
                    chunk_count = 0

                if chunk_count == 0:
                    answer = "Chưa có tài liệu nào được upload vào GraphRAG. Vui lòng upload tài liệu trước."
                else:
                    answer = f"Không tìm thấy thông tin liên quan đến câu hỏi này trong {chunk_count} chunks đã lưu. Thử đặt câu hỏi theo cách khác hoặc upload thêm tài liệu liên quan."
                metric_snapshot = metrics.to_dict()
                return {
                    "answer": answer,
                    "sources": [],
                    "entities": [],
                    "graph_facts": [],
                    "retrieved_chunk_count": 0,
                    "doc_passages": [],
                    "time_total_s": 0,
                    "answer_tokens": len(answer.split()),
                    "word_count": len(answer.split()),
                    "confidence_score": None,
                    "latency_breakdown": metric_snapshot.get("timings", {}),
                    "system_metrics": {
                        "time_total_s": 0,
                        "answer_tokens": len(answer.split()),
                        "word_count": len(answer.split()),
                    },
                }

            doc_passages, section_ids, doc_ids = self.internal.collect_context(hits)
            if not doc_passages:
                metrics.increment("empty_context", 1)
                metrics.log_summary()
                metric_snapshot = metrics.to_dict()
                return {
                    "answer": "Không tìm thấy thông tin liên quan trong dữ liệu đã lưu. Thử đặt câu hỏi khác hoặc upload thêm tài liệu liên quan.",
                    "sources": [],
                    "entities": [],
                    "graph_facts": [],
                    "retrieved_chunk_count": len(hits),
                    "doc_passages": [],
                    "time_total_s": 0,
                    "answer_tokens": 0,
                    "word_count": 0,
                    "confidence_score": None,
                    "latency_breakdown": metric_snapshot.get("timings", {}),
                    "system_metrics": {
                        "time_total_s": 0,
                        "answer_tokens": 0,
                        "word_count": 0,
                    },
                }

            reranking_scores = None
            reranking_time_s = None
            if enable_reranking and doc_passages:
                with metrics.stage("reranking"):
                    reranker = LLMReranker(self.provider)
                    doc_passages, reranking_scores, reranking_time_s = await reranker.rerank(
                        query, doc_passages, top_k=len(doc_passages)
                    )

            # Chạy song song entity extraction + hierarchical context để giảm latency
            import asyncio as _asyncio

            async def _extract_entities_safe():
                try:
                    return self.internal.extract_query_entities(query)
                except Exception as e:
                    log.warning(f"Entity extraction failed: {e}")
                    return []

            async def _get_hierarchical_context():
                section_summaries = self.internal.get_section_summaries(list(section_ids))
                doc_summary_parts = self.internal.get_document_summaries(list(doc_ids))
                return section_summaries, doc_summary_parts

            with metrics.stage("extract_entities+hierarchical_context"):
                (entities, (section_summaries, doc_summary_parts)) = await _asyncio.gather(
                    _extract_entities_safe(),
                    _get_hierarchical_context(),
                )
            metrics.increment("entities_count", len(entities))

            with metrics.stage("graph_traversal"):
                graph_facts = self.internal.get_entity_subgraph(
                    entities, depth=self.internal._graph_depth
                ) if entities else []

            with metrics.stage("llm_generation"):
                prompt = self.internal.build_answer_prompt(
                    doc_summary_parts,
                    section_summaries,
                    graph_facts,
                    doc_passages,
                    query,
                    history_block=history_block,
                )
                response = self.provider.invoke(prompt)
                answer = response.content.strip()

            if doc_passages and self._is_refusal(answer):
                with metrics.stage("fallback_generation"):
                    fallback_prompt = self._build_fallback_prompt(query, doc_passages)
                    response = self.provider.invoke(fallback_prompt)
                    fallback_answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
                    if fallback_answer:
                        answer = fallback_answer

            # Nếu vẫn từ chối dù đã có passage, ép tạo câu trả lời bám sát ngữ cảnh.
            if doc_passages and self._is_refusal(answer):
                with metrics.stage("grounded_retry_generation"):
                    retry_prompt = self._build_grounded_retry_prompt(query, doc_passages)
                    response = self.provider.invoke(retry_prompt)
                    retry_answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
                    if retry_answer:
                        answer = retry_answer

            metrics.log_summary()
            log.info("GraphRAG query completed")

            doc_id_list = list(doc_ids)
            sources = self.internal.collect_source_pages(hits, doc_id_list)
            if not sources and doc_passages:
                sources = self.internal.collect_sources_from_passages(doc_passages)
            metric_snapshot = metrics.to_dict()
            return {
                "answer": answer,
                "sources": sources,
                "entities": entities,
                "retrieved_chunk_count": len(hits),
                "doc_passages": doc_passages[:10],
                "graph_facts": graph_facts[:20],
                "entity_count": len(entities),
                "source_count": len(sources),
                "doc_passage_count": len(doc_passages),
                "graph_fact_count": len(graph_facts),
                **({
                    "reranking_scores": reranking_scores,
                    "reranking_time_s": reranking_time_s,
                } if reranking_scores is not None else {}),
                "time_total_s": round(metric_snapshot.get("total_time", 0), 2),
                "answer_tokens": len(answer.split()),
                "word_count": len(answer.split()),
                "confidence_score": None,
                "latency_breakdown": metric_snapshot.get("timings", {}),
                "system_metrics": {
                    "time_total_s": round(metric_snapshot.get("total_time", 0), 2),
                    "answer_tokens": len(answer.split()),
                    "word_count": len(answer.split()),
                },
                "graph_metrics": {
                    "entity_count": len(entities),
                    "source_count": len(sources),
                    "doc_passage_count": len(doc_passages),
                    "graph_fact_count": len(graph_facts),
                },
            }

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
        source_filters: List[str] | None = None,
        turn_id: str = None,
        save_user_message: bool = True,
        enable_reranking: bool | None = None,
        **kwargs
    ) -> dict:
        start = time.perf_counter()
        result = await self.retrieve(
            query,
            source=source,
            source_filters=source_filters,
            enable_reranking=enable_reranking,
            **kwargs
        )
        total_time = time.perf_counter() - start
        answer = result.get("answer", "")
        latency_breakdown = result.get("latency_breakdown") or {}
        graph_metrics = result.get("graph_metrics") or {}

        # Confidence scoring
        confidence_score = None
        try:
            context_text = " ".join([p.get("content", "") for p in result.get("doc_passages", [])])
            scorer = ConfidenceScorer(self.provider)
            confidence_score = await scorer.score(query, context_text, answer)
        except Exception as e:
            log.warning(f"Confidence scoring failed: {e}")

        return {
            "answer": answer,
            "sources": result.get("sources", []),
            "entities": result.get("entities", []),
            "time_total_s": round(total_time, 2),
            "answer_tokens": len(answer.split()),
            "word_count": len(answer.split()),
            "retrieved_chunk_count": result.get("retrieved_chunk_count", 0),
            "doc_passages": result.get("doc_passages", []),
            "graph_facts": result.get("graph_facts", []),
            "entity_count": result.get("entity_count", len(result.get("entities", []))),
            "source_count": result.get("source_count", len(result.get("sources", []))),
            "doc_passage_count": result.get("doc_passage_count", len(result.get("doc_passages", []))),
            "graph_fact_count": result.get("graph_fact_count", len(result.get("graph_facts", []))),
            "confidence_score": confidence_score,
            "latency_breakdown": latency_breakdown,
            "system_metrics": {
                "time_total_s": round(total_time, 2),
                "answer_tokens": len(answer.split()),
                "word_count": len(answer.split()),
            },
            "graph_metrics": {
                "entity_count": graph_metrics.get("entity_count", len(result.get("entities", []))),
                "source_count": graph_metrics.get("source_count", len(result.get("sources", []))),
                "doc_passage_count": graph_metrics.get("doc_passage_count", len(result.get("doc_passages", []))),
                "graph_fact_count": graph_metrics.get("graph_fact_count", len(result.get("graph_facts", []))),
            },
            **({
                "reranking_scores": result.get("reranking_scores"),
                "reranking_time_s": result.get("reranking_time_s"),
            } if result.get("reranking_scores") is not None else {}),
        }

    def _is_refusal(self, answer: str) -> bool:
        if not answer:
            return True
        text = answer.strip().lower()
        refusal_markers = [
            "tôi không có đủ thông tin",
            "không đủ thông tin để trả lời",
            "không tìm thấy thông tin",
            "không có thông tin",
            "vui lòng liên hệ",
            "i'm sorry",
            "the provided text does not contain",
            "does not contain any information",
            "if you could provide more context",
            "i don't have enough data",
            "i don't have enough information",
            "not enough information",
            "please contact support",
        ]
        return any(marker in text for marker in refusal_markers)

    def _build_fallback_prompt(self, query: str, doc_passages: List[dict]) -> str:
        passages = [
            p.get("content", "")
            for p in doc_passages
            if isinstance(p, dict) and p.get("content")
        ]
        context = "\n\n---\n\n".join(passages)
        return (
            "Ban la tro ly AI. Hay tra loi cau hoi dua tren cac doan sau. "
            "Neu cau hoi bang tieng Viet thi bat buoc tra loi bang tieng Viet. "
            "Neu thong tin chi duoc mot phan, hay tra loi phan chac chan nhat truoc va ghi ro phan chua du. "
            "Chi tra loi 'Khong du thong tin de tra loi' khi cac doan hoan toan khong lien quan.\n\n"
            f"Doan van:\n{context}\n\n"
            f"Cau hoi: {query}\n\n"
            "Tra loi:"
        )

    def _build_grounded_retry_prompt(self, query: str, doc_passages: List[dict]) -> str:
        passages = [
            p.get("content", "")
            for p in doc_passages[:8]
            if isinstance(p, dict) and p.get("content")
        ]
        context = "\n\n---\n\n".join(passages)
        return (
            "Ban la tro ly RAG. NHIEM VU: tra loi dua tren doan trich, KHONG duoc tu choi chung chung. "
            "Neu cau hoi bang tieng Viet thi bat buoc tra loi bang tieng Viet. "
            "Neu chi co mot phan thong tin, van phai tra loi phan do va neu ro gioi han. "
            "Cam dung cac cau mo dau kieu: 'I'm sorry', 'the provided text does not contain', 'khong du thong tin' tru khi tat ca doan deu khong lien quan. "
            "Trinh bay gon: 1) Tra loi chinh 2) Bang chung tu doan trich.\n\n"
            f"Doan trich:\n{context}\n\n"
            f"Cau hoi: {query}\n\n"
            "Tra loi:"
        )

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

    async def multi_hop_retrieve(
        self,
        query: str,
        max_hops: int = 2,
        source: str = None,
        source_filters: List[str] | None = None,
    ) -> dict:
        """Multi-hop reasoning: retrieve in multiple hops, merging context."""
        import time
        start = time.perf_counter()

        all_doc_passages: list = []
        seen_contents: set = set()
        hop_count = 0

        def _merge_passages(new_passages: list) -> int:
            """Add new passages, dedup by content. Returns count of new passages added."""
            added = 0
            for p in new_passages:
                content = p.get("content", "")
                if content and content not in seen_contents:
                    seen_contents.add(content)
                    all_doc_passages.append(p)
                    added += 1
            return added

        # Hop 1: retrieve with original query
        result_hop1 = await self.retrieve(query, source=source, source_filters=source_filters)
        hop_count = 1
        entities_hop1 = result_hop1.get("entities", [])
        _merge_passages(result_hop1.get("doc_passages", []))

        # Hop 2: use entities from hop 1 as sub-query
        if max_hops >= 2 and entities_hop1:
            hop2_query = " ".join(entities_hop1[:5])
            result_hop2 = await self.retrieve(hop2_query, source=source, source_filters=source_filters)
            new_added = _merge_passages(result_hop2.get("doc_passages", []))
            if new_added > 0:
                hop_count = 2

        # Generate final answer with merged context
        final_answer = result_hop1.get("answer", "")
        if hop_count == 2 and all_doc_passages:
            try:
                # Build a new prompt with merged passages
                merged_prompt = self.internal.build_answer_prompt(
                    doc_summary_parts=[],
                    section_summaries=[],
                    graph_facts=[],
                    doc_passages=all_doc_passages[:10],
                    question=query,
                )
                response = self.provider.invoke(merged_prompt)
                final_answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
            except Exception as e:
                log.warning(f"Multi-hop final generation failed, using hop1 answer: {e}")

        total_time = time.perf_counter() - start
        sources = result_hop1.get("sources", [])

        return {
            "answer": final_answer,
            "hop_count": hop_count,
            "sources": sources,
            "doc_passages": all_doc_passages[:10],
            "time_total_s": round(total_time, 2),
            "entities": entities_hop1,
        }
