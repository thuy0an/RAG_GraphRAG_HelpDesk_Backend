import hashlib
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import UploadFile
from langchain.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.Features.LangChainAPI.persistence.Neo4JStore import Neo4JStore

log = logging.getLogger(__name__)


class GraphRAGInternal:
    _SECTION_EXTRACT_PROMPT = """Phân tích đoạn văn bản sau và trả về một JSON object với các trường:
1. "title": tiêu đề ngắn của đoạn (tối đa 8 từ)
2. "summary": tóm tắt ngắn gọn (2-3 câu)
3. "entities": danh sách các thực thể với tên và loại
4. "relations": danh sách các bộ ba [chủ thể, quan hệ, đối tượng]

Loại thực thể: PERSON, ORG, CONCEPT, SKILL, PLACE, OTHER
Chuẩn hóa tên thực thể về chữ thường.

Chỉ trả về JSON hợp lệ:
{{
    "title": "...",
    "summary": "...",
    "entities": [{{"name": "...", "type": "..."}}],
    "relations": [["chủ thể", "quan hệ", "đối tượng"]]
}}

Văn bản:
{text}

JSON:"""

    _DOC_SUMMARY_PROMPT = """Tóm tắt tài liệu sau trong 3-5 câu bằng tiếng Việt.
Tập trung vào các chủ đề chính, thực thể quan trọng và mục đích tổng thể của tài liệu.

Các đoạn của tài liệu:
{sections}

Tóm tắt:"""

    _ENTITY_EXTRACT_PROMPT = """Liệt kê các thực thể chính (người, tổ chức, khái niệm, kỹ năng, địa điểm) trong câu hỏi sau.
Chỉ trả về một JSON array, ví dụ: ["thực thể 1", "thực thể 2"]. Nếu không có, trả về [].

Câu hỏi: {question}
Thực thể:"""

    _ANSWER_PROMPT = """Bạn là trợ lý AI chuyên nghiệp. Trả lời câu hỏi dựa trên ngữ cảnh bên dưới.
Trả lời bằng tiếng Việt. Nếu câu hỏi bằng tiếng Anh, trả lời bằng tiếng Anh.
Không sử dụng bất kỳ ngôn ngữ nào khác.

Hướng dẫn:
- Nếu câu hỏi là lời chào: chỉ chào lại ngắn gọn, không dùng ngữ cảnh.
- Nếu là câu hỏi tiếp nối: sử dụng lịch sử hội thoại để hiểu ngữ cảnh.
- Nếu là câu hỏi thực sự: trả lời đầy đủ dựa trên ngữ cảnh, ưu tiên dữ liệu tài liệu hơn suy diễn.
- Sau phần trả lời, thêm nguồn theo định dạng:
    - Nguồn: <tên file>
    - Trang: <số trang>
- Nếu hoàn toàn không có dữ liệu liên quan trong ngữ cảnh: trả lời "Tôi không có thông tin về vấn đề này, vui lòng liên hệ bộ phận hỗ trợ."

=== Tổng quan tài liệu ===
{doc_summary}

=== Tóm tắt các đoạn (ngữ cảnh phân cấp) ===
{section_context}

=== Đồ thị tri thức (quan hệ thực thể) ===
{graph_context}

=== Đoạn văn bản liên quan ===
{doc_context}

{history_section}Câu hỏi: {question}

Trả lời:"""

    def __init__(
        self,
        provider: BaseChatModel,
        embedding: Embeddings,
        neo4j_store: Neo4JStore,
        config: object,
    ) -> None:
        self.provider = provider
        self.embedding = embedding
        self.neo4j_store = neo4j_store
        self._config = config
        self._faiss_index = None
        self._faiss_loaded = False
        self._init_graph_rag_settings()
        self._ensure_indexes()

    @property
    def top_k(self) -> int:
        return self._top_k

    def _init_graph_rag_settings(self) -> None:
        graph_cfg = getattr(self._config, "graph_rag", None)

        self._chunk_size = getattr(graph_cfg, "chunk_size", 800) if graph_cfg else 800
        self._chunk_overlap = getattr(graph_cfg, "chunk_overlap", 100) if graph_cfg else 100
        self._section_size = getattr(graph_cfg, "section_size", 6) if graph_cfg else 6
        self._top_k = getattr(graph_cfg, "top_k", 8) if graph_cfg else 8
        self._graph_depth = getattr(graph_cfg, "graph_depth", 2) if graph_cfg else 2

        default_index_dir = Path("specs/data/graph_rag/faiss_index")
        index_dir = getattr(graph_cfg, "faiss_index_dir", None) if graph_cfg else None
        self._faiss_index_dir = Path(index_dir) if index_dir else default_index_dir
        self._faiss_index_dir.parent.mkdir(parents=True, exist_ok=True)

        label_prefix = getattr(graph_cfg, "label_prefix", "GR") if graph_cfg else "GR"
        self._label_prefix = self._sanitize_prefix(label_prefix)
        self._vector_index_name = f"{self._label_prefix.lower()}_chunk_embedding"

    def _embedding_dim(self) -> int:
        try:
            return len(self.embedding.embed_query("dimension_probe"))
        except Exception:
            return 768

    def _ensure_indexes(self) -> None:
        """Ensure Neo4j constraints and vector index exist for GraphRAG labels."""
        doc_label = self._label("Document")
        section_label = self._label("Section")
        chunk_label = self._label("Chunk")
        entity_label = self._label("Entity")

        constraints = [
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (d:{doc_label}) REQUIRE d.id IS UNIQUE",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (s:{section_label}) REQUIRE s.id IS UNIQUE",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (c:{chunk_label}) REQUIRE c.id IS UNIQUE",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (e:{entity_label}) REQUIRE e.id IS UNIQUE",
        ]

        for stmt in constraints:
            try:
                self.neo4j_store.execute_query(stmt)
            except Exception:
                pass

        dims = self._embedding_dim()
        try:
            self.neo4j_store.execute_query(
                f"""
                CREATE VECTOR INDEX {self._vector_index_name} IF NOT EXISTS
                FOR (c:{chunk_label}) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {dims},
                  `vector.similarity_function`: 'cosine'
                }}}}
                """
            )
        except Exception:
            pass

    def _sanitize_prefix(self, prefix: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_]", "", prefix or "")
        return cleaned or "GR"

    def _label(self, base: str) -> str:
        prefix = self._label_prefix
        if prefix and not prefix.endswith("_"):
            prefix = f"{prefix}_"
        return f"{prefix}{base}"

    def _uid(self, *parts: str) -> str:
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def _normalize_entity(self, name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())

    def _call_json(self, prompt: str) -> dict:
        try:
            response = self.provider.invoke(prompt)
            match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            return {}
        return {}

    def load_and_chunk_file(self, file: UploadFile, chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
        suffix = Path(file.filename or "").suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix or ".pdf", delete=False) as temp_file:
            file.file.seek(0)
            temp_file.write(file.file.read())
            temp_path = Path(temp_file.name)

        try:
            docs = self._load_document(temp_path)
            if not docs:
                return []

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size or self._chunk_size,
                chunk_overlap=chunk_overlap or self._chunk_overlap,
            )
            chunks = splitter.split_documents(docs)

            for i, chunk in enumerate(chunks):
                chunk.metadata["source_file"] = file.filename or temp_path.name
                chunk.metadata["chunk_index"] = i
                page_number = chunk.metadata.get("page_number")
                if page_number is None:
                    page_number = chunk.metadata.get("page")
                fallback_page = False
                if page_number is None:
                    page_number = 1
                    fallback_page = True
                if isinstance(page_number, int) and page_number >= 0 and not fallback_page:
                    page_number = page_number + 1
                chunk.metadata["page_number"] = page_number

            return chunks
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def _load_document(self, file_path: Path) -> List[Document]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            try:
                from langchain_community.document_loaders import PyMuPDFLoader
                loader = PyMuPDFLoader(str(file_path))
            except Exception:
                from langchain_community.document_loaders import UnstructuredPDFLoader
                loader = UnstructuredPDFLoader(str(file_path), mode="single")
        elif suffix in (".doc", ".docx"):
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        return loader.load()

    def _group_into_sections(self, chunks: List[Document]) -> List[List[Document]]:
        sections = []
        for i in range(0, len(chunks), self._section_size):
            sections.append(chunks[i : i + self._section_size])
        return sections

    def _extract_section(self, text: str) -> dict:
        data = self._call_json(self._SECTION_EXTRACT_PROMPT.format(text=text[:3000]))
        return {
            "title": data.get("title", "Section"),
            "summary": data.get("summary", ""),
            "entities": [
                {
                    "name": self._normalize_entity(e.get("name", "")),
                    "type": e.get("type", "CONCEPT"),
                }
                for e in data.get("entities", [])
                if isinstance(e, dict) and e.get("name")
            ],
            "relations": [
                (
                    self._normalize_entity(str(r[0])),
                    str(r[1]).strip(),
                    self._normalize_entity(str(r[2])),
                )
                for r in data.get("relations", [])
                if isinstance(r, list) and len(r) == 3 and r[0] and r[2]
            ],
        }

    def _doc_summary(self, section_summaries: List[str]) -> str:
        joined = "\n\n".join(section_summaries[:10])
        try:
            response = self.provider.invoke(
                self._DOC_SUMMARY_PROMPT.format(sections=joined)
            )
            return response.content.strip()
        except Exception:
            return joined[:500]

    def build_lexical_graph(self, chunks: List[Document], filename: str) -> dict:
        t_total_start = time.perf_counter()

        embeddings = self.embedding.embed_documents([c.page_content for c in chunks])
        section_groups = self._group_into_sections(chunks)

        section_summaries: List[str] = []
        total_entities = 0
        total_relations = 0

        doc_id = self._uid(filename)
        self._upsert_document(doc_id=doc_id, filename=filename, summary="")
        for sec_idx, sec_chunks in enumerate(section_groups):
            sec_text = "\n\n".join(c.page_content for c in sec_chunks)
            sec_data = self._extract_section(sec_text)

            section_id = self._uid(doc_id, str(sec_idx))
            section_summaries.append(f"{sec_data['title']}: {sec_data['summary']}")

            self._upsert_section(
                section_id=section_id,
                doc_id=doc_id,
                index=sec_idx,
                title=sec_data["title"],
                summary=sec_data["summary"],
                text=sec_text[:1000],
            )
            if sec_idx > 0:
                self._link_sections(self._uid(doc_id, str(sec_idx - 1)), section_id)

            prev_chunk_id: Optional[str] = None
            for chunk in sec_chunks:
                chunk_idx = chunk.metadata.get("chunk_index", 0)
                chunk_id = self._uid(doc_id, str(chunk_idx))
                chunk.metadata["doc_id"] = doc_id
                chunk.metadata["section_id"] = section_id
                chunk.metadata["chunk_id"] = chunk_id
                page_number = chunk.metadata.get("page_number")
                embedding = embeddings[chunk_idx] if chunk_idx < len(embeddings) else []
                self._upsert_chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    section_id=section_id,
                    index=chunk_idx,
                    text=chunk.page_content,
                    page_number=page_number,
                    embedding=embedding,
                )
                if prev_chunk_id:
                    self._link_chunks(prev_chunk_id, chunk_id)
                prev_chunk_id = chunk_id

            for ent in sec_data["entities"]:
                ent_id = self._uid(ent["name"])
                self._upsert_entity(ent_id, ent["name"], ent["type"])
                for chunk in sec_chunks:
                    chunk_id = self._uid(doc_id, str(chunk.metadata.get("chunk_index", 0)))
                    self._link_chunk_entity(chunk_id, ent_id)
            total_entities += len(sec_data["entities"])

            for subj, rel, obj in sec_data["relations"]:
                src_id = self._uid(subj)
                tgt_id = self._uid(obj)
                self._upsert_entity(src_id, subj)
                self._upsert_entity(tgt_id, obj)
                self._link_entities(src_id, tgt_id, rel)
            total_relations += len(sec_data["relations"])

        doc_summary = self._doc_summary(section_summaries)
        self._upsert_document(doc_id=doc_id, filename=filename, summary=doc_summary)

        t_total = time.perf_counter() - t_total_start
        return {
            "chunks": len(chunks),
            "sections": len(section_groups),
            "entities": total_entities,
            "relations": total_relations,
            "time_total_s": round(t_total, 2),
        }

    def upsert_faiss_index(self, chunks: List[Document]) -> None:
        if not chunks:
            return

        faiss_store = self._get_faiss_store()
        if faiss_store is None or faiss_store.index.ntotal == 0:
            # Chưa có index hoặc index rỗng → tạo mới từ chunks
            faiss_store = self._rebuild_faiss_index(chunks)
            if faiss_store is None:
                log.warning("FAISS not available, skipping FAISS index creation")
                return
        else:
            faiss_store.add_documents(chunks)

        self._faiss_index_dir.mkdir(parents=True, exist_ok=True)
        faiss_store.save_local(str(self._faiss_index_dir))
        self._faiss_index = faiss_store
        log.info(f"FAISS index saved to {self._faiss_index_dir} ({faiss_store.index.ntotal} vectors)")

    def _get_faiss_store(self):
        if self._faiss_loaded:
            return self._faiss_index

        self._faiss_loaded = True
        try:
            from langchain_community.vectorstores import FAISS
        except Exception:
            log.warning("FAISS not available; falling back to Neo4j cosine search")
            return None

        if self._faiss_index_dir.exists():
            try:
                self._faiss_index = FAISS.load_local(
                    str(self._faiss_index_dir),
                    self.embedding,
                    allow_dangerous_deserialization=True,
                )
                log.info(f"FAISS index loaded from {self._faiss_index_dir} ({self._faiss_index.index.ntotal} vectors)")
                return self._faiss_index
            except Exception as e:
                log.warning(f"Failed to load FAISS index from {self._faiss_index_dir}: {e}")

        log.info(f"No FAISS index found at {self._faiss_index_dir}, will use Neo4j vector search")
        self._faiss_index = None
        return self._faiss_index

    def _rebuild_faiss_index(self, chunks: List[Document]):
        try:
            from langchain_community.vectorstores import FAISS
        except Exception:
            return None

        return FAISS.from_documents(chunks, self.embedding)

    def _vector_index_search(self, query_embedding: List[float], k: int = 8, doc_ids: Optional[List[str]] = None) -> List[Dict]:
        try:
            if doc_ids:
                results = self.neo4j_store.execute_query(
                    f"""
                    CALL db.index.vector.queryNodes('{self._vector_index_name}', $k, $embedding)
                    YIELD node AS c, score
                    WHERE c.doc_id IN $doc_ids
                    RETURN c.id AS chunk_id, c.text AS text, c.doc_id AS doc_id,
                              c.section_id AS section_id, c.index AS chunk_index,
                              c.page_number AS page_number, score
                    """,
                    {"k": k * 5, "embedding": query_embedding, "doc_ids": doc_ids},
                )
                return results[:k]

            return self.neo4j_store.execute_query(
                f"""
                CALL db.index.vector.queryNodes('{self._vector_index_name}', $k, $embedding)
                YIELD node AS c, score
                RETURN c.id AS chunk_id, c.text AS text, c.doc_id AS doc_id,
                         c.section_id AS section_id, c.index AS chunk_index,
                         c.page_number AS page_number, score
                """,
                {"k": k, "embedding": query_embedding},
            )
        except Exception:
            return []

    def vector_search_chunks(self, query_embedding: List[float], k: int = 8, doc_ids: Optional[List[str]] = None) -> List[Dict]:
        # 1. Thử FAISS trước (nhanh nhất)
        faiss_store = self._get_faiss_store()
        if faiss_store and faiss_store.index.ntotal > 0:
            log.debug(f"Using FAISS index for search (doc_ids={doc_ids})")
            results = faiss_store.similarity_search_with_score_by_vector(query_embedding, k=k * 5)
            hits = []
            for doc, score in results:
                if doc_ids and doc.metadata.get("doc_id") not in doc_ids:
                    continue
                hits.append({
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "text": doc.page_content,
                    "doc_id": doc.metadata.get("doc_id"),
                    "section_id": doc.metadata.get("section_id"),
                    "page_number": doc.metadata.get("page_number"),
                    "score": score,
                })
                if len(hits) >= k:
                    break
            if hits:
                log.info(f"FAISS returned {len(hits)} hits")
                return hits
            log.warning("FAISS returned 0 hits (index may be stale or doc_ids filter too strict), falling back to Neo4j")

        # 2. Fallback: Neo4j vector index
        log.debug(f"Using Neo4j vector index for search (doc_ids={doc_ids})")
        vector_hits = self._vector_index_search(query_embedding, k=k, doc_ids=doc_ids)
        if vector_hits:
            log.info(f"Neo4j vector index returned {len(vector_hits)} hits")
            return vector_hits

        # 3. Last resort: cosine search trực tiếp trên Neo4j
        log.warning("Neo4j vector index returned 0 hits, trying cosine search")
        return self._cosine_search_chunks(query_embedding, k=k, doc_ids=doc_ids)

    def _cosine_search_chunks(self, query_embedding: List[float], k: int = 8, doc_ids: Optional[List[str]] = None) -> List[Dict]:
        chunk_label = self._label("Chunk")
        doc_filter = ""
        params = {"query_emb": query_embedding, "top_k": k}
        if doc_ids:
            doc_filter = "AND c.doc_id IN $doc_ids"
            params["doc_ids"] = doc_ids
        results = self.neo4j_store.execute_query(
            f"""
            MATCH (c:{chunk_label})
            WHERE c.embedding IS NOT NULL
            {doc_filter}
            WITH c, reduce(dot = 0.0, i IN range(0, size(c.embedding)-1) |
                dot + c.embedding[i] * $query_emb[i]) AS dot_product,
                 reduce(sq = 0.0, i IN range(0, size(c.embedding)-1) |
                sq + c.embedding[i] * c.embedding[i]) AS c_norm_sq,
                 reduce(sq = 0.0, i IN range(0, size($query_emb)-1) |
                sq + $query_emb[i] * $query_emb[i]) AS q_norm_sq
            WITH c, dot_product / (sqrt(c_norm_sq) * sqrt(q_norm_sq)) AS score
            WHERE score > 0
            RETURN c.id AS chunk_id, c.text AS text, c.doc_id AS doc_id,
                   c.section_id AS section_id, c.index AS chunk_index,
                   c.page_number AS page_number, score
            ORDER BY score DESC
            LIMIT $top_k
        """,
            params,
        )
        return results

    def collect_source_pages(self, hits: List[Dict], doc_ids: List[str]) -> List[Dict]:
        if not doc_ids:
            return []

        pages_by_doc: Dict[str, set] = {doc_id: set() for doc_id in doc_ids}

        for hit in hits:
            doc_id = hit.get("doc_id")
            if not doc_id or doc_id not in pages_by_doc:
                continue
            page_number = self._coerce_page_number(hit.get("page_number"))
            if isinstance(page_number, int):
                pages_by_doc[doc_id].add(page_number)

        doc_names = self.get_document_names(doc_ids)
        id_to_name = {doc_ids[i]: doc_names[i] for i in range(len(doc_ids))}

        sources = []
        for doc_id in doc_ids:
            pages = sorted(pages_by_doc.get(doc_id, set()))
            sources.append(
                {
                    "filename": id_to_name.get(doc_id, doc_id),
                    "pages": pages,
                }
            )

        return sources

    def collect_sources_from_passages(self, doc_passages: List[dict]) -> List[Dict]:
        pages_by_file: Dict[str, set] = {}
        for passage in doc_passages:
            if not isinstance(passage, dict):
                continue
            filename = passage.get("filename") or "Không rõ"
            pages = passage.get("pages") or []
            pages_by_file.setdefault(filename, set())
            for page in pages:
                page_number = self._coerce_page_number(page)
                if isinstance(page_number, int):
                    pages_by_file[filename].add(page_number)

        sources = []
        for filename, pages in pages_by_file.items():
            sources.append({
                "filename": filename,
                "pages": sorted(pages),
            })
        return sources

    def get_graph_stats(self, source: Optional[str] = None) -> Dict:
        doc_label = self._label("Document")
        chunk_label = self._label("Chunk")
        section_label = self._label("Section")
        entity_label = self._label("Entity")

        if source:
            doc_id = self._uid(source)
            nodes = self.neo4j_store.execute_query(
                f"MATCH (n) WHERE n.doc_id = $doc_id OR n.id = $doc_id RETURN COUNT(n) AS node_count",
                {"doc_id": doc_id},
            )
        else:
            nodes = self.neo4j_store.execute_query(
                f"MATCH (n:{doc_label}) RETURN COUNT(n) AS node_count"
            )

        chunks = self.neo4j_store.execute_query(
            f"MATCH (c:{chunk_label}) RETURN COUNT(c) AS count"
        )
        sections = self.neo4j_store.execute_query(
            f"MATCH (s:{section_label}) RETURN COUNT(s) AS count"
        )
        entities = self.neo4j_store.execute_query(
            f"MATCH (e:{entity_label}) RETURN COUNT(e) AS count"
        )

        return {
            "node_count": nodes[0].get("node_count", 0) if nodes else 0,
            "chunks": chunks[0].get("count", 0) if chunks else 0,
            "sections": sections[0].get("count", 0) if sections else 0,
            "entities": entities[0].get("count", 0) if entities else 0,
        }

    def collect_context(self, hits: List[Dict]) -> tuple[List[dict], set, set]:
        seen_chunks = set()
        doc_passages = []
        section_ids = set()
        doc_ids = set()

        self._hydrate_hits(hits)

        # Collect all doc_ids first to batch-lookup filenames
        for hit in hits:
            if hit.get("doc_id"):
                doc_ids.add(hit["doc_id"])

        # Batch lookup filenames from Neo4j
        doc_id_list = list(doc_ids)
        doc_names = self.get_document_names(doc_id_list) if doc_id_list else []
        id_to_filename = {doc_id_list[i]: doc_names[i] for i in range(len(doc_id_list))}

        for hit in hits:
            chunk_id = hit.get("chunk_id") or hit.get("id")
            if not chunk_id or chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            text = hit.get("text") or hit.get("content", "")
            if text:
                page_number = self._coerce_page_number(hit.get("page_number"))
                # Resolve filename: prefer hit's own filename, fallback to doc lookup
                filename = hit.get("filename") or id_to_filename.get(hit.get("doc_id", ""), "")
                doc_passages.append({
                    "content": text,
                    "filename": filename,
                    "pages": [page_number] if page_number is not None else [],
                })
            if hit.get("section_id"):
                section_ids.add(hit["section_id"])

        return doc_passages, section_ids, doc_ids

    def _hydrate_hits(self, hits: List[Dict]) -> None:
        if not hits:
            return
        missing_chunk_ids = []
        for hit in hits:
            chunk_id = hit.get("chunk_id") or hit.get("id")
            if not chunk_id:
                continue
            if not hit.get("doc_id") or hit.get("page_number") is None or not hit.get("filename"):
                missing_chunk_ids.append(chunk_id)

        if not missing_chunk_ids:
            return

        chunk_label = self._label("Chunk")
        doc_label = self._label("Document")
        results = self.neo4j_store.execute_query(
            f"""
            UNWIND $ids AS cid
            MATCH (c:{chunk_label} {{id: cid}})
            OPTIONAL MATCH (d:{doc_label} {{id: c.doc_id}})
            RETURN c.id AS chunk_id, c.doc_id AS doc_id, c.page_number AS page_number, d.filename AS filename
        """,
            {"ids": missing_chunk_ids},
        )
        lookup = {row.get("chunk_id"): row for row in results}

        for hit in hits:
            chunk_id = hit.get("chunk_id") or hit.get("id")
            if not chunk_id:
                continue
            info = lookup.get(chunk_id)
            if not info:
                continue
            if not hit.get("doc_id") and info.get("doc_id"):
                hit["doc_id"] = info.get("doc_id")
            if hit.get("page_number") is None and info.get("page_number") is not None:
                hit["page_number"] = info.get("page_number")
            if not hit.get("filename") and info.get("filename"):
                hit["filename"] = info.get("filename")

    def _coerce_page_number(self, value):
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed.isdigit():
                return int(trimmed)
            return trimmed or None
        return None

    def extract_query_entities(self, question: str) -> List[str]:
        try:
            response = self.provider.invoke(
                self._ENTITY_EXTRACT_PROMPT.format(question=question)
            )
            match = re.search(r"\[.*?\]", response.content, re.DOTALL)
            if match:
                entities = json.loads(match.group())
                return [e for e in entities if isinstance(e, str) and e.strip()]
        except Exception:
            return []
        return []

    def get_entity_subgraph(self, entity_names: List[str], depth: int = 2) -> List[str]:
        if not entity_names:
            return []

        entity_label = self._label("Entity")
        results = self.neo4j_store.execute_query(
            f"""
            UNWIND $names AS name
            MATCH (e:{entity_label})-[r1:RELATED_TO]->(e2:{entity_label})
            WHERE toLower(e.name) CONTAINS toLower(name)
               OR toLower(e2.name) CONTAINS toLower(name)
            WITH e, r1, e2
            OPTIONAL MATCH (e2)-[r2:RELATED_TO]->(e3:{entity_label})
            RETURN
              e.name + ' --[' + r1.relation + ']--> ' + e2.name AS fact1,
              CASE WHEN r2 IS NOT NULL
                THEN e2.name + ' --[' + r2.relation + ']--> ' + e3.name
                ELSE null END AS fact2
        """,
            {"names": entity_names},
        )

        facts = set()
        for row in results:
            if row.get("fact1"):
                facts.add(row["fact1"])
            if row.get("fact2"):
                facts.add(row["fact2"])
        return list(facts)

    def get_section_summaries(self, section_ids: List[str]) -> List[str]:
        if not section_ids:
            return []
        section_label = self._label("Section")
        results = self.neo4j_store.execute_query(
            f"""
            UNWIND $ids AS sid
            MATCH (s:{section_label} {{id: sid}})
            RETURN s.title AS title, s.summary AS summary
            ORDER BY s.index
        """,
            {"ids": section_ids},
        )
        return [f"[{r['title']}] {r['summary']}" for r in results if r.get("summary")]

    def get_document_summaries(self, doc_ids: List[str]) -> List[str]:
        summaries = []
        doc_label = self._label("Document")
        for doc_id in doc_ids:
            results = self.neo4j_store.execute_query(
                f"MATCH (d:{doc_label} {{id: $id}}) RETURN d.summary AS summary",
                {"id": doc_id},
            )
            if results and results[0].get("summary"):
                summaries.append(results[0]["summary"])
        return summaries

    def get_document_names(self, doc_ids: List[str]) -> List[str]:
        if not doc_ids:
            return []
        doc_label = self._label("Document")
        results = self.neo4j_store.execute_query(
            f"""
            UNWIND $ids AS did
            MATCH (d:{doc_label} {{id: did}})
            RETURN d.id AS id, d.filename AS filename
        """,
            {"ids": doc_ids},
        )
        id_to_name = {r.get("id"): r.get("filename") for r in results}
        return [id_to_name.get(did, did) for did in doc_ids]

    def build_answer_prompt(
        self,
        doc_summary_parts: List[str],
        section_summaries: List[str],
        graph_facts: List[str],
        doc_passages: List[dict],
        question: str,
        history_block: str = "",
    ) -> str:
        """
        Build prompt cho GraphRAG.

        Args:
            doc_passages: List[dict] với mỗi dict chứa 'content', 'filename', 'pages'.
            history_block: ConversationHistory_Block từ format_history_block().
                           Default="" để backward compatible với code hiện tại.
        """
        # Escape curly braces trong history_block để tránh lỗi str.format()
        safe_history = history_block.replace("{", "{{").replace("}", "}}")

        if safe_history.strip():
            history_section = (
                f"=== Lịch sử hội thoại ===\n{safe_history}\n=== Kết thúc lịch sử ===\n\n"
            )
        else:
            history_section = ""

        # Build rich passage blocks like PaCRAG context (content + source/pages)
        passage_blocks = []
        for p in doc_passages:
            if not isinstance(p, dict):
                continue
            content = (p.get("content") or "").replace("\n", " ").strip()
            if not content:
                continue
            filename = p.get("filename") or "không xác định"
            pages = p.get("pages") or []
            pages_str = ", ".join([str(pg) for pg in pages]) if pages else "không xác định"
            block = (
                f"{content}\n"
                f"Nguồn: {filename}\n"
                f"Trang: {pages_str}"
            )
            passage_blocks.append(block)

        doc_summary_text = "\n\n".join(doc_summary_parts).strip()
        if not doc_summary_text:
            doc_summary_text = "Không có tóm tắt tài liệu cấp cao. Hãy ưu tiên dùng các đoạn văn bản liên quan ở phần bên dưới để trả lời."

        section_context_text = "\n".join(section_summaries).strip()
        if not section_context_text:
            section_context_text = "Không có tóm tắt theo section. Hãy suy luận trực tiếp từ các đoạn văn bản liên quan."

        graph_context_text = "\n".join(graph_facts).strip()
        if not graph_context_text:
            graph_context_text = "Không có quan hệ thực thể rõ ràng trong đồ thị cho truy vấn này. Hãy dựa vào nội dung đoạn văn để trả lời."

        doc_context_text = "\n\n---\n\n".join(passage_blocks).strip()
        if not doc_context_text:
            doc_context_text = "Không có đoạn văn bản liên quan để trích xuất thông tin."

        return self._ANSWER_PROMPT.format(
            doc_summary=doc_summary_text,
            section_context=section_context_text,
            graph_context=graph_context_text,
            doc_context=doc_context_text,
            history_section=history_section,
            question=question,
        )

    # =====================
    # Neo4j write helpers
    # =====================
    def _upsert_document(self, doc_id: str, filename: str, summary: str) -> None:
        doc_label = self._label("Document")
        self.neo4j_store.execute_query(
            f"""
            MERGE (d:{doc_label} {{id: $id}})
            SET d.filename = $filename, d.summary = $summary,
                d.created_at = timestamp()
        """,
            {"id": doc_id, "filename": filename, "summary": summary},
        )

    def _upsert_section(
        self,
        section_id: str,
        doc_id: str,
        index: int,
        title: str,
        summary: str,
        text: str,
    ) -> None:
        section_label = self._label("Section")
        doc_label = self._label("Document")
        self.neo4j_store.execute_query(
            f"""
            MERGE (s:{section_label} {{id: $id}})
            SET s.doc_id = $doc_id, s.index = $index,
                s.title = $title, s.summary = $summary, s.text = $text
            WITH s
            MATCH (d:{doc_label} {{id: $doc_id}})
            MERGE (d)-[:HAS_SECTION]->(s)
        """,
            {
                "id": section_id,
                "doc_id": doc_id,
                "index": index,
                "title": title,
                "summary": summary,
                "text": text,
            },
        )

    def _link_sections(self, prev_id: str, next_id: str) -> None:
        section_label = self._label("Section")
        self.neo4j_store.execute_query(
            f"""
            MATCH (a:{section_label} {{id: $prev}}), (b:{section_label} {{id: $next}})
            MERGE (a)-[:NEXT_SECTION]->(b)
        """,
            {"prev": prev_id, "next": next_id},
        )

    def _upsert_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        section_id: str,
        index: int,
        text: str,
        page_number: Optional[int],
        embedding: List[float],
    ) -> None:
        chunk_label = self._label("Chunk")
        section_label = self._label("Section")
        self.neo4j_store.execute_query(
            f"""
            MERGE (c:{chunk_label} {{id: $id}})
            SET c.doc_id = $doc_id, c.section_id = $section_id,
                c.index = $index, c.text = $text, c.content = $text,
                c.page_number = $page_number, c.embedding = $embedding
            WITH c
            MATCH (s:{section_label} {{id: $section_id}})
            MERGE (s)-[:HAS_CHUNK]->(c)
        """,
            {
                "id": chunk_id,
                "doc_id": doc_id,
                "section_id": section_id,
                "index": index,
                "text": text,
                "page_number": page_number,
                "embedding": embedding,
            },
        )

    def _link_chunks(self, prev_id: str, next_id: str) -> None:
        chunk_label = self._label("Chunk")
        self.neo4j_store.execute_query(
            f"""
            MATCH (a:{chunk_label} {{id: $prev}}), (b:{chunk_label} {{id: $next}})
            MERGE (a)-[:NEXT_CHUNK]->(b)
        """,
            {"prev": prev_id, "next": next_id},
        )

    def _upsert_entity(self, entity_id: str, name: str, etype: str = "CONCEPT") -> None:
        entity_label = self._label("Entity")
        self.neo4j_store.execute_query(
            f"""
            MERGE (e:{entity_label} {{id: $id}})
            SET e.name = $name, e.type = $etype
        """,
            {"id": entity_id, "name": name, "etype": etype},
        )

    def _link_chunk_entity(self, chunk_id: str, entity_id: str) -> None:
        chunk_label = self._label("Chunk")
        entity_label = self._label("Entity")
        self.neo4j_store.execute_query(
            f"""
            MATCH (c:{chunk_label} {{id: $cid}}), (e:{entity_label} {{id: $eid}})
            MERGE (c)-[:MENTIONS]->(e)
        """,
            {"cid": chunk_id, "eid": entity_id},
        )

    def _link_entities(self, src_id: str, tgt_id: str, relation: str) -> None:
        entity_label = self._label("Entity")
        self.neo4j_store.execute_query(
            f"""
            MATCH (a:{entity_label} {{id: $src}}), (b:{entity_label} {{id: $tgt}})
            MERGE (a)-[:RELATED_TO {{relation: $rel}}]->(b)
        """,
            {"src": src_id, "tgt": tgt_id, "rel": relation},
        )

    def delete_document(self, doc_id: str) -> None:
        doc_label = self._label("Document")
        section_label = self._label("Section")
        chunk_label = self._label("Chunk")
        self.neo4j_store.execute_query(
            f"""
            MATCH (d:{doc_label} {{id: $id}})
            OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:{section_label})
            OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c:{chunk_label})
            DETACH DELETE c, s, d
        """,
            {"id": doc_id},
        )
