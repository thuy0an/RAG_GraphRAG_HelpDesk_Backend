from collections import defaultdict
import json
from langchain_community.storage.redis import RedisStore
from redisvl.index import SearchIndex
from redisvl.query import TextQuery, VectorQuery

class HybridRetriever:
    """
    Hybrid retriever combining vector and BM25 search.
    Data access layer for retrieval operations.
    """

    def __init__(self, embeddings, redis_url):
        self.embeddings = embeddings
        self.redis_url = redis_url
        self.index = SearchIndex.from_yaml("config/redis_index.yaml")
        self.index.connect(redis_url=self.redis_url)
        self.store = RedisStore(redis_url=self.redis_url)

    async def retriever(self, query: str, k: int = 5):
        """Retrieve documents using hybrid search (vector + BM25)"""
        query_embed = await self.embeddings.aembed_query(query)

        vector_query = VectorQuery(
            vector=query_embed,
            vector_field_name="embedding",
            num_results=k,
            return_fields=["_metadata_json", "text"]
        )

        bm25_query = TextQuery(
            text=query,
            text_field_name="text",
            num_results=k,
            return_fields=["_metadata_json", "text"]
        )

        vector_docs = self.index.query(vector_query)
        bm25_docs = self.index.query(bm25_query)

        fused = self.rrf_fusion([bm25_docs, vector_docs])

        filtered_score_fused = []
        for doc_id, score in fused:
            filtered_score_fused.append((doc_id, score))

        top_docs = filtered_score_fused[:k]
        print(top_docs)

        doc_map = {}
        for doc in list(vector_docs) + list(bm25_docs):
            metadata = json.loads(doc["_metadata_json"])

            doc_map[doc["id"]] = {
                "text": doc["text"],
                "metadata": metadata
            }

        parent_to_children = defaultdict(list)
        for doc_id, _ in top_docs:
            metadata = doc_map[doc_id]["metadata"].copy()
            metadata.pop('parent_id', None)
            parent_id = doc_map[doc_id]["metadata"]["parent_id"]
            parent_to_children[parent_id].append({
                "id": doc_id, "metadata": metadata
            })

        parent_ids = list(parent_to_children.keys())
        parent_docs = self.store.mget(parent_ids)

        results = []
        for i, parent in enumerate(parent_docs):

            if not parent:
                continue

            parent_id = parent_ids[i]
            parent_text = parent.decode()

            child_ids = parent_to_children[parent_id]

            results.append({
                "id": parent_id,
                "content": parent_text,
                "children": child_ids,
            })

        return results

    async def retriever_v2(self, query: str, k: int = 5):
        """Retrieve documents using hybrid search (vector + BM25) with JSON parent docs"""
        query_embed = await self.embeddings.aembed_query(query)

        vector_query = VectorQuery(
            vector=query_embed,
            vector_field_name="embedding",
            num_results=k,
            return_fields=["_metadata_json", "text"]
        )

        bm25_query = TextQuery(
            text=query,
            text_field_name="text",
            num_results=k,
            return_fields=["_metadata_json", "text"]
        )

        vector_docs = self.index.query(vector_query)
        bm25_docs = self.index.query(bm25_query)

        fused = self.rrf_fusion([bm25_docs, vector_docs])

        filtered_score_fused = []
        for doc_id, score in fused:
            filtered_score_fused.append((doc_id, score))

        top_docs = filtered_score_fused[:k]
        print(top_docs)

        doc_map = {}
        for doc in list(vector_docs) + list(bm25_docs):
            metadata = json.loads(doc["_metadata_json"])

            doc_map[doc["id"]] = {
                "text": doc["text"],
                "metadata": metadata
            }

        parent_to_children = defaultdict(list)
        for doc_id, _ in top_docs:
            metadata = doc_map[doc_id]["metadata"].copy()
            parent_id = doc_map[doc_id]["metadata"]["parent_id"]
            parent_to_children[parent_id].append({
                "id": doc_id, "metadata": metadata
            })

        parent_ids = list(parent_to_children.keys())
        parent_docs = self.store.mget(parent_ids)

        results = []
        for i, parent in enumerate(parent_docs):
            if not parent:
                continue

            parent_id = parent_ids[i]

            try:
                parent_json = json.loads(parent.decode())
                parent_text = parent_json.get("page_content", "")
                parent_metadata = parent_json.get("metadata", {})
            except json.JSONDecodeError:
                parent_text = parent.decode()
                parent_metadata = {}

            child_ids = parent_to_children[parent_id]

            results.append({
                "id": parent_id,
                "content": parent_text,
                "metadata": parent_metadata,
                "children": child_ids,
            })

        return results

    def rrf_fusion(self, rank_lists, k: int = 60):
        """Reciprocal Rank Fusion for combining search results"""
        print("RRF fusion...")

        score_map = defaultdict(float)
        for ranking in rank_lists:
            for rank, doc in enumerate(ranking, start=1):
                doc_id = doc["id"]
                score_map[doc_id] += 1 / (k + rank)
        return sorted(score_map.items(), key=lambda x: x[1], reverse=True)