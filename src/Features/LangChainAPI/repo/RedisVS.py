import asyncio
import json
import logging
from re import search
import time
from typing import Any, Dict, List, Optional
from langchain_community.storage.redis import RedisStore
from redis import Redis
from SharedKernel.ai.AIConfig import AIConfigFactory
from SharedKernel.ai.vector_store.VectorStoreConfig import VectoreStoreConfigFactory
from SharedKernel.persistence.Decorators import Repository
from SharedKernel.utils.yamlenv import load_env_yaml
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
config = load_env_yaml()

class RedisVS:
    def __init__(self, ai_factory: AIConfigFactory):
        self.ai_config = ai_factory.create(config.ai.llm_provider)
        self.embeddings = self.ai_config.create_embedding()
        self.vs_config = VectoreStoreConfigFactory.create(config.vector_store.provider)
        self.vs_repo = self.vs_config.get_vecstore(self.embeddings)

        self.redis_url = self.vs_config.get_url()
        print(f"Redis URL: {self.redis_url}")
        self.docstore_redis = RedisStore(redis_url=self.redis_url)
        ...
    # 
    # DOCUMENT 
    # 
    async def abatch_add_documents(self, documents: List[Any]):
        if not documents:
            print("No documents to add")
            return []

        total_docs = len(documents)
        batch_size = min(100, total_docs) 

        all_ids = []
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_ids = await self.vs_repo.aadd_documents(batch)
            all_ids.extend(batch_ids)
            
            print(f"Processed batch {i//batch_size + 1}/{(total_docs - 1)//batch_size + 1}")

        print(f"Added {len(documents)} documents with metadata")
        return all_ids
        ...


    async def add_documents_with_metadata(self, documents: List[Document]):      
        if documents:
            print(f"Adding {len(documents)} documents with metadata...")
            
            await self.vs_repo.aadd_documents(documents)
            
            print(f"Successfully added {len(documents)} documents")
            
            sources = set(doc.metadata.get('source', 'unknown') for doc in documents)
            print(f"Sources: {list(sources)}")
        else:
            print("No documents to add")

    async def delete_documents_by_metadata(self, metadata: dict):
        search_results = await self.vs_repo.asimilarity_search(
            query="test",  
            k=1000,
            filter=metadata
        )
        log.info(f"deleted result {search_results}")
        ...

    #
    # RAG
    # 
    async def search(self, 
        query = "What is LangGraph?", 
        k = 10, 
        score_threshold: float = 0.7,
        metadata_filter = None
    ):
        if not query:
            return []

        try:
            if metadata_filter:
                results = await self.vs_repo.asimilarity_search_with_score(
                    query, 
                    k=k, 
                    filter=metadata_filter
                )
            else:
                results = await self.vs_repo.asimilarity_search_with_score(
                    query, 
                    k=k
                )
            
            formatted_results = []
            for doc, score in results:
                print(f"score thres: {score_threshold}, score: {score}")
                if score < score_threshold:
                    log.info("Check score\n")
                    if doc.metadata is None:
                        doc.metadata = {}
                    
                    doc.metadata['vector_distance'] = float(score)
                    
                    doc_id = doc.metadata.get('id', None)
                    
                    formatted_result = {
                        "id": doc_id,
                        "metadata": doc.metadata,
                        "page_content": doc.page_content,
                        "score": float(score)
                        # "type": "Document"
                    }
                    
                    formatted_results.append(formatted_result)
                
                log.info(f"Found {len(formatted_results)} results for query: '{query[:50]}...'\n")
            return formatted_results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
        ...

    async def manual_rag(
        self, 
        query: str,
        provider, 
        k: int = 3,
        score_threshold: float = 0.7,
        metadata_filter: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        include_metadata: bool = True
    ):
        system_prompt = """
        Bạn là trợ lý AI chuyên nghiệp.

        Hãy trả lời câu hỏi của người dùng dựa trên context
        """

        more = """
        YÊU CẦU BẮT BUỘC:
        1. Trả lời câu hỏi dựa trên ngữ cảnh
        2. KẾT THÚC câu trả lời với 3 dòng thông tin nguồn:

        Trong ngữ cảnh có metadata ở cuối mỗi tài liệu với định dạng:
        (Source: <tên file>, Chunk: <chunk id>, Type: <loại file>, Relevance Score: <điểm liên quan>)

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
        system_prompt += more

        search_results = await self.search(
            query=query,
            k=k,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter
        )

        context = self._format_context(
            search_results=search_results,
            include_metadata=include_metadata
        )
        log.info(f"Context: {context}")
        
        template = f"""{system_prompt}

        context: {context}

        câu hỏi: {query}

        hãy trả lời câu hỏi dựa trên context và metadata
        """
        log.info(f"Template: {template}")

        if search_results is None:
            return ""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | provider 
        answer = await chain.ainvoke({"query": query})

        sources = []
        for result in search_results:
            metadata = result["metadata"]
            sources.append({
                "source": metadata.get('source', 'unknown'),
                "chunk_index": metadata.get('source', 'unknown'),
                "page_number": metadata.get('page_number', 'unknown'),
                "type": metadata.get('type', 'unknown'),
                "relevance": result["score"]
            })
        
        return {
            "answer": answer.content,
            "query": query,
            "context": context,
            "sources": sources,
            "document_count": len(search_results),
            "score_threshold": score_threshold,
            "filter_applied": metadata_filter
        }
        ...

    def get_all_docs(self):        
        from redis.client import Redis as RedisClient
        cursor = 0
        docs = []
        
        r = RedisClient.from_url(self.redis_url)

        while True:
            cursor, keys = r.scan(cursor, match=f"{self.vs_config.index_name}:*", count=100)
            
            for k in keys:
                doc_data = r.hgetall(k)
                if doc_data:
                    content = doc_data.get(b'text', b'').decode('utf-8')
                    
                    metadata = {}
                    for key, value in doc_data.items():
                        if key == b'text':
                            continue
                        try:
                            key_str = key.decode('utf-8')
                            value_str = value.decode('utf-8')
                            metadata[key_str] = value_str
                        except UnicodeDecodeError:
                            continue

                    if '_metadata_json' in metadata:
                        try:
                            json_meta = json.loads(metadata['_metadata_json'])
                            metadata.update(json_meta)
                            del metadata['_metadata_json']
                        except json.JSONDecodeError:
                            pass
                    
                    metadata['id'] = k.decode('utf-8')
                    
                    docs.append(Document(page_content=content, metadata=metadata))
            if cursor == 0:
                break
        
        r.close()
        return docs
        ...

    async def abstract_rag(self, query: str, k: int = 1):
        from langchain_classic.retrievers import EnsembleRetriever
        all_docs = self.get_all_docs()
        print(f"{all_docs}\n\n\n")
        vector_retriever = self.vs_repo.as_retriever(search_kwargs={"k": k})
        result = await vector_retriever.ainvoke(query)
        print(result)

        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = k

        vector_retriever = self.vs_repo.as_retriever(search_kwargs={"k": k})

        hybrid_retriever = HybridRetriever(bm25_retriever, vector_retriever)

        result = await hybrid_retriever.asearch(query)
        print(result)
    ...

    def _format_context(
        self,
        search_results: List[Dict[str, Any]],
        include_metadata: bool = True,
        max_context_length: int = 4000
    ):
        if not search_results:
            return "No relevant documents found."

        context_parts = []
        current_length = 0

        for i, result in enumerate(search_results):
            doc_content = result["page_content"]
            score = result["score"]
            metadata = result["metadata"]
            
            doc_text = f"Document {i+1}:\n{doc_content}"
            
            if include_metadata and metadata:
                metadata_info = []
                
                if metadata.get('source'):
                    metadata_info.append(f"Source: {metadata['source']}")
                if metadata.get('chunk_index') is not None:
                    metadata_info.append(f"Chunk: {metadata['chunk_index']}")
                if metadata.get('page_number') is not None:
                    metadata_info.append(f"Page: {metadata['page_number']}")
                if metadata.get('type'):
                    metadata_info.append(f"Type: {metadata['type']}")
                metadata_info.append(f"Relevance Score: {score:.3f}")
                
                if metadata_info:
                    doc_text += f"\n({', '.join(metadata_info)})"
            
            if current_length + len(doc_text) > max_context_length:
                print(f"⚠️ Context truncated at {i} documents")
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        formatted_context = "\n\n".join(context_parts)
        
        log.info(f"📝 Formatted context: {len(context_parts)} documents, {len(formatted_context)} chars")
        
        return formatted_context
    ...
    
class HybridRetriever:
    def __init__(self, 
        bm25_retriever: BM25Retriever, 
        vector_retriever: Any,
        k: int = 5,
        rrf_k: float = 60.0
    ):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.k = k
        self.rrf_k = rrf_k 

    async def asearch(self, query):

        bm25_task = asyncio.create_task(self.bm25.ainvoke(query))
        vector_task = asyncio.create_task(self.vector.ainvoke(query))

        bm25_docs, vector_docs = await asyncio.gather(bm25_task, vector_task)
        
        # print(bm25_docs,"\n\n\n")
        # print(vector_docs,"\n\n\n")

        result = self.reciprocal_rank_fusion([bm25_docs, vector_docs])

        return result

    def reciprocal_rank_fusion(self, results_list):
        """
        Reciprocal Rank Fusion1
        """
        print("RRF...")

        score_map = {}
        for results in results_list:
            for rank, doc in enumerate(results):
                content = doc.page_content
                if content not in score_map:
                    score_map[content] = [0.0, doc]
                
                score = 1.0 / (self.rrf_k + rank)
                score_map[content][0] += score 
        ranked = sorted(score_map.items(), key=lambda x: x[1][0], reverse=True)
        
        docs_with_scores = []
        for _, (score, doc) in ranked:
            doc.metadata['ranked_score'] = score
            docs_with_scores.append(doc)
        
        return docs_with_scores
    