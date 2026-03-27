# import asyncio
# from collections import defaultdict
# import json
# import logging
# from re import search
# import time
# from typing import Any, Dict, List, Optional
# from langchain_community.storage.redis import RedisStore
# from langchain_core.stores import InMemoryStore
# from langchain_redis import RedisVectorStore
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from redis import Redis
# from redisvl.index import SearchIndex
# from redisvl.query import AggregateHybridQuery, FilterQuery, HybridQuery, TextQuery, VectorQuery, hybrid
# from transformers.utils import doc
# from SharedKernel.ai.AIConfig import AIConfigFactory
# from SharedKernel.ai.vector_store.VectorStoreConfig import VectoreStoreConfigFactory
# from SharedKernel.persistence.Decorators import Repository
# from SharedKernel.utils.yamlenv import load_env_yaml
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.retrievers import BM25Retriever
# from langchain_classic.retrievers import EnsembleRetriever, MultiVectorRetriever, ParentDocumentRetriever
# from langchain_core.output_parsers import StrOutputParser

# from src.Features.LangChainAPI.persistence.MemoryRepo import MemoryRepo

# logging.basicConfig(level=logging.INFO)
# log = logging.getLogger(__name__)
# config = load_env_yaml()

# class RedisVS:
#     def __init__(self, ai_factory: AIConfigFactory, memory_repo: MemoryRepo = None):
#         self.ai_config = ai_factory.create(config.ai.llm_provider)
#         self.embeddings = self.ai_config.create_embedding()
#         self.vs_config = VectoreStoreConfigFactory.create(config.vector_store.provider)
#         self.redis_vs = self.vs_config.get_vecstore(self.embeddings)
#         self.redis_url = self.vs_config.get_url()
#         self.store = RedisStore(redis_url=self.redis_url)
#         self.memory_repo = memory_repo  # Inject MemoryRepo for persistence
#         ...
    
#     ###################### 
#     # [REPOSITORY - MOVE TO RedisVSRepository.py] DOCUMENT OPERATIONS
#     ######################
#     async def abatch_add_documents(self, documents: List[Any]):
#         if not documents:
#             print("No documents to add")
#             return []

#         total_docs = len(documents)
#         batch_size = min(100, total_docs) 

#         all_ids = []
#         for i in range(0, total_docs, batch_size):
#             batch = documents[i:i + batch_size]
#             batch_ids = await self.redis_vs.aadd_documents(batch)
#             all_ids.extend(batch_ids)
            
#             print(f"Processed batch {i//batch_size + 1}/{(total_docs - 1)//batch_size + 1}")

#         print(f"Added {len(documents)} documents with metadata")
#         return all_ids
#         ...

#     # [REPOSITORY - MOVE TO RedisVSRepository.py]
#     async def add_documents_with_metadata(self, documents: List[Document]):      
#         if documents:
#             print(f"Adding {len(documents)} documents with metadata...")
            
#             await self.redis_vs.aadd_documents(documents)
            
#             print(f"Successfully added {len(documents)} documents")
            
#             sources = set(doc.metadata.get('source', 'unknown') for doc in documents)
#             print(f"Sources: {list(sources)}")
#         else:
#             print("No documents to add")

#     # [REPOSITORY - MOVE TO RedisVSRepository.py]
#     async def abatch_add_documents_with_metadata(self, chunks: List[Document]):
#         documents = []
#         current_time = int(time.time())

#         for i, chunk in enumerate(chunks):
#             metadata = chunk.metadata if chunk.metadata else {}

#             metadata.update({
#                 "chunk_index": i,
#                 "total_chunks": len(chunks),
#                 "timestamp": current_time,
#                 "content_length": len(chunk.page_content)
#             })
        
#             documents.append(Document(
#                 page_content=chunk.page_content,
#                 metadata=metadata
#             ))

#         print(f"Saving {len(documents)} documents to vector store")
    
#         if documents:
#             print(f"Sample metadata: {documents[0].metadata}")
        
#         await self.abatch_add_documents(documents)

#     # [REPOSITORY - MOVE TO RedisVSRepository.py]
#     async def add_PaC_documents(self, chunks: Any):
#         parent_ids = []
#         for parent in chunks.get("parent", []): 
#             key = parent.metadata.get('parent_id', "")
#             self.store.mset([(key, parent.page_content)])
#             parent_ids.append(key)
#             print(f"Saved parent: {key}")

#         print(chunks.get('children'))
#         await self.abatch_add_documents_with_metadata(chunks.get('children', []))
#         print(f"Saved child")
#         ...

#     # [REPOSITORY - MOVE TO RedisVSRepository.py]
#     async def delete_documents_by_metadata(self, metadata: dict):
#         try:
#             r = Redis.from_url(self.redis_url)
#             index = SearchIndex.from_yaml("config/redis_index.yaml")
#             index.connect(redis_url=self.redis_url)

#             target_source = metadata["source"]
#             filter_expression = f'@source:{{"{target_source}"}}'

#             BATCH_SIZE = 500
#             PIPELINE_BATCH = 500

#             offset = 0
#             deleted_count = 0

#             while True:
#                 filter_query = FilterQuery(
#                     return_fields=["id"],
#                     filter_expression=filter_expression,
#                     num_results=BATCH_SIZE
#                 )

#                 results = index.query(filter_query)
#                 if not results:
#                     break

#                 pipeline = r.pipeline()
#                 batch_count = 0

#                 for result in results:
#                     doc_id = result.get("id")
#                     if doc_id:
#                         pipeline.unlink(doc_id)
#                         batch_count += 1
#                         deleted_count += 1

#                     if batch_count == PIPELINE_BATCH:
#                         pipeline.execute()
#                         pipeline = r.pipeline()
#                         batch_count = 0

#                 pipeline.execute()
#                 offset += BATCH_SIZE
#                 print(f"Deleted batch, total: {deleted_count}")

#             print(f"Deleted {deleted_count} vector documents for source: {target_source}")

#             cursor = 0
#             deleted_parents = 0
#             while True:
#                 cursor, keys = r.scan(cursor, match=f"parent_docs:{target_source}:*", count=100)
#                 if not keys:
#                     if cursor == 0:
#                         break
#                     continue

#                 pipeline = r.pipeline()
#                 batch_count = 0

#                 for key in keys:
#                     key_str = key.decode('utf-8')
#                     if target_source in key_str:
#                         pipeline.unlink(key)
#                         batch_count += 1
#                         deleted_parents += 1

#                     if batch_count == PIPELINE_BATCH:
#                         pipeline.execute()
#                         pipeline = r.pipeline()
#                         batch_count = 0

#                 pipeline.execute()

#                 if cursor == 0:
#                     break
            
#             print(f"Deleted {deleted_count} vector docs and {deleted_parents} parent docs for source: {target_source}")
#         except Exception as e:
#             log.error(f"Error deleting documents: {e}")
#             return
#         ...

#     # [SERVICE - MOVE TO RedisVSService.py] RAG with history
#     async def rag_PaC_history(self,         
#         query: str,
#         provider,
#         session_id: str = None
#     ):
#         hybrid_retriver = HybridRetriever(self.embeddings, self.redis_url)
#         hybrid_docs = await hybrid_retriver.retriever(query, 10)
#         context = self._format_context_PaC(hybrid_docs)

#         system_prompt = """
#         Bạn là trợ lý AI chuyên nghiệp.

#         Hãy trả lời câu hỏi của người dùng dựa trên context

#         YÊU CẦU BẮT BUỘC:
#         1. Trả lời câu hỏi dựa trên ngữ cảnh
#         2. KẾT THÚC câu trả lời với 3 dòng thông tin nguồn:

#         Trong ngữ cảnh có metadata ở cuối mỗi tài liệu với định dạng:
#         Source: <tên file>, Page: <trang>

#         Hãy trích xuất thông tin từ metadata này và trình bày lại theo định dạng sau:

#         - Nguồn: <tên file>
#         - Trang: <không xác định nếu không có thông tin>

#         QUAN TRỌNG:
#         - Chỉ sử dụng thông tin từ metadata.
#         - Nếu không có thông tin trang thì ghi: "không xác định".
#         - Không sử dụng định dạng khác.

#         Ví dụ output:

#         - Nguồn: example.pdf
#         - Trang: không xác định
#         """
#         template = f"""{system_prompt}

#         context: {context}

#         Câu hỏi: {query}

#         Hãy trả lời câu hỏi dựa trên context và metadata

#         Lưu ý nếu không tìm thấy thông tin thì output: tôi không có thông tin vui lòng liên hệ bộ phận hỗ trợ
#         """
#         print(context)
#         prompt = ChatPromptTemplate.from_template(template)
#         chain = prompt | provider

#         async def response():
#             async for chunk in chain.astream({"query": query}):
#                 if hasattr(chunk, "content"):
#                     yield chunk.content

#             if session_id:
#                 print(f"Session ID: {session_id}")

#         return response()
#     ...

#     # [SERVICE - MOVE TO RedisVSService.py] Query RAG and save to memory
#     async def query_and_save(
#         self,
#         query: str,
#         provider,
#         session_id: str,
#         top_k: int = 5
#     ) -> str:
#         """
#         Query RAG and save answer to memory via MemoryRepo.
        
#         Args:
#             query: User query string
#             provider: LLM provider for generating answer
#             session_id: Session ID to save answer to
#             top_k: Number of chunks to retrieve
            
#         Returns:
#             Generated answer string
#         """
#         if not self.memory_repo:
#             raise ValueError("MemoryRepo not injected")
        
#         if not session_id:
#             raise ValueError("session_id required for saving answer")
        
#         await self.memory_repo.init_db()
        
#         hybrid_retriver = HybridRetriever(self.embeddings, self.redis_url)
#         hybrid_docs = await hybrid_retriver.retriever(query, top_k)
        
#         # Format context
#         context = self._format_context_PaC(hybrid_docs)
        
#         # Generate answer
#         system_prompt = """
#         Bạn là trợ lý AI chuyên nghiệp.

#         Hãy trả lời câu hỏi của người dùng dựa trên context

#         YÊU CẦU BẮT BUỘC:
#         1. Trả lời câu hỏi dựa trên ngữ cảnh
#         2. KẾT THÚC câu trả lời với thông tin nguồn
#         """
#         template = f"""{system_prompt}

#         context: {context}

#         Câu hỏi: {query}

#         Hãy trả lời câu hỏi dựa trên context và metadata
#         """
        
#         prompt = ChatPromptTemplate.from_template(template)
#         chain = prompt | provider
        
#         answer_parts = []
#         async for chunk in chain.astream({"query": query}):
#             if hasattr(chunk, "content"):
#                 answer_parts.append(chunk.content)
        
#         answer = "".join(answer_parts)
        
#         sources = self._extract_sources(hybrid_docs)
#         await self.memory_repo.add_message(
#             session_id=session_id,
#             role="assistant",
#             content=f"{answer}\n\n[Retrieved from: {sources}]"
#         )
        
#         return answer

#     # [SERVICE - MOVE TO RedisVSService.py] Helper: Extract sources
#     def _extract_sources(self, search_results: List[Dict[str, Any]]) -> str:
#         """Extract source files from search results"""
#         if not search_results:
#             return "No sources"
        
#         sources = set()
#         for result in search_results:
#             file_name = result["id"].split(":")[1] if ":" in result["id"] else "unknown"
#             sources.add(file_name)
        
#         return ", ".join(sources) if sources else "unknown"

#     # [SERVICE - MOVE TO RedisVSService.py] Helper: Format context
#     def _format_context_PaC(self,         
#         search_results: List[Dict[str, Any]]):

#         if not search_results:
#             return "No relevant documents found."

#         context_parts = []
#         for idx, result in enumerate(search_results):
#             doc_content = result["content"].replace("\n", " ").strip()
#             file_name = result["id"].split(":")[1]
#             page = result["id"].split(":")[2].split("_")[1]
            
#             metadata_info = []

#             if file_name:
#                 metadata_info.append(f"Source: {file_name}")
#             if page:
#                 metadata_info.append(f"Page: {page}")
            
#             doc_content = doc_content + "\n" + (" | ".join(metadata_info))
#             context_parts.append(doc_content)

#         formatted_context = "\n\n".join(context_parts)
#         return formatted_context
#     ...
    
#     # [REPOSITORY - MOVE TO RedisVSRepository.py] HybridRetriever class
#     class HybridRetriever:
#         def __init__(self, embeddimgs, redis_url):
#             self.embeddings = embeddimgs
#             self.redis_url = redis_url
#             self.index = SearchIndex.from_yaml("config/redis_index.yaml")
#             self.index.connect(redis_url=self.redis_url)
#             self.store = RedisStore(redis_url=self.redis_url)
#             ...
#         ...
    
#     async def retriever(self, query: str, k: int = 5):
#         query_embed = await self.embeddings.aembed_query(query)

#         vector_query = VectorQuery(
#             vector=query_embed,
#             vector_field_name="embedding",
#             num_results=k,
#             return_fields=["_metadata_json", "text"]
#         )

#         bm25_query = TextQuery(
#             text=query,
#             text_field_name="text",
#             num_results=k,
#             return_fields=["_metadata_json", "text"]
#         )

#         vector_docs = self.index.query(vector_query)
#         bm25_docs = self.index.query(bm25_query)

#         fused = self.rrf_fusion([bm25_docs, vector_docs])

#         filtered_score_fused = []
#         for doc_id, score in fused:
#             filtered_score_fused.append((doc_id, score))

#         top_docs = filtered_score_fused[:k]
#         print(top_docs)

#         doc_map = {}
#         for doc in list(vector_docs) + list(bm25_docs):
#             metadata = json.loads(doc["_metadata_json"])

#             doc_map[doc["id"]] = {
#                 "text": doc["text"],
#                 "metadata": metadata
#             }

#         parent_to_children = defaultdict(list)
#         for doc_id, _ in top_docs:
#             metadata = doc_map[doc_id]["metadata"].copy()
#             metadata.pop('parent_id', None) 
#             parent_id = doc_map[doc_id]["metadata"]["parent_id"]
#             parent_to_children[parent_id].append({
#                 "id": doc_id, "metadata": metadata
#             })

#         parent_ids = list(parent_to_children.keys())
#         parent_docs = self.store.mget(parent_ids)

#         results = []
#         for i, parent in enumerate(parent_docs):

#             if not parent:
#                 continue

#             parent_id = parent_ids[i]
#             parent_text = parent.decode()

#             child_ids = parent_to_children[parent_id]

#             results.append({
#                 "id": parent_id,
#                 "content": parent_text,
#                 "children": child_ids,
#             })
        
#         return results

#     def rrf_fusion(self, rank_lists, k: int = 60):
#         """
#         Reciprocal Rank Fusion
#         """
#         print("RRF fusion...")

#         score_map = defaultdict(float)
#         for ranking in rank_lists:
#             for rank, doc in enumerate(ranking, start=1):
#                 doc_id = doc["id"]
#                 score_map[doc_id] += 1 / (k + rank)
#         return sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    