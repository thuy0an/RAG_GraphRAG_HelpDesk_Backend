from functools import total_ordering
from typing import Any, List, Optional
from SharedKernel.AIConfig import AIConfig, AIConfigFactory
from SharedKernel.VectorStoreFactory import VectoreStoreConfigFactory
from SharedKernel.utils.yamlenv import load_env_yaml
from langchain_core import embeddings
from langchain_core.vectorstores import VectorStore
import numpy as np

config = load_env_yaml()

class RedisVectorRepo:
    def __init__(self, ai_factory: AIConfig):
        self.vector_store_config = VectoreStoreConfigFactory.create(config.vector_store.provider)
        self.ai_config = ai_factory.create(config.ai.llm_provider)
        self.embeddings = self.ai_config.create_embedding()
        self.vector_store: Optional[VectorStore] = None
        self._init_store()
        ...

    def _init_store(self):
        self.vector_store = self.vector_store_config.create_vector_store(self.embeddings)
        ...

    async def add_documents(self, texts: List[Any]):
        if texts:
            total_chunks = len(texts)
            if total_chunks >= 100:
                batch_size = total_chunks // 5
            else:
                batch_size = total_chunks
            
            print(f"Total chunks: {total_chunks}")
            print(f"Batch size: {batch_size}")

            for i in range(0, total_chunks, batch_size):
                batch = texts[i:i + batch_size]
                await self.vector_store.aadd_texts(batch)
                print(f"Processed batch {i//batch_size}/{total_chunks//batch_size}")

            print(f"Successfully added {len(texts)} documents")
        else:
            print("No documents to add")
        ...

    async def search(self, query = "What is LangGraph?"):
        if not query:
            print("No query provided")
            return []

        embed_q = self.embeddings.embed_query(query)

        results = await self.vector_store.asimilarity_search(
            query,
            k=5,
            embed_query=embed_q
        )

        print(f"Found {len(results)} similar documents:")
        for i, doc in enumerate(results):
            print(f" {i+1}. {doc.page_content[:100]}...")
    
        return results
        ...

    # def abstract_retriver(self, query: str = "What is LangChain?"):
    #     embed = self.vector_store.from_texts(self.docs, embedding=self.embeddings)
    #     retriever = embed.as_retriever(search_kwargs={"k": 3})

    #     retrieved_documents = retriever.invoke(query)

    #     print(retrieved_documents[0].page_content)
    #     ...

    # def low_retriever(self, query: str = "What is LangGraph?"):
    #     embed_q = self.embeddings.embed_query(query)
    #     # print(str(embed_q)[:100])

    #     doc_embeddings = self.embeddings.embed_documents(self.docs)

    #     query_vec = np.array(embed_q)
    #     doc_vecs = np.array(doc_embeddings)
        
    #     similarities = np.dot(doc_vecs, query_vec) / (
    #         np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec)
    #     )
        
    #     best_idx = np.argmax(similarities)
    #     best_doc = self.docs[best_idx]
    #     best_score = similarities[best_idx]
        
    #     print(f"Most similar: {best_doc} (score: {best_score:.4f}")

    #     for i, (doc, score) in enumerate(zip(self.docs, similarities)):
    #         print(f" {i+1}. {score:.4f}: {doc}")
    #     ...
    ...