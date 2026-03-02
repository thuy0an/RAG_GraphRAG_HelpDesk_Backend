# import json
# import os
# from typing import Dict, List
# from fastapi import Depends
# from langchain_core.documents import Document
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.runnables.passthrough import RunnablePassthrough
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from pydantic import BaseModel, Field
# from src.Features.AI_API.config.LLMConfig import gemini_config
# from src.Features.AI_API.config.VectorDbConfig import VectorDBConfig
# # from src.Features.TestAPI.dto import SQLGenerationResult
# from src.Shared.base import get_logger
# from langchain_core.output_parsers import StrOutputParser
# from src.Shared.base.Logger import Logger
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from src.Shared.persistence.BaseRepository import BaseRepository 
# import tempfile

# logger = get_logger(__name__)

# class AIService:
#     def __init__(self, base_repo: BaseRepository = Depends()):
#         self.model = gemini_config.model()
#         self.embeddings = gemini_config.embeddings()
#         self.base_repo = base_repo

#         self.system_prompt = "You are a helpful assistant."
#         self.conversation_history: List[Dict[str, str]] = []

#         self._store: dict[str, InMemoryChatMessageHistory] = {}

#         # TÌM KỸ THUẬT TỐI ƯU
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1500,
#             chunk_overlap=200,
#             add_start_index=True,
#             separators=["\n\n", "\n", " ", ""] 
#         )

#         self.vector_config = VectorDBConfig()
#         self.vector_store = self.vector_config.create_vector_store(
#             embeddings=gemini_config.embeddings()
#         )
#         pass

#     # 
#     # DOCUMENT PROCESSING
#     # 
#     async def ingest_doc(self, file_name: str, file_content):   
#         suffix = os.path.splitext(file_name)[1]
#         logger.info(f"File name: {file_name}, Suffix: {suffix}")

#         if hasattr(file_content, 'read'):
#             content_bytes = file_content.read()
#         else:
#             content_bytes = file_content

#         with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=suffix) as temp_file:
#             temp_file.write(content_bytes)
#             temp_path = temp_file.name

#         loader = PyPDFLoader(temp_path)

#         try:
#             docs = loader.load()
            
#             all_splits = self.text_splitter.split_documents(docs)

#             await self.save_chunks_to_vector_store(all_splits)

#             return {
#                     "status": "success",
#                     "chunks_created": len(all_splits),
#                     "metadata": {"filename": file_name}
#                 }
#         finally:
#             if os.path.exists(temp_path):
#                 os.remove(temp_path)
#         pass

#     async def save_chunks_to_vector_store(self, splits):
#         try:    
#             await self.vector_store.aadd_documents(splits)
#             logger.info(f"Successfully indexed {len(splits)} chunks to Redis.")
#         except Exception as e:
#             logger.error(f"Error saving to Redis: {e}")
#             raise
#         pass

#     async def debug_query(self, query: str):
#         query_vector = self.embeddings.embed_query(query)
#         logger.info(f"Vector sinh ra có {len(query_vector)} chiều.")
#         logger.info(f"5 số đầu tiên: {query_vector[:5]}")

#         results = await self.vector_store.asimilarity_search(query, k=5)
#         if results:
#             logger.info(f"Tìm thấy nội dung: {results[0].page_content}")

#     #
#     # RAG
#     #

#     async def rag_query(self, query: str):
#         retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
#         await self.debug_query(query)

#         system_prompt = """
#         Bạn là một trợ lý chuyên giải đáp thắc mắc. 
#         Hãy sử dụng các phần nội dung đã được cung cấp sau đây để trả lời câu hỏi.
#         Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết. 
#         Hãy giữ câu trả lời ngắn gọn và súc tích.

#         Dưới đây là các phần nội dung:
#         {context}

#         Câu hỏi: 
#         {question}
#         """

#         def format_docs(docs):
#             return "\n\n".join(doc.page_content for doc in docs)

#         prompt = ChatPromptTemplate.from_messages([
#             ("system", system_prompt),
#             ("human", "Context: {context}\n\nQuestion: {question}")
#         ])

#         rag_chain = (
#             {
#                 "context": retriever | format_docs, 
#                 "question": RunnablePassthrough()
#             }
#             | prompt
#             | self.model
#             | StrOutputParser()
#         )
        
#         response = await rag_chain.ainvoke(query)
        
#         docs = await retriever.ainvoke(query)
#         logger.info(f"Found {len(docs)} documents")
        
#         for doc in docs:
#             logger.info(f"Doc: {doc.page_content[:100]}...")
#         sources = [doc.metadata for doc in docs]

#         return {
#             "answer": response,
#             "sources": sources
#         }
#     #
#     # 
#     #
    
#     async def rag_sql(self, query: str):
#         if(query == "ingest"):
#             metadata_samples = [
#                 {
#                     "table": "Ticket",
#                     "description": "Lưu trữ thông tin ticket hỗ trợ khách hàng bao gồm tiêu đề, mô tả, trạng thái, ưu tiên và thông tin người dùng liên quan.",
#                     "columns": "id, subject, description, status, priority, category, customer_id, assigned_agent_id, dept_id, satisfaction_rating, customer_feedback, created_at, due_date, attachment_url"
#                 },
#             ]

#             docs = []
#             for item in metadata_samples:
#                 content_json = json.dumps(item, ensure_ascii=False)
                
#                 doc = Document(
#                     page_content=content_json
#                 )
#                 docs.append(doc)
        
#             await self.vector_store.aadd_documents(docs)
#             logger.info("Đã nạp Metadata Schema vào Redis.")

        
#         relevant_tables = await self.vector_store.asimilarity_search(query, k=3)
#         logger.info(relevant_tables)

#         schema_context = ""
#         for table_doc in relevant_tables:
#             try:
#                 table_data = json.loads(table_doc.page_content)
                
#                 t_name = table_data.get('table')
#                 t_desc = table_data.get('description')
#                 t_cols = table_data.get('columns')
                
#                 schema_context += f"- Table: {t_name}\n  Description: {t_desc}\n  Columns: [{t_cols}]\n\n"
#             except json.JSONDecodeError:
#                 schema_context += f"- {table_doc.page_content}\n"
#         logger.info(f"Schema context: {schema_context}")

#         prompt = f"""
#         Dựa trên schema các bảng liên quan sau đây:
#         {schema_context}
        
#         Hãy viết câu lệnh SQL cho yêu cầu: 
#         {query}

#         Trả về kết quả với:
#         - sql: câu lệnh SQL
#         - dialect: mysql/postgres/sqlite
#         - explanation: giải thích ngắn gọn
#         - confidence: mức độ tin cậy (0-1)
#         """
        
#         structed_chain = (
#             ChatPromptTemplate.from_template(prompt)
#             # | self.model.with_structured_output(SQLGenerationResult)
#         )

#         result = await structed_chain.ainvoke({})
#         return result
#         pass
    
#     # 
#     #
#     # 
#     def _format_messages(self, messages: List[Dict[str, str]]) -> List:
#         formatted_messages = []
#         for msg in messages:
#             if msg["role"] == "system":
#                 formatted_messages.append(SystemMessage(content=msg["content"]))
#             elif msg["role"] == "user":
#                 formatted_messages.append(HumanMessage(content=msg["content"]))
#             elif msg["role"] == "assistant":
#                 formatted_messages.append(AIMessage(content=msg["content"]))
#         return formatted_messages
#         pass

#     pass