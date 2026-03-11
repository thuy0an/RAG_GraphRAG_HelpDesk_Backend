from collections import defaultdict
import json
import logging
import os
import shutil
import tempfile
import time
from typing import Any, List
import uuid
from langchain_classic.retrievers import ParentDocumentRetriever
from Features.LangChainAPI.LangChainDTO import Callback, ChunkResponse, RagRequest, RagType, SplitRequest
from Features.LangChainAPI.repo.RedisVS import RedisVS
from SharedKernel.ai.AIConfig import AIConfigFactory
from SharedKernel.persistence.Decorators import Service
from SharedKernel.utils.yamlenv import load_env_yaml
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_unstructured import UnstructuredLoader
from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
config = load_env_yaml()

# @Service
class Synthesis:
    def __init__(self, ai_factory: AIConfigFactory, provider, callbacks) -> None:
        self.provider = provider
        self.callbacks = callbacks or {}
        self.ai_factory = ai_factory
        self.loader = Loader(self.provider, self.callbacks)
        self.process = Process(self.provider, self.callbacks)
        self.vs_repo = RedisVS(self.ai_factory)
    ...

    async def call_rag(self, req: RagRequest):
        # child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
        # parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        
        # retriever = ParentDocumentRetriever(
        #     vectorstore=self.vs_repo.vectorstore,
        #     docstore=self.vs_repo.docstore_redis,
        #     child_splitter=child_splitter,
        #     parent_splitter=parent_splitter,
        # )
        
        # relevant_docs = retriever.invoke(req.query)
        # print(relevant_docs)

        # await self.vs_repo.abstract_rag(
        #     query=req.query,
        #     k=5
        # )
        # return result
        ...

    #
    # INGEST
    #
    async def ingest_pdf(self, file: UploadFile):
        docs = self.loader.load_pdf(file)
        if not docs["documents"]:
            print("No documents loaded")
            return
        all_chunks = self.process.split_by_page(docs)
        await self.batch_ingest_with_metadata(all_chunks)
    ...

    """

    """
    async def ingest_pdf_Pc(self, file: UploadFile):
        docs = self.loader.load_pdf_Pc(file)
        log.info(docs)
        # if not docs:
        #     print("No documents loaded")
        #     return
        # chunks = self.process.split_by_page_PaC(docs)
        # print(chunks)
        # parent_ids = []
        # for parent in chunks["parent"]:
        #     key = f"parent:{docs.get('filename', 'unknown')}:page_{parent.metadata.get('page_number', 0)}:{parent.metadata.get('parent_index', 0)}"
        #     self.vs_repo.docstore_redis.mset([(key, parent.page_content)])
        #     parent_ids.append(key)
        #     print(f"Saved parent: {key}")
    ...

    """
    TODO: chua implement, xoa doc hien co de cap nhat docs moi
    """
    async def update_docs(self):
        ...


    async def batch_ingest_with_metadata(self, chunks: List[Any]):
        documents = []
        current_time = int(time.time())

        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata if chunk.metadata else {}

            metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "timestamp": current_time,
                "content_length": len(chunk.content)
            })
        
            documents.append(Document(
                page_content=chunk.content,
                metadata=metadata
            ))

        print(f"Saving {len(documents)} documents to vector store")
    
        if documents:
            print(f"Sample metadata: {documents[0].metadata}")
        
        await self.vs_repo.abatch_add_documents(documents)
    ...

###
class Loader:
    def __init__(self, provider, callbacks) -> None:
        self.provider = provider
        self.callbacks = callbacks or {}
    ...
    
    def load_webpage(self, target_url: str):
        loader = PlaywrightURLLoader(
            urls=[target_url], 
            headless=True
        )

        documents = loader.load()
        
        if not documents:
            print(f"Cảnh báo: Không lấy được nội dung tại {target_url}")
            return []
            
        return documents[0].page_content
    ...


    def load_pdf_Pc(self, file: UploadFile):
        if file.filename and not file.filename.lower().endswith('.pdf'):
            raise ValueError(f"File {file.filename} is not a PDF file")

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        loader = UnstructuredLoader(
            file_path="data.pdf",
            mode="paged"
        )
        documents = loader.load()
        
        formatted_docs = []
        for doc in documents:
            metadata = {
                "page_number": doc.metadata.get("page_number", 1),
                "source": file.filename,
                "language": doc.metadata.get("languages", ["unknown"])[0],
                "content_type": file.content_type
            }
            
            formatted_doc = Document(
                page_content=doc.page_content,
                metadata=metadata
            )
            formatted_docs.append(formatted_doc)
        
        os.unlink(temp_path)
        return formatted_docs
        ...


###
law_separators=[
    r'(?=\n\nĐiều\s+\d+)', 
    r'(?=\nĐiều\s+\d+)',

    r'(?=\n\d+\.\s)', 
    
    "\n\n",
    "\n",
    " ",
]
class Process:

    def __init__(self,         
        provider: BaseChatModel, 
        callbacks: Callback
    ) -> None:
        self.provider = provider
        self.callbacks = callbacks or {}
    ...

    def _split_docs(self, text: str):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.splitter.chunk_size,
            chunk_overlap=config.splitter.chunk_overlap,
            separators=law_separators,
            is_separator_regex=True,
            strip_whitespace=True  
        )

        docs = text_splitter.create_documents([text])

        chunks = [
            ChunkResponse(
                index=i, 
                content=doc.page_content, 
                length=len(doc.page_content)
            )
            for i, doc in enumerate(docs)
        ]

        return chunks
        ...

    def split_by_page(self, docs: List[Document]):

        for doc in docs:
            page_chunks = self._split_docs(doc.page_content)
            print(f" -> Created {len(page_chunks)} chunks")

            for chunk in page_chunks:
                ...
        # page_groups = {}
        # for doc in docs["documents"]:
        #     page_number = doc.metadata.get('page_number', 0)
        #     page_groups.setdefault(page_number, []).append(doc)
        
        # print(f"Grouped into {len(page_groups)} pages]\n\n")

        # all_chunks = []
        # for page_number, page_docs in sorted(page_groups.items()):
        #     print(f"Processing page {page_number} with {len(page_docs)} elements")
            
        #     page_text = "\n\n".join(doc.page_content for doc in page_docs)
        #     print(page_text)
            
        #     page_chunks = self._split_docs(page_text)
            
        #     print(f" -> Created {len(page_chunks)} chunks")
            
        #     for chunk in page_chunks:
        #         if not chunk.metadata:
        #             chunk.metadata = {}

        #         chunk.metadata.update({
        #             "source": docs.get("filename", ""),
        #             "language": docs.get("language", ""),
        #             "content_type": docs.get("content_type", ""),
        #             'page_number': page_number,
        #             'total_elements': len(page_docs),
        #         })

        #         print(f"  Chunk {chunk.index}: {len(chunk.content)} chars")
        #         print(f"   Page: {chunk.metadata['page_number']}")
        #     all_chunks.extend(page_chunks)

        # print(f"Total chunks created: {len(all_chunks)}")

        # return all_chunks
        ...

    def _split_docs_PaC(self, text: str):
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=law_separators,
            is_separator_regex=True,
            strip_whitespace=True
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=law_separators,
            is_separator_regex=True,
            strip_whitespace=True
        )
        
        parent_chunks = parent_splitter.create_documents([text])
        
        child_chunks = []
        for i, parent in enumerate(parent_chunks):
            children = child_splitter.create_documents([parent.page_content])
            for child in children:
                child.metadata = child.metadata or {}
                child.metadata.update({
                    "parent_id": str(i)
                })
            child_chunks.extend(children)
        
        return {
            "parent"   : parent_chunks,
            "children" : child_chunks
        }

    def split_PaC(self, docs: List[Document]):
        parent_chunks = []
        child_chunks = []

        for doc in docs:
            page_chunks = self._split_docs_PaC(doc.page_content)
            print(f" -> Parent {len(page_chunks['parent'])} chunks")
            print(f" -> Children {len(page_chunks['children'])} chunks")

            for idx, parent in enumerate(page_chunks["parent"]):
                parent.metadata = parent.metadata or {}
                parent.metadata.update({
                    "source": doc.metadata.get("source", ""),
                    "language": doc.metadata.get("language", ""),
                    "content_type": doc.metadata.get("content_type", ""),
                    "page_number": doc.metadata.get("page_number", 0),
                    "parent_index": idx,
                })
                parent_chunks.append(parent)
            
            for idx, child in enumerate(page_chunks["children"]):
                child.metadata = child.metadata or {}
                child.metadata.update({
                    "source": doc.metadata.get("source", ""),
                    "language": doc.metadata.get("language", ""),
                    "content_type": doc.metadata.get("content_type", ""),
                    "page_number": doc.metadata.get("page_number", 0),
                })
                child_chunks.append(child)

        print(parent_chunks, "\n\n\n", child_chunks)
###  