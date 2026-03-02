import json
import os
import shutil
import tempfile
from typing import List
from Features.LangChainAPI.RedisVectorRepo import RedisVectorRepo
from Features.LangChainAPI.LangChainDTO import ChunkResponse, SplitRequest
from SharedKernel.utils.yamlenv import load_env_yaml
from langchain_community.document_loaders import PlaywrightURLLoader, UnstructuredPDFLoader
from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter

config = load_env_yaml()

class LPIService:
    def __init__(self, ai_factory, provider, callbacks) -> None:
        self.provider = provider
        self.callbacks = callbacks or {}
        self.loader = LoaderService(self.provider, self.callbacks)
        self.process = ProcessService(self.provider, self.callbacks)
        self.vector_store_repo = RedisVectorRepo(ai_factory)
    ...

    async def pdf_handler(self, file: UploadFile):
        doc = self.loader.load_pdf(file)
        if doc:
            print(f"Document loaded successfully")
        chunks = self.process.split_process(SplitRequest(text=doc[0].page_content))
        if len(chunks) > 0:
            print(f"Chunks created: {len(chunks)} chunks")
        chunks_content = [chunk.content for chunk in chunks]
        await self.vector_store_repo.add_documents(chunks_content)
        ...
    ...
class LoaderService:
    def __init__(self, provider, callbacks) -> None:
        self.provider = provider
        self.callbacks = callbacks or {}
    ...
    
    def load_webpage(self, target_url: str):
        try :
            loader = PlaywrightURLLoader(
                urls=[target_url], 
                headless=True,
                # remove_selectors=["header", "footer", "nav"]
            )

            documents = loader.load()
            
            if not documents:
                print(f"Cảnh báo: Không lấy được nội dung tại {target_url}")
                return []
                
            return documents[0].page_content
        except Exception as e:
            print(f"Lỗi khi tải trang động {target_url}: {str(e)}")
            return []
    ...

    def load_pdf(self, file: UploadFile):
        if file.filename and not file.filename.lower().endswith('.pdf'):
            raise ValueError(f"File {file.filename} is not a PDF file")

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        loader = UnstructuredPDFLoader(temp_path, mode="single")
        documents = loader.load()

        os.unlink(temp_path)
            
        return documents
    ...
...

class ProcessService:
    def __init__(self, provider, callbacks) -> None:
        self.provider = provider
        self.callbacks = callbacks or {}
    ...

    def split_process(self, req: SplitRequest):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.splitter.chunk_size,
            chunk_overlap=config.splitter.chunk_overlap,
            separators=config.splitter.separators,
            length_function=len,
            is_separator_regex=False
        )

        docs = text_splitter.create_documents([req.text])

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
...
class IngestService:
    def __init__(self, provider, callbacks) -> None:
        
        ...
...