import os
import shutil
import tempfile
import aiofiles
import aiofiles.os
from fastapi import UploadFile
from langchain_community.document_loaders import (
    PlaywrightURLLoader,
    UnstructuredPDFLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
from src.SharedKernel.exception.APIException import APIException
from src.SharedKernel.base.Metrics import Metrics
from langchain_community.document_loaders import BSHTMLLoader
from SharedKernel.base.Logger import get_logger

logger = get_logger(__name__)

class Loader:
    def __init__(self) -> None:
        self.loaders = {
            ".pdf": self.load_pdf,
            ".doc": self.load_docx,
            ".docx": self.load_docx,
            ".txt": self.load_txt,
            ".md": self.load_txt,
            ".html": self.load_html,
        }

    def load_file(self, file: UploadFile):
        """Generic file loader - auto detect file type"""
        ext = os.path.splitext(file.filename.lower())[1] if file.filename else ".txt"
        loader_func = self.loaders.get(ext, self.load_txt)
        return loader_func(file)

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

    async def load_pdf(self, file: UploadFile):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
            temp_file.close()

        file_size_kb = os.path.getsize(temp_path) / 1024
        suffix = os.path.splitext(file.filename)[1].upper() if file.filename else ".PDF"
        path = os.path.basename(temp_path)

        logger.info(f"DOCUMENT LOADER:File:  {file.filename} - Type:  {suffix} - Size:  {file_size_kb:.1f} KB")

        loader = UnstructuredPDFLoader(temp_path, mode="paged", strategy="fast")
        documents = loader.load()

        formatted_docs = []
        for doc in documents:
            metadata = {
                "page_number": doc.metadata.get("page_number", 1),
                "source": file.filename,
                "language": doc.metadata.get("languages", ["unknown"])[0],
                "content_type": file.content_type,
            }

            formatted_doc = Document(page_content=doc.page_content, metadata=metadata)
            formatted_docs.append(formatted_doc)
        os.unlink(temp_path)

        return formatted_docs

    async def load_docx(self, file: UploadFile):
        if file.filename and not file.filename.lower().endswith((".doc", ".docx")):
            raise APIException(f"File {file.filename} is not a DOCX file")

        suffix = ".docx" if file.filename and file.filename.lower().endswith(".docx") else ".doc"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
            temp_file.close()

        file_size_kb = os.path.getsize(temp_path) / 1024
        suffix_label = os.path.splitext(file.filename)[1].upper() if file.filename else ".DOCX"
        logger.info(
            f"DOCUMENT LOADER\nFile:  {file.filename}\nType:  {suffix_label}\nSize:  {file_size_kb:.1f} KB"
        )

        loader = Docx2txtLoader(temp_path)
        documents = loader.load()

        formatted_docs = []
        for doc in documents:
            metadata = {
                "page_number": 1,
                "source": file.filename,
                "language": doc.metadata.get("language", "unknown"),
                "content_type": file.content_type,
            }
            formatted_doc = Document(page_content=doc.page_content, metadata=metadata)
            formatted_docs.append(formatted_doc)

        os.unlink(temp_path)
        return formatted_docs

    async def load_txt(self, file: UploadFile):
        content = file.file.read().decode("utf-8", errors="ignore")
        return [
            Document(
                page_content=content,
                metadata={"source": file.filename, "content_type": file.content_type},
            )
        ]

    async def load_html(self, file: UploadFile):
        content = file.file.read().decode("utf-8", errors="ignore")
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp_file:
            temp_file.write(content.encode())
            temp_path = temp_file.name
            temp_file.close()

        loader = BSHTMLLoader(temp_path)
        documents = loader.load()
        os.unlink(temp_path)

        for doc in documents:
            doc.metadata["source"] = file.filename
        return documents
