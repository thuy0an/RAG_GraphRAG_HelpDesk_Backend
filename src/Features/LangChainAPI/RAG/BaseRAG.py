from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.embeddings import Embeddings
from fastapi import UploadFile
from src.SharedKernel.base.Metrics import Metrics
from src.Features.LangChainAPI.RAG.Loader import Loader
from src.Features.LangChainAPI.RAG.Process import Process
from src.Features.LangChainAPI.persistence.MemoryRepository import MemoryRepository
from src.SharedKernel.utils.yamlenv import load_env_yaml


class BaseRAG(ABC):
    def __init__(
        self,
        provider: BaseChatModel,
        embedding: Embeddings
    ) -> None:
        self.provider = provider
        self.embedding = embedding
        self.loader = Loader()
        self.process = Process()
        self.memory_repo = MemoryRepository()

    def _get_history_limit(self) -> int:
        """
        Đọc conversation_history_limit từ config YAML.
        Default = 5 nếu không được định nghĩa.
        0 = disable conversation history (stateless).
        """
        try:
            config = load_env_yaml()
            conv_cfg = getattr(config, "conversational_rag", None)
            return int(getattr(conv_cfg, "conversation_history_limit", 5)) if conv_cfg else 5
        except Exception:
            return 5

    @abstractmethod
    async def index(self, file: UploadFile, **kwargs) -> None:
        """Ingest document vào retriever"""
        pass

    @abstractmethod
    async def retrieve(self, query: str, session_id: str = None, **kwargs) -> AsyncGenerator[str, None]:
        """Retrieve và generate response"""
        pass

    @abstractmethod
    async def delete(self, identifier: str, **kwargs) -> None:
        """Delete documents theo identifier (file_name/source)"""
        pass
