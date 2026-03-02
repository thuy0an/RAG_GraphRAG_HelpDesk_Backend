import builtins
import json
import os
from typing import Any, Callable, List, Iterator, Optional, Dict, Sequence
from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
import requests
from src.SharedKernel.base.Logger import get_logger

load_dotenv()

logger = get_logger(__name__)

class Ollama(BaseChatModel, Embeddings):
    host: str = os.getenv("OLLAMA_HOST")
    model: str = os.getenv("OLLAMA_MODEL")

    def __init__(self):
        super().__init__()
        logger.info(f"Initialized Ollama with model: {self.model}")
    
    @property
    def _llm_type(self) -> str:
        return "ollama-requests"

    def _generate(
        self, 
        messages: list[BaseMessage], 
        stop: list[str] | None = None, 
        run_manager: CallbackManagerForLLMRun | None = None, 
        **kwargs: Any
    ) -> ChatResult:

        # TODO: Implement the actual LLM call logic here
        # For now, return a placeholder response

        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model,
            "prompt": f"{messages}",
            "stream": False
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

        return data.get("response", "")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents(texts)
        
    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(text)