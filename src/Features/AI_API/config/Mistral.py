import builtins
import json
import os
from typing import Any, Callable, List, Iterator, Sequence
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

class Mistral(BaseChatModel, Embeddings):
    model: str = os.getenv("MISTRAL_MODEL", "")
    embedding_model: str = os.getenv("MISTRAL_EMBEDDING_MODEL", "")

    api_key: str = os.getenv("MISTRAL_API_KEY", "")
    base_url: str = "https://api.mistral.ai/v1"

    def __init__(self):
        super().__init__()
        logger.info(f"Initialized MistralLLM with model: {self.model}")
  
    @property
    def _llm_type(self) -> str:
        return "mistral-requests"

    # 
    # MAIN GENERATE
    # 
    def _generate(
        self, 
        messages: list[BaseMessage], 
        stop: list[str] | None = None, 
        run_manager: CallbackManagerForLLMRun | None = None, 
        **kwargs: Any
    ) -> ChatResult:

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "tools": kwargs.get("tools"),
            "tool_choice": "auto",
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        message = data["choices"][0]["message"]

        additional_kwargs = {}
        if "tool_calls" in message:
            additional_kwargs["tool_calls"] = message["tool_calls"]

        ai_message = AIMessage(
            content=message.get("content", ""),
            additional_kwargs=additional_kwargs,
        )

        return ChatResult(
            generations=[ChatGeneration(message=ai_message)]
        )
    
    #
    # STREAM
    # 
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        converted = self._convert_messages(messages)
        payload = {
            "model": self.model,
            "messages": converted,
            "stream": True
        }
        
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
            # logger.info(f"Tools payload: {kwargs['tools']}")
    
        # logger.info(f"Full payload: {payload}")
        
        if stop:
            payload["stop"] = stop
        
        response = requests.post(url, json=payload, headers=headers, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if not line:
                continue
            
            decoded = line.decode("utf-8")
            
            if decoded.startswith("data: "):
                decoded = decoded[len("data: "):]
            
            if decoded == "[DONE]":
                break
            
            try:
                data = json.loads(decoded)
                delta = data["choices"][0]["delta"]
                
                # Handle content
                content = delta.get("content", "")
                if content:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content=content)
                    )
                
                tool_calls = delta.get("tool_calls", [])
                if tool_calls:
                    for tool_call in tool_calls:
                        # logger.info(f"Yielding tool call chunk: {tool_call}")
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(
                                content="",
                                additional_kwargs={"tool_calls": [tool_call]}
                            )
                        )
            except Exception as e:
                logger.error(f"{e}")
                continue
    
    def bind_tools(
        self, 
        tools: Sequence[builtins.dict[str, Any] | type | Callable | BaseTool], 
        *, 
        tool_choice: str | None = None, 
        **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.args_schema.model_json_schema() if hasattr(tool, 'args_schema') else {}
                    }
                })
            else:
                formatted_tools.append(tool)

        return self.bind(tools=formatted_tools, **kwargs)

    #
    # UTILS
    #
    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        converted = []

        for m in messages:
            if isinstance(m, HumanMessage):
                converted.append({"role": "user", "content": m.content})

            elif isinstance(m, SystemMessage):
                converted.append({"role": "system", "content": m.content})

            elif isinstance(m, AIMessage):
                msg = {"role": "assistant", "content": m.content}

                if "tool_calls" in m.additional_kwargs:
                    msg["tool_calls"] = m.additional_kwargs["tool_calls"]

                converted.append(msg)

            elif isinstance(m, ToolMessage):
                converted.append({
                    "role": "tool",
                    "content": m.content,
                    "tool_call_id": m.tool_call_id,
                })

        return converted

    def _embed(self, texts: list[str]):
        url = f"{self.mistral_url}/embeddings"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.embedding_model,
            "input": texts
        }

        try:
            res = requests.post(url, json=payload, headers=headers)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

        return data

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(text)