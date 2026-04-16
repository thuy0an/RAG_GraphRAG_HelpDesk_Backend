from abc import ABC, abstractmethod
from SharedKernel.utils.yamlenv import load_env_yaml

config = load_env_yaml()


class AIConfig(ABC):
    @abstractmethod
    def create_provider(self):
        pass

    @abstractmethod
    def create_embedding(self):
        pass


class OllamaConfig(AIConfig):
    def __init__(self):
        self.host = config.ai.ollama.host
        self.model = config.ai.ollama.model
        self.embed = config.ai.ollama.embed

    def create_provider(self):
        from langchain_ollama import ChatOllama
        return ChatOllama(base_url=self.host, model=self.model)

    def create_embedding(self):
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(base_url=self.host, model=self.embed)


class MistralConfig(AIConfig):
    def __init__(self):
        self.model = config.ai.mistral.model
        self.api_key = config.ai.mistral.api_key
        self.embed = config.ai.mistral.embed

    def create_provider(self):
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=self.model, api_key=self.api_key)

    def create_embedding(self):
        from langchain_mistralai import MistralAIEmbeddings
        return MistralAIEmbeddings(model=self.embed, api_key=self.api_key)


class AIConfigFactory:
    def create(self, provider: str) -> AIConfig:
        if provider == "ollama":
            return OllamaConfig()
        elif provider == "mistral":
            return MistralConfig()
        else:
            raise ValueError(f"Unknown AI provider: {provider}")
