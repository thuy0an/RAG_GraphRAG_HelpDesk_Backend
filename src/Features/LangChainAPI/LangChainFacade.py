from fastapi import Depends
from src.SharedKernel.config.LLMConfig import LLMFactory, EmbeddingFactory
from src.SharedKernel.utils.yamlenv import load_env_yaml
from src.Features.LangChainAPI.RAG.GraphRAG import GraphRAG
from src.Features.LangChainAPI.RAG.PaCRAG import PaCRAG

config = load_env_yaml()

class LangChainFacade:
    def __init__(self):
        self.provider = LLMFactory.create(config.llm.provider)
        self.embedding = EmbeddingFactory.create(config.llm.provider)

        self.PaCRAG = PaCRAG(self.provider, self.embedding)
        self.GraphRAG = GraphRAG(self.provider, self.embedding)
    

