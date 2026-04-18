from fastapi import Depends
from src.SharedKernel.config.LLMConfig import LLMFactory, EmbeddingFactory
from src.SharedKernel.utils.yamlenv import load_env_yaml
from src.Features.LangChainAPI.RAG.GraphRAG import GraphRAG
from src.Features.LangChainAPI.RAG.PaCRAG import PaCRAG
from src.Features.ConversationAPI.ConversationService import ConversationService

config = load_env_yaml()

class LangChainFacade:
    def __init__(self, conversation_service: ConversationService = Depends()):
        self.provider = LLMFactory.create(config.llm.provider)
        self.embedding = EmbeddingFactory.create(config.llm.provider)

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"LangChainFacade DEBUG: conversation_service={conversation_service}, type={type(conversation_service)}")

        self.PaCRAG = PaCRAG(self.provider, self.embedding, conversation_service)
        self.GraphRAG = GraphRAG(self.provider, self.embedding, conversation_service)

        logger.info(f"LangChainFacade DEBUG: PaCRAG.conversation_service={self.PaCRAG.conversation_service}")
        logger.info(f"LangChainFacade DEBUG: GraphRAG.conversation_service={self.GraphRAG.conversation_service}")
    

