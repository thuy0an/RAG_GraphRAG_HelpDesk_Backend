from fastapi import Depends
from SharedKernel.config.AIConfig import AIConfigFactory
from SharedKernel.utils.yamlenv import load_env_yaml
from src.Features.LangChainAPI.RAG.Synthesizer import Synthesizer
from src.Features.LangChainAPI.service.AgentService import AgentService
from src.Features.LangChainAPI.service.PromptService import PromptService
from SharedKernel.threading.ThreadPoolManager import ThreadPoolManager

config = load_env_yaml()

class LangChainFacade:
    def __init__(self,
        config_ai: AIConfigFactory = Depends(),
        thread_pool: ThreadPoolManager = Depends() 
    ):       
        self.ai_factory = config_ai
        ai_config = self.ai_factory.create(config.llm.provider)
        self.provider = ai_config.create_provider()

        self.prompt_service = PromptService()
        self.agent_service = AgentService()
        self.synthesizer = Synthesizer(
            self.ai_factory, 
            self.provider,
            thread_pool    
        )
        ...
    

