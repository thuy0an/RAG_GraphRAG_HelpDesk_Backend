from typing import Any
from Features.LangChainAPI.LangChainDTO import ChatRequest, ChatTechniqueRequest, ChatTemplateRequest, TechType, TemplateType, YouTubeVideo
from Features.LangChainAPI.prompt import COT_PROMPT, FEW_SHOT_PROMPT, REACT_PROMPT, YOUTUBE_DESCRIPTION, YOUTUBE_TEMPLATE, ZERO_SHOT_PROMPT, System_Instruction
from SharedKernel.utils.yamlenv import load_env_yaml
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

class OutputService:
    def __init__(self, provider, callbacks) -> None:
        self.provider = provider
        self.callbacks = callbacks or {}

    async def structed_output(self) -> dict:
        async def chain_pydantic_output(instruction: str, input_message: Any):
            instruction += """
            {format_instructions}
            """

            parser = PydanticOutputParser(pydantic_object=YouTubeVideo)

            template = PromptTemplate(
                template=instruction,
                input_variables=["description"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )

            chain = template | self.provider | parser

            response = await self.callbacks["ainvoke"](chain, input_message)

            return response.model_dump()

        return {
            "chain_pydantic_output": await chain_pydantic_output(YOUTUBE_TEMPLATE, YOUTUBE_DESCRIPTION),
            # "with_structured_output": await self.with_structured_output_parser(YOUTUBE_TEMPLATE, YOUTUBE_DESCRIPTION),
            # "structured_output_parser": await self.structured_output_parser()
        }