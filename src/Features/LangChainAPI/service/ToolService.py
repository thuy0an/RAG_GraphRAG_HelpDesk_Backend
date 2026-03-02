import json
from typing import Any
from Features.LangChainAPI.LangChainDTO import Callback
from Features.LangChainAPI.LangTools import LangTools, ascrawl_web_tool, crawl_web_tool
from SharedKernel.base.Logger import logger
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage

class ToolService:
    def __init__(self, provider: BaseChatModel, callbacks: Callback):
        self.provider = provider
        self.callbacks = callbacks or {}

        self.tool_list = [
            crawl_web_tool
        ]

        self.provider_with_tools = self.provider.bind_tools(self.tool_list)

    async def call_search_tools(self):
        query = "Tìm kiếm tool, để fetch thông tin từ link sau: https://youtube.fandom.com/wiki/BHGaming"
        return await self.handler_tool_call(self.provider_with_tools, query)

    async def handler_tool_call(self, provider: Any, query: str):
            response = await provider.ainvoke([HumanMessage(content=query)])
            
            if not response.tool_calls:
                return response.content

            print(f"Tool: {response.tool_calls}")

            messages = [HumanMessage(content=query), response]

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                logger.info(f"Tool name: {tool_name}")
                logger.info(f"Tool args: {tool_args}")

                for tool in self.tool_list:
                    if tool.name == tool_name:
                        tool_output = await tool.ainvoke(tool_args)

                        messages.append(ToolMessage(
                                content=str(tool_output),
                                tool_call_id=tool_call["id"]
                            ))
                        
            return await self.callbacks.astream(provider, messages)

    