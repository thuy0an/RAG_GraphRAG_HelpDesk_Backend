# import json
# import os
# from typing import Any, Dict, List
# from fastapi import Depends, File, UploadFile
# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
# from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
# from langchain_core.runnables import RunnableWithMessageHistory
# from langchain_core.runnables.history import MessagesOrDictWithMessages
# from langchain_core.tools import tool
# from numpy.ctypeslib import as_array
# from src.Features.LangChainAPI.LangTools import YouTubeVideo, add, crawl_tool, crawl_tool_stream, duckduckgo_search_tool, test
# from src.Features.AI_API.config.AI_Config import AIConfig
# from src.Features.LangChainAPI.LangChainDTO import ChatRequest, ChatTechniqueRequest, ChatTemplateRequest, PromptType, TechType, TemplateType
# from src.Features.LangChainAPI.prompt import CoT, System_Instruction, few_shot, youtube_description, youtube_template, zero_shot
# from langchain_community.chat_message_histories import SQLChatMessageHistory
# from langchain_community.document_loaders import UnstructuredPDFLoader
# import tempfile
# import shutil

# logger = get_logger("LangChain Service")

# # # def get_llm():
# # #   config = AIConfig()
# # #   return config

# store: Dict[str, InMemoryChatMessageHistory] = {}

# class LangChainService:
#     # ====================
#     # INIT
#     # ====================
#     def __init__(self, ai_config: AIConfig):       
#         # PROVIDER
#         self.provider = ai_config.create_provider()
#         self.custom_model = ai_config.create_custom()

#         # TOOLS
#         self.tool_list = [
#             add, 
#             test, 
#             crawl_tool
#         ]
#         self.provider_with_tools = self.provider.bind_tools(self.tool_list)

#         self.instruction = ChatPromptTemplate.from_messages([
#             ("system", "Bạn là trợ lý AI"),
#             MessagesPlaceholder("history"),
#             ("human", "{message}")
#         ])

#         db_path = self._create_file("../.data", "memory.db")
#         self.memory_url = f"sqlite:///{db_path}"
#     # ====================
#     # PROMPT
#     # ====================
#     async def prompt(self, req: ChatRequest):
#         result = await self.provider.ainvoke([System_Instruction(req)])
#         return result.content

#     async def stream_prompt(self, req: ChatRequest):
#         async for chunk in self.provider.astream([System_Instruction(req)]):
#             if chunk.content:
#                 yield chunk.content
            
#     async def prompt_template(self, req: ChatTemplateRequest):
#         if req.template == TemplateType.from_template:
#             template = "Viết một blog ngắn nói về {topic}" 
#             _prompt = PromptTemplate.from_template(template)
#             prompt = _prompt.format(topic=req.message)
#             async for chunk in self.provider.astream(prompt):
#                 if chunk.content:
#                     yield chunk.content
#         elif req.template == TemplateType.prompt_template:
#             template = "Viết một blog ngắn nói về {topic} khoảng {words}"
#             _prompt = PromptTemplate(
#                 # nội dung chính
#                 template=template,
#                 # biến đầu vào
#                 input_variables=['topic'],
#                 # định nghĩ kiểu dữ liệu
#                 input_types={},
#                 # biến cố định
#                 partial_variables={'words': '300 từ'}
#             )
#             prompt = _prompt.format(topic=req.message)

#             # hoặc
#             # prompt = _prompt.partial(words="300 từ")
#             # prompt = _prompt.format(topic=req.message)
#             async for chunk in self.provider.astream(prompt):
#                 yield chunk
#         elif req.template == TemplateType.chat_template:
#             template = ChatPromptTemplate.from_template("Viết một blog ngắn nói về {topic} khoảng {words}")
#             chain = template | self.model 
#             placeholder = {
#                 "topic": req.message,
#                 "words": "300 từ"
#             }
#             async for chunk in chain.astream(placeholder):
#                 yield chunk
#         elif req.template == TemplateType.message_placeholder:
#             template = ChatPromptTemplate.from_messages([
#                 ("system", "Bạn là trợ lý AI"),
#                 MessagesPlaceholder(variable_name="hoi_thoai"),
#                 ("human", "Tóm tắt nội dung trong {so_tu} từ"),
#             ])

#             chain = template | self.model | StrOutputParser()

#             placeholder = {
#                 "so_tu": 20,
#                 "hoi_thoai": [
#                     ("human", 
#                     """Xin chào bạn Teddy, tên bạn là gì thế mình quên rồi :D, 
#                     Viết một blog ngắn nói về manga khoảng 200 từ"""
#                     )
#                 ],
#             }

#             async for chunk in chain.astream(placeholder):
#                 yield chunk

#     # ====================
#     # PROMPT TECHNIQUES
#     # ====================
#     async def promt_techniques(self, req: ChatTechniqueRequest):
#         if req.tech == TechType.zero_shot:
#             async for chunk in self.provider.astream(few_shot):
#                 if chunk.content:
#                     yield chunk.content
#         if req.tech == TechType.few_shot:
#             async for chunk in self.provider.astream(few_shot):
#                 if chunk.content:
#                     yield chunk.content
#         if req.tech == TechType.CoT:
#             async for chunk in self.provider.astream(CoT):
#                 if chunk.content:
#                     yield chunk.content
#         if req.tech == TechType.ReAct:
#             # TODO: Implement ReAct technique

#             async for chunk in self._astream(""):
#                 yield chunk
#         pass
        
    
#     # ====================
#     # TOOLS
#     # tools
#     # tools_stream
#     # ====================
#     async def tools(self):
#         messages = [HumanMessage(content="Tìm kiếm tool, để fetch thông tin từ link sau: https://youtube.fandom.com/wiki/BHGaming")]

#         while True:
#             result = await self.provider_with_tools.ainvoke(messages)

#             if not result.tool_calls:
#                 yield result.content
#                 break
                
#             messages.append(result)
        
#             for tool_call in result.tool_calls:
#                 logger.info(f"Tool call: {tool_call}")
#                 tool_output = await self._run_tool(tool_call)
#                 messages.append(
#                     ToolMessage(
#                         content=str(tool_output),
#                         tool_call_id=tool_call["id"]
#                     )
#                 )

#     async def tools_stream(self):
#         messages = [
#             HumanMessage(
#                 content="Tìm kiếm tool, để fetch thông tin từ link sau: https://vnsharing.forumotion.net/t2157-bleach-suzuki-mai, https://vnsharing.forumotion.net/t1941-tsubasa-reservoir-chronicle"
#             )
#         ]

#         result = await self.provider_with_tools.ainvoke(messages)

#         if result.tool_calls:
#             messages.append(result)

#             for tool_call in result.tool_calls:
#                 tool_output = await self._run_tool(tool_call)
#                 messages.append(
#                     ToolMessage(
#                         content=str(tool_output),
#                         tool_call_id=tool_call["id"]
#                     )
#                 )

#         async for chunk in self.provider_with_tools.astream(messages):
#             if chunk.content:
#                 yield chunk.content

#     # ====================
#     # STRUCTED OUTPUT
#     # ====================
#     async def structed_output(self):
#         return {
#             "chain_pydantic_output": await self.chain_pydantic_output(youtube_template, youtube_description),
#             "with_structured_output": await self.with_structured_output_parser(youtube_template, youtube_description),
#             "structured_output_parser": await self.structured_output_parser()
#         }

#     async def chain_pydantic_output(self, instruction: str, input_message: Any):
#         instruction += """
#         {format_instructions}
#         """

#         parser = PydanticOutputParser(pydantic_object=YouTubeVideo)

#         template = PromptTemplate(
#             template=instruction,
#             input_variables=["description"],
#             partial_variables={"format_instructions": parser.get_format_instructions()},
#         )

#         chain = template | self.provider | parser

#         response = await chain.ainvoke(input_message)

#         return response.model_dump()
#     # ====================
#     # MEMORY SERVICE
#     # ====================
#     async def short_chat(self, session_id: str, message: str):
#         chain = self.instruction | self.provider
#         chain_with_memory = RunnableWithMessageHistory(
#             chain,
#             self._get_inmem_session,
#             input_messages_key="message",
#             history_messages_key="history"
#         )

#         input_data = { 
#             "message": message 
#         }
#         config = { 
#             "configurable": { "session_id": session_id } 
#         }
        
#         response = await chain_with_memory.ainvoke(
#             input_data,
#             config=config
#         )

#         return response.content 

#     async def short_chat_no_runnable(self, session_id: str, message: str):
#         history = self._get_inmem_session(session_id)
#         chain = self.instruction | self.provider

#         response = await chain.ainvoke({
#             "history": history.messages,
#             "message": message
#         })
#         logger.info(f"[No runnable]: {response}")

#         history.add_user_message(message)
#         history.add_message(response)

#         return response.content

#     async def long_chat(self, session_id: str, message: str):
#         history = self._get_sql_session_history(session_id, self.memory_url)

#         history.add_user_message(message)

#         all_messages = history.messages

#         logger.info(f"[Long chat] Session {session_id} có {len(all_messages)} messages")
#         for i, msg in enumerate(all_messages):
#             logger.info(f"  [{i}] {msg.type}: {msg.content[:30]}...")
#         system_msg = SystemMessage(content="Bạn là trợ lý AI thông minh")

#         messages_to_llm = [system_msg] + all_messages
#         # logger.info(f" Full msg: {messages_to_llm}")

#         response = await self.provider.ainvoke(messages_to_llm)

#         history.add_message(response)

#         return response.content

#     async def other_chat(self, user_id: str, user_input: str):
#         history = self._get_session_history(user_id)

#         past_messages = history.messages

#         system_msg = SystemMessage(content="Bạn là trợ lý AI, hãy nhớ thông tin từ các tin nhắn trước.")
#         messages = [system_msg] + past_messages + [HumanMessage(content=user_input)]

#         print("DEBUG:", [(m.type, m.content) for m in messages])

#         response = await self.provider.ainvoke(messages)

#         history.add_user_message(user_input)
#         history.add_ai_message(response.content)

#         return response.content

#         pass
#     def chat(self, user_id: str, user_input: str) -> str:
#         response = self.chain_history.invoke(
#             {"input": user_input},
#             config={"configurable": {"session_id": user_id}}
#         )
#         return response.content
#     # ====================
#     # DOCUMENT LOADER
#     # ====================
#     def load_pdf(self, files: List[UploadFile]):
#         for file in files:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#                 shutil.copyfileobj(file.file, temp_file)
#                 temp_file_path = temp_file.name
#             try:
#                 elements = partition_pdf(temp_file_path, extract_tables=True)
                
#                 for element in elements:
#                     if hasattr(element, 'text'):
#                         logger.info(f"Text: {element.text}")
#                         yield str(element.text + "\n")
#                     elif hasattr(element, 'table'):
#                         logger.info(f"Table: {element.table}")
#                         yield str(element.table + "\n")
                        
#             finally:
#                 os.unlink(temp_file_path)
#     # ====================
#     # UTILS
#     # ====================
#     def _get_inmem_session(self, session_id: str) -> InMemoryChatMessageHistory:
#         history = store.get(session_id, InMemoryChatMessageHistory())
#         logger.info(f"Session {session_id} hiện có {len(history.messages)} tin nhắn.")

#         if session_id not in store:
#             store[session_id] = InMemoryChatMessageHistory()
#             logger.info(f"Tạo session mới: {session_id}")
#         else:
#             logger.info(f"Session {session_id} hiện có {len(store[session_id].messages)} tin nhắn.")

#             for i, msg in enumerate(history.messages):
#                 logger.info(f"[{i}] {msg.type}: {msg.content}")
        
#         return store[session_id]

#     def _get_sql_session_history(self, session_id: str, connection_url: str):
#         return SQLChatMessageHistory(
#             session_id=session_id,
#             connection=connection_url
#         )
#         pass

#     def _create_file(self, path: str, file: str):
#         project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#         data_dir = os.path.join(project_root, path)
        
#         os.makedirs(data_dir, exist_ok=True)
        
#         db_path = os.path.join(data_dir, file)
#         return db_path

#     async def _run_tool(self, tool_call):
#         try:
#             for tool in self.tool_list:
#                 if tool.name == tool_call["name"]:
#                     args = tool_call["args"]
#                     return await tool.ainvoke(args)
#         except Exception as e:
#             logger.error(f"{e}")
#             return f"Error: {str(e)}"


from pipes import Template
from typing import Any
from Features.LangChainAPI.LangChainDTO import Callback, ChatRequest, ChatTechniqueRequest, ChatTemplateRequest, TechType, TemplateType
from Features.LangChainAPI.prompt import COT_PROMPT, FEW_SHOT_PROMPT, REACT_PROMPT, YOUTUBE_DESCRIPTION, YOUTUBE_TEMPLATE, ZERO_SHOT_PROMPT, System_Instruction
from Features.LangChainAPI.service.LPIService import LPIService
from Features.LangChainAPI.service.MemoryService import MemoryService
from Features.LangChainAPI.service.OutputService import OutputService
from Features.LangChainAPI.service.PromptService import PromptService
from Features.LangChainAPI.service.ToolService import ToolService
from SharedKernel.AIConfig import AIConfigFactory
from SharedKernel.persistence.Decorators import Service
from SharedKernel.utils.yamlenv import load_env_yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

config = load_env_yaml()

@Service
class LangChainFacade:
    def __init__(self,
        aiconfig_factory: AIConfigFactory
    ):       
        self.ai_factory = aiconfig_factory
        ai_config = AIConfigFactory.create(config.ai.llm_provider)
        self.provider = ai_config.create_provider()

        self.callbacks = Callback(
            ainvoke=self.ainvoke,
            astream=self.astream
        )
        
        self.prompt = PromptService(self.provider, self.callbacks)
        self.tools = ToolService(self.provider, self.callbacks)
        self.output_parser = OutputService(self.provider, self.callbacks)
        self.memory = MemoryService(self.provider, self.callbacks)
        self.LPI = LPIService(self.ai_factory, self.provider, self.callbacks)
    
    async def ainvoke(self, provider: Any, req: Any):
        result = await provider.ainvoke(req)
        if hasattr(result, 'content') and result.content:
            return result.content
        return result

    async def astream(self, provider: Any, req: Any) -> dict:
        async def gen(req):
            async for chunk in provider.astream(req):
                if chunk.content:
                    yield chunk.content

        return { "content": gen(req) }
    

