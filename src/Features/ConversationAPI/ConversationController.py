from fastapi import APIRouter, Depends, FastAPI, status
from src.Features.ConversationAPI.ConversationService import ConversationService
from src.Features.ConversationAPI.ConversationDTO import AddConversationHistoryRequest, SessionListResponse, SearchConversationHistoriesRequest
from src.SharedKernel.base.APIResponse import APIResponse
from src.SharedKernel.persistence.Decorators import Controller

@Controller
class ConversationController:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.router = APIRouter(prefix="/api/v1/conversations_history", tags=["Conversation History"])
        self.setup_routes()
        self.app.include_router(self.router)
    
    def setup_routes(self):
        @self.router.post("/", response_model=APIResponse)
        async def add_conversation_history(request: AddConversationHistoryRequest, conversation_service: ConversationService = Depends()):
            result = await conversation_service.add_conversation_history(
                request.session_id, 
                request.role, 
                request.content
            )
            return APIResponse(
                message="Message added successfully", 
                result=result,
                status_code=status.HTTP_200_OK
            )
        
        @self.router.get("/{session_id}", response_model=APIResponse)
        async def get_conversation_histories(
            session_id: str,
            req: SearchConversationHistoriesRequest = Depends(),
            conversation_service: ConversationService = Depends()
        ):
            result = await conversation_service.get_conversation_histories(
                session_id,
                req
            )
            return APIResponse(
                message="Messages retrieved successfully",
                result=result,
                status_code=status.HTTP_200_OK
            )
        
        @self.router.delete("/{session_id}", response_model=APIResponse)
        async def clear_conversation_messages(session_id: str, conversation_service: ConversationService = Depends()):
            result = await conversation_service.clear_conversation_histories_by_session(session_id)
            return APIResponse(
                    message="Messages cleared successfully", 
                    result=result,
                    status_code=status.HTTP_200_OK   
                )
        
        @self.router.get("/sessions", response_model=APIResponse)
        async def get_all_sessions(conversation_service: ConversationService = Depends()):
            result = await conversation_service.get_all_sessions()
            return APIResponse(
                message="Sessions retrieved successfully", 
                result={
                    "sessions": result, 
                    "total_count": len(result),
                },
                status_code=status.HTTP_200_OK   
            )
