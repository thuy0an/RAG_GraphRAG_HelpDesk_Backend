# # import json
# # from typing import List, Optional
# # from fastapi import APIRouter, Depends, File, Form, UploadFile
# # from starlette import status
# # from src.Shared.base import get_logger
# # from src.Features.TicketAPI.TicketDTO import TicketCreateDTO, TicketSearchRequest, TicketUpdateDTO
# # from src.Shared.base.APIResponse import APIResponse
# # from src.Features.TicketAPI.TicketService import TicketService

# # logger = get_logger(__name__)

# # router = APIRouter(
# #     prefix="/api/v1/tickets",
# #     tags=["Ticket"]
# # )

# # @router.get("/")
# # async def get_ticket(
# #     req: TicketSearchRequest = Depends(),
# #     service: TicketService = Depends()
# # ):
# #     result = await service.search(req)

# #     return APIResponse(
# #         message="Get tickets",
# #         status_code=status.HTTP_200_OK,
# #         data=result
# #     )

# # @router.post("/")
# # async def create_ticket(
# #     dto: TicketCreateDTO,
# #     service: TicketService = Depends()
# # ):
# #     result = await service.create_ticket(dto)
# #     return APIResponse(
# #         message="Create ticket",
# #         status_code=status.HTTP_201_CREATED,
# #         data=result
# #     )

# # @router.post("/form")
# # async def create_ticket_form(
# #     ticket: str | None = Form(...),
# #     attachments: List[UploadFile] | None = File(None),
# #     service: TicketService = Depends()
# # ):
# #     result = None
# #     if ticket:
# #         ticket_data = json.loads(ticket)
# #         dto = TicketCreateDTO(**ticket_data)
# #         result = await service.create_ticket_with_attachments(dto, attachments)
# #     return APIResponse(
# #         message="Create ticket",
# #         status_code=status.HTTP_201_CREATED,
# #         data=result
# #     )

# # @router.put("/form/{id}")
# # async def edit_ticket_form(
# #     id: str,
# #     ticket: str | None = Form(None),
# #     attachments: List[UploadFile] | None = File(None),
# #     service: TicketService = Depends()
# # ):
# #     ticket_data = json.loads(ticket)
# #     dto = TicketUpdateDTO(**ticket_data)
# #     await service.edit_ticket_with_attachments(id, dto, attachments)
# #     return APIResponse(
# #         message="Cap nhat ticket thanh cong",
# #         status_code=status.HTTP_201_CREATED
# #     )

# # @router.post("/raw")
# # async def create_raw(
# #     service: TicketService = Depends()
# # ):
# #     result = await service.create_raw()
# #     return APIResponse(
# #         message="Create ticket",
# #         status_code=status.HTTP_201_CREATED,
# #         data=result
# #     )

# # @router.put("/")
# # async def edit_ticket(
# #     id: str,
# #     dto: TicketUpdateDTO,
# #     service: TicketService = Depends()
# # ):
# #     result = await service.edit_ticket(id, dto)
# #     return APIResponse(
# #         message="Edit account",
# #         status_code=status.HTTP_200_OK,
# #         data=result
# #     )

# # @router.delete("/{id}")
# # async def soft_delete_account(
# #     id: str,
# #     service: TicketService = Depends()
# # ):
# #     result = await service.soft_delete_ticket(id)
# #     return APIResponse(
# #         message="Delete account",
# #         status_code=status.HTTP_400_BAD_REQUEST,
# #         data=result
# #     )

from fastapi import APIRouter, FastAPI, Request, status
from fastapi.params import Depends
from lagom import Container
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from Features.TicketAPI.TicketDTO import TicketCreateDTO, TicketSearchRequest
from Features.TicketAPI.TicketService import TicketService
from SharedKernel.base.APIResponse import APIResponse
from SharedKernel.persistence import PersistenceManager
from SharedKernel.persistence.Decorators import Controller
from SharedKernel.persistence.PersistenceManager import PersistenceManagerFactory
from SharedKernel.utils.yamlenv import load_env_yaml

config = load_env_yaml()

@Controller
class TicketController:
    def __init__(self, app: FastAPI, container: Container) -> None:
        self.app = app
        self.container = container 
        self.router = APIRouter(
            prefix="/api/v1/tickets",
            tags=["Ticket"]
        )
        self.register_route()
        self.app.include_router(self.router)

    def register_route(self):
        def ticket_service(): return self.container[TicketService]
        
        @self.router.get("/")
        async def get_ticket(request: Request, query_dto: TicketSearchRequest = Depends()):
            result = await ticket_service().search(query_dto)

            return APIResponse(
                message="Get tickets",
                status_code=status.HTTP_200_OK,
                data=result
            )

        @self.router.post("/")
        async def create_ticket(dto: TicketCreateDTO):
            result = await ticket_service().create_ticket(dto)

            return APIResponse(
                message="Create ticket",
                status_code=status.HTTP_201_CREATED,
                data=result
            )

        