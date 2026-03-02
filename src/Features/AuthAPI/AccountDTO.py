from typing import Optional
from pydantic import BaseModel

class AccountBaseDTO(BaseModel):
    # email: str
    pass

class AccountCreateDTO(AccountBaseDTO):
    username: str
    password: str

class AccountUpdateDTO(AccountBaseDTO):
    username: str
    password: str

class AccountLoginDTO(AccountBaseDTO):
    username: str
    password: str

class AccountSearchRequest(BaseModel):
    page: int = 1
    page_size: int = 10