"""Script tạo tất cả tables trong MySQL từ SQLModel entities"""
import asyncio
import sys
sys.path.insert(0, "src")

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from SharedKernel.utils.yamlenv import load_env_yaml

# Import tất cả entities để SQLModel biết cần tạo tables nào
from Domain.base_entities import Accounts, Departments, Tickets, Messages, Attachment
from Domain.history_entities import *

async def init_db():
    config = load_env_yaml()
    engine = create_async_engine(config.database.mysql.url, echo=True)
    
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    
    await engine.dispose()
    print("\n✓ Tất cả tables đã được tạo thành công!")

asyncio.run(init_db())
