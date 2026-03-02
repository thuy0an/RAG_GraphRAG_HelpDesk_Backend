
import asyncio
from functools import wraps
import functools
import logging
from typing import Any, Callable, Type

def Transactional(cls):
    """
    Decorator để tự wrap tất cả các method async
    của class với decorator `transaction`.
    """
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and asyncio.iscoroutinefunction(attr_value):
            setattr(cls, attr_name, Transaction(attr_value))
    return cls

def Transaction(func: Callable) -> Callable:
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs) -> Any:
        try:
            result = await func(self, *args, **kwargs)
            await self.session.commit()
            print(f"Transaction committed - Session ID: {id(self.session)}")
            return result
        except Exception as e:
            await self.session.rollback()
            print(f"Transaction rolled back - Session ID: {id(self.session)}, Error: {e}")
            raise
        finally:
            await self.session.close()
            print(f"Session closed - Session ID: {id(self.session)}")
    return wrapper

def Service(cls):
    cls.__di_type__ = "service"
    return cls

def Repository(cls):
    cls.__di_type__ = "repository"
    return cls

def Controller(cls: Type[Any]) -> Type[Any]:
    cls.__di_type__ = "controller"
    return cls