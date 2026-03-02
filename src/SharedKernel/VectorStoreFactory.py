from abc import ABC, abstractmethod
from typing import Dict, Type
from SharedKernel.utils.yamlenv import load_env_yaml
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_redis import RedisVectorStore
from redisvl.schema import IndexSchema


config = load_env_yaml()

class VectoreStoreConfig(ABC):
    @abstractmethod
    def create_vector_store(self, embedding: Embeddings) -> VectorStore:
        pass

class VectoreStoreConfigFactory:
    _registry: Dict[str, Type[VectoreStoreConfig]] = {}

    @classmethod
    def register(cls, type_name: str, ai_class: Type[VectoreStoreConfig]):
        cls._registry[type_name] = ai_class

    @classmethod
    def create(cls, type_name: str) -> VectoreStoreConfig:
        ai_config_class = cls._registry.get(type_name)
        if not ai_config_class:
            raise ValueError(f"Vector store config '{type_name}' is not registered.")
        return ai_config_class()

class RedisVectorConfig(VectoreStoreConfig):
    def __init__(self) -> None:
        self.redis_url = config.redis.url
        self.index_name = config.vector_store.redis.index_name 
        self.prefix = config.vector_store.redis.prefix
        self.content_field = config.vector_store.redis.content_field


    def create_vector_store(self, embedding: Embeddings) -> VectorStore:
        index_schema = IndexSchema.from_dict({
            "index": {
                "name": self.index_name,
                "prefix": self.prefix,
            },
            "fields": [
                {"name": config.vector_store.redis.content_field, "type": "text"},
                {
                    "name": config.vector_store.redis.vector_field,
                    "type": "vector",
                    "attrs": {
                        "dims": config.vector_store.redis.vector_dim,
                        "distance_metric": config.vector_store.redis.distance_metric,
                        "algorithm": config.vector_store.redis.index_method,
                        "datatype": "float32"
                    },
                },
            ],
        })
        
        return RedisVectorStore(
            embeddings=embedding,
            redis_url=self.redis_url,
            index_name=self.index_name,
            index_schema=index_schema,
        )
        ...

class InMemoryVectorConfig(VectoreStoreConfig):
    def __init__(self) -> None:
        pass

    def create_vector_store(self, embedding: Embeddings) -> VectorStore: 
        return InMemoryVectorStore(embedding=embedding)
        ...

VectoreStoreConfigFactory.register("in_mem", InMemoryVectorConfig)
VectoreStoreConfigFactory.register("redis", RedisVectorConfig)
