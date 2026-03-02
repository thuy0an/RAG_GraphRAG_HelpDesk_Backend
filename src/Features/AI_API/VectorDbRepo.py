# import redis
# from typing import List
# from src.Features.LangChain.VectorDbConfig import VectorDBConfig
# config = VectorDBConfig()

# class VectorDBRepo:
#     def __init__(self):
#         self.client = config.client

#     async def create_index(self):
#         try:
#             await self.client.execute_command(
#                     "FT.CREATE", config.index_name,
#                     "ON", "HASH",
#                     "PREFIX", "1", config.prefix,
#                     "SCHEMA",
#                     config.content_field, "TEXT",
#                     config.vector_field, "VECTOR", config.index_method, "6",
#                     "TYPE", "FLOAT32",
#                     "DIM", config.vector_dim,
#                     "DISTANCE_METRIC", config.distance_metric
#                 )
#             return True
#         except redis.ResponseError as e:
#                 if "Index already exists" in str(e):
#                     return True
#                 else:
#                     raise 

#     async def add_document(self, doc_id: str, content: str, vector: bytes):
#         key = f"{config.prefix}:{doc_id}" 
#         await self.client.hset(
#             key,
#             mapping={
#                 config.content_field: content,
#                 config.vector_field: vector
#             }
#         )


#     async def search_vectors(self, query_vector: bytes, k: int = 5) -> List[dict]:
#         response = await self.client.execute_command(
#             "FT.SEARCH", config.index_name,
#             f"*=>[KNN {k} @{config.vector_field} $vec AS score]",
#             "SORTBY", "score",
#             "RETURN", "2", config.content_field, "score",
#             "PARAMS", "2", "vec", query_vector,
#             "DIALECT", "2"
#         )

#         results = []
#         for i in range(1, len(response), 2):
#             fields = response[i + 1]
#             results.append({
#                 "content": fields[1].decode(),
#                 "score": float(fields[3]),
#             })
#         return results