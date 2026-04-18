from typing import List, Optional
from pydantic import BaseModel, Field

class ChunkedUploadRequest(BaseModel):
    """Request for file upload with chunking parameters"""
    
    files: List = Field(..., description="List of files to upload")
    
    parent_chunk_size: Optional[int] = Field(default=2048, description="Parent chunk size in characters")
    parent_chunk_overlap: Optional[int] = Field(default=400, description="Parent chunk overlap in characters")

    child_chunk_size: Optional[int] = Field(default=512, description="Child chunk size in characters")
    child_chunk_overlap: Optional[int] = Field(default=100, description="Child chunk overlap in characters")

class ChunkedUploadResponse(BaseModel):
    """Response for chunked upload with indexing results"""
    
    uploaded_files: List[dict] = Field(..., description="List of uploaded file information")
    indexing_results: List[dict] = Field(..., description="Indexing results for each file")
    chunk_parameters: dict = Field(..., description="Chunk parameters used for indexing")
