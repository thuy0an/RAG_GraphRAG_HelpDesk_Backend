from enum import Enum, StrEnum
from typing import Any, Callable, List, Optional
from pydantic import BaseModel, Field

class PromptType(StrEnum):
    NONE = "none"
    STREAM = "stream"

class ChatRequest(BaseModel):
    message: str

class ChatMessageRequest(BaseModel):
    session_id: str = "user1234"
    message: str

class TemplateType(str, Enum):
    from_template = "from_template"
    prompt_template = "prompt_template"
    chat_template = "chat_template"
    message_placeholder = "message_placeholder"

class ChatTemplateRequest(BaseModel):
    message: str
    template: TemplateType

class TechType(str, Enum):
    ZERO = "zero_shot"
    FEW = "few_shot"
    COT = "CoT"
    REACT = "ReAct"

class ChatTechniqueRequest(BaseModel):
    message: str
    tech: TechType

class StructedOutputType(str, Enum):
    none = "none"
    stream = "stream"

class MemoryType(str, Enum):
    short = "short",
    long = "long"

class YouTubeVideo(BaseModel):
    title: str = Field(description="Tiêu đề video")
    channel: str = Field(description="Tên kênh YouTube")
    views: int = Field(description="Số lượt xem")
    upload_date: str = Field(description="Ngày đăng video")
    is_short: bool = Field(description="Video có phải YouTube Shorts hay không")

class Callback(BaseModel):
    ainvoke: Callable[[Any], Any]
    astream: Callable[[Any], Any]

class SplitRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Văn bản cần tách")
    chunk_size: int = Field(default=1000, ge=50, description="Kích thước tối đa mỗi chunk (ký tự)")
    chunk_overlap: int = Field(default=200, ge=0, description="Số ký tự trùng lặp giữa các chunk")
    separators: Optional[List[str]] = Field(default=None, description="Dấu phân cách tùy chọn")

class ChunkResponse(BaseModel):
    index: int
    content: str
    length: int
