from pydantic import BaseModel, Field
from typing import List, Literal
from datetime import datetime

from config import DEFAULT_MODEL
ModelName = Literal["siglip", "siglip2"]
SearchMode = Literal["keyframe", "transcript_semantic", "transcript_exact"]

class Item(BaseModel):
    keyframe_id: str
    video_id: str
    timestamp_ms: int

class QueryRequest(BaseModel):
    query: str = Field(..., description="Truy vấn đoạn video cần tìm")
    limit: int = Field(100, description="Số lượng kết quả trả về")
    mode: SearchMode = Field("keyframe", description="Chế độ search")
    model: ModelName | None = Field(
        DEFAULT_MODEL,
        description="Model dùng để search",
    )

class QueryResponse(BaseModel):
    results: List[Item] = Field(default_factory=list)
    total: int = 0

class LogEntry(QueryRequest, QueryResponse):
    request_id: str
    timestamp: datetime | None
