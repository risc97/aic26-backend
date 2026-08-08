from pydantic import BaseModel, Field
from typing import List

class Item(BaseModel):
    keyframe_id: str
    video_id: str
    
class QueryRequest(BaseModel):
    query_id: str
    query: str = Field(..., description="Truy vấn đoạn video cần tìm")
    limit: int = Field(100, description="Số lượng kết quả trả về")
    mode: str = Field("keyframe", description="Chế độ search")

class QueryResponse(BaseModel):
    results: List[Item] = Field(default_factory=list)
    total: int = 0

class Video(BaseModel):
    video_id: str
    duration_ms: int
    fps: float
    keyframe_count: int | None = None

class Keyframe(BaseModel):
    keyframe_id: int
    video_id: str
    frame_idx: int
    timestamp_ms: int
    segment_id: int
    image_url: str
    video_url: str

class SearchHit(Keyframe):
    rank: int
    score: float