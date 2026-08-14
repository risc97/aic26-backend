from pydantic import BaseModel, Field
from typing import List, Literal
from datetime import datetime

from config import DEFAULT_MODEL
ModelName = Literal["c2lip", "siglip2"]

class Item(BaseModel):
    keyframe_id: str
    video_id: str

class QueryRequest(BaseModel):
    query: str = Field(..., description="Truy vấn đoạn video cần tìm")
    limit: int = Field(100, description="Số lượng kết quả trả về")
    mode: str = Field("keyframe", description="Chế độ search")
    model: ModelName = Field(DEFAULT_MODEL, description="Model dùng để search")

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

class LogEntry(QueryRequest, QueryResponse):
    request_id: str
    timestamp: datetime | None

class TemporalStage(BaseModel):
    query: str = Field(..., description="Sự kiện cần tìm ở bước này")
    variants: List[str] = Field(
        default_factory=list,
        description="Cách diễn đạt khác của cùng sự kiện, gộp bằng max",
    )

class GapSpec(BaseModel):
    min_frames: int = Field(1, ge=1, description="Cách stage trước ít nhất bao nhiêu keyframe")
    max_frames: int | None = Field(None, description="Cách stage trước nhiều nhất bao nhiêu keyframe")
    min_gap_ms: int = Field(0, ge=0)
    max_gap_ms: int | None = Field(120_000)

class TemporalQueryRequest(BaseModel):
    stages: List[TemporalStage] = Field(..., min_length=2, description="Chuỗi sự kiện, đúng thứ tự")
    limit: int = Field(20, description="Số video trả về")
    model: ModelName = Field(DEFAULT_MODEL, description="Model dùng để search")
    candidate_k: int = Field(2000, description="Độ sâu recall mỗi stage")
    gaps: List[GapSpec] | None = Field(
        None, description="Ràng buộc khoảng cách, một phần tử cho mỗi bước chuyển (len = len(stages) - 1)",
    )
    require_all_stages: bool = Field(False, description="Bắt buộc mọi stage phải khớp")
    skip_penalty: float = Field(0.5, description="Phạt khi một stage không khớp, bỏ qua nếu require_all_stages")
    span_penalty: float = Field(0.0, description="Phạt theo z-unit trên mỗi phút độ dài chuỗi")

class TemporalMatch(Item):
    stage: int
    query: str
    score: float
    timestamp_ms: int

class TemporalItem(BaseModel):
    rank: int
    video_id: str
    score: float
    matches: List[TemporalMatch] = Field(default_factory=list)
    skipped_stages: List[int] = Field(default_factory=list)

class TemporalQueryResponse(BaseModel):
    results: List[TemporalItem] = Field(default_factory=list)
    total: int = 0