from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import List, Literal
from datetime import datetime

from config import DEFAULT_MODEL
ModelName = Literal["siglip", "siglip2", "gte", "pe"]
SearchMode = Literal["keyframe", "transcript_semantic", "transcript_exact",
                     "ocr_exact", "temporal"]

class Item(BaseModel):
    keyframe_id: str
    video_id: str
    timestamp_ms: int
    frame_idx: int
    video_fps: float
    score: float

class TranscriptItem(BaseModel):
    video_id: str
    transcript_id: str
    text: str
    time_start_ms: int
    time_end_ms: int
    keyframes: List[Item]

class OcrItem(Item):
    text: str
    score: float

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


class KeyframeListResponse(BaseModel):
    video_id: str
    total: int = 0
    keyframes: List[Item] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str = Field(..., description="Truy vấn đoạn video cần tìm")
    limit: int = Field(100, description="Số lượng kết quả trả về")
    mode: SearchMode = Field("keyframe", description="Chế độ search")
    model: ModelName | None = Field(
        DEFAULT_MODEL,
        description="Model dùng để search",
    )
    phrase: bool = Field(
        False,
        description="Chỉ dùng cho mode exact: True = các từ phải liền nhau đúng "
                    "thứ tự ('sạt lở'), False = keyframe chứa đủ các từ ở bất kỳ đâu",
    )

    @model_validator(mode="after")
    def validate_mode_and_model(self) -> "QueryRequest":
        if self.mode == "keyframe" and self.model not in ("siglip", "siglip2", "pe"):
            raise ValueError(
                f"When mode is 'keyframe', model must be 'siglip', 'siglip2' or 'pe', got '{self.model}'"
            )

        if self.mode == "transcript_semantic" and self.model != "gte":
            raise ValueError(
                f"When mode is 'transcript_semantic', model must be 'gte', got '{self.model}'"
            )

        return self

class QueryResponse(BaseModel):
    results: List[Item] | List[TranscriptItem] | List[TemporalItem] | List[OcrItem] = Field(default_factory=list)
    total: int = 0

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
    max_skips: int | None = Field(
        0, ge=0,
        description="Số stage tối đa được phép bỏ qua; 0 = mọi stage phải khớp, null = không giới hạn",
    )
    skip_penalty: float = Field(0.5, description="Phạt khi một stage không khớp, trong giới hạn max_skips")
    score_cap: float | None = Field(
        None, gt=0,
        description="Giới hạn điểm mỗi stage ở +-cap z-unit, tránh một keyframe lấn át cả chuỗi; null = không giới hạn",
    )
    span_penalty: float = Field(0.0, description="Phạt theo z-unit trên mỗi phút độ dài chuỗi")
    beta: float | None = Field(
        None, gt=0,
        description="Tham số beta cho biến đổi log(1 + beta * sim); null = không áp dụng",
    )

class TemporalQueryResponse(BaseModel):
    results: List[TemporalItem] = Field(default_factory=list)
    total: int = 0
