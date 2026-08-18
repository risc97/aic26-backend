from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import List, Literal
from datetime import datetime

from config import DEFAULT_MODEL
ModelName = Literal["siglip", "siglip2", "gte"]
SearchMode = Literal["keyframe", "transcript_semantic", "transcript_exact"]

class Item(BaseModel):
    keyframe_id: str
    video_id: str
    timestamp_ms: int
    frame_idx: int

class TranscriptItem(BaseModel):
    video_id: str
    transcript_id: str
    text: str
    time_start_ms: int
    time_end_ms: int
    keyframes: List[Item]


class QueryRequest(BaseModel):
    query: str = Field(..., description="Truy vấn đoạn video cần tìm")
    limit: int = Field(100, description="Số lượng kết quả trả về")
    mode: SearchMode = Field("keyframe", description="Chế độ search")
    model: ModelName | None = Field(
        DEFAULT_MODEL,
        description="Model dùng để search",
    )

    @model_validator(mode="after")
    def validate_mode_and_model(self) -> "QueryRequest":
        if self.mode == "keyframe" and self.model not in ("siglip", "siglip2"):
            raise ValueError(
                f"When mode is 'keyframe', model must be 'siglip' or 'siglip2', got '{self.model}'"
            )

        if self.mode == "transcript_semantic" and self.model != "gte":
            raise ValueError(
                f"When mode is 'transcript_semantic', model must be 'gte', got '{self.model}'"
            )

        return self

class QueryResponse(BaseModel):
    results: List[Item] | List[TranscriptItem] = Field(default_factory=list)
    total: int = 0

class LogEntry(QueryRequest, QueryResponse):
    request_id: str
    timestamp: datetime | None
