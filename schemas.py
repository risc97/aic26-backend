from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import List, Literal
from datetime import datetime

from config import DEFAULT_MODEL
KeyframeModel = Literal["siglip", "siglip2", "pe"]
TranscriptModel = Literal["gte"]
ModelName   = Literal["siglip", "siglip2", "gte", "pe", "owlv2-base", "owlv2-large"]
SearchMode  = Literal["keyframe", "transcript_semantic", "transcript_exact",
                      "ocr_exact", "temporal", "detect", "temporal_detect"]
DetectModel = Literal["owlv2-base", "owlv2-large"]
Region = Literal["left", "right", "center", "top", "bottom"]

class Item(BaseModel):
    keyframe_id: str
    video_id: str
    timestamp_ms: int
    frame_idx: int
    video_fps: float
    score: float | None = None

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


class KeyframeListResponse(BaseModel):
    video_id: str
    total: int = 0
    keyframes: List[Item] = Field(default_factory=list)

class QueryRequest(BaseModel):
    query: str = Field(..., description="Truy vấn đoạn video cần tìm")
    limit: int = Field(100, description="Số lượng kết quả trả về")

class KeyframeQueryRequest(QueryRequest):
    model: KeyframeModel = Field("siglip2", description="Model dùng để search")

class TranscriptSemanticQueryRequest(QueryRequest):
    model: TranscriptModel = Field("gte", description="Model dùng để search")

class TranscriptExactQueryRequest(QueryRequest):
    phrase: bool = Field(
            False,
            description="True = các từ phải liền nhau đúng thứ tự ('sạt lở'), "
                        "False = câu thoại chứa đủ các từ ở bất kỳ đâu",
        )

class OcrQueryRequest(QueryRequest):
    phrase: bool = Field(
            False,
            description="Chỉ dùng cho mode exact: True = các từ phải liền nhau đúng "
                        "thứ tự ('sạt lở'), False = keyframe chứa đủ các từ ở bất kỳ đâu",
        )

class QueryResponse(BaseModel):
    request_id: str
    mode: SearchMode
    model: ModelName | None = None
    total: int = 0

class KeyframeQueryResponse(QueryResponse):
    mode: Literal["keyframe"] = "keyframe"
    model: KeyframeModel
    results: List[Item] = Field(default_factory=list)

class TranscriptSemanticQueryResponse(QueryResponse):
    mode: Literal["transcript_semantic"] = "transcript_semantic"
    model: Literal["gte"] = "gte"
    results: List[TranscriptItem] = Field(default_factory=list)

class TranscriptExactQueryResponse(QueryResponse):
    mode: Literal["transcript_exact"] = "transcript_exact"
    results: List[TranscriptItem] = Field(default_factory=list)

class OcrQueryResponse(QueryResponse):
    mode: Literal["ocr_exact"] = "ocr_exact"
    results: List[OcrItem] = Field(default_factory=list)

class SimilarResponse(BaseModel):
    video_id: str
    keyframe_id: str
    model: KeyframeModel
    total: int = 0
    results: List[Item] = Field(default_factory = list)

class LogEntry(BaseModel):
    request_id: str
    timestamp: datetime | None
    query: str
    limit: int
    mode: SearchMode
    model: ModelName | None = None
    results: List[OcrItem] | List[TemporalItem] | List[TranscriptItem] | List[Item] = Field(default_factory=list)
    total: int = 0

class TemporalStage(BaseModel):
    query: str = Field(..., description="Sự kiện cần tìm ở bước này")
    variants: List[str] = Field(
        default_factory=list,
        description="Cách diễn đạt khác của cùng sự kiện, gộp bằng max",
    )

class TemporalQueryRequest(BaseModel):
    stages: List[TemporalStage] = Field(..., min_length=2, description="Chuỗi sự kiện, đúng thứ tự")
    seed: int = Field(0, ge=0, description="Stage để search toàn corpus")
    limit: int = Field(100, description="Số video trả về")
    model: ModelName = Field(DEFAULT_MODEL, description="Model dùng để search")
    chains_per_video: int = Field(1, ge=1, description="Số chuỗi tối đa mỗi video đóng góp")
    r: int = Field(2000, ge=1, description="Độ sâu recall của q1 (S_R)")
    rrf_k: float = 60.0
    weights: List[float] | None = Field(
        None, description="w_i cho từng query, đúng len(stages); null = tất cả bằng 1",
    )
    max_gap_ms: int | None = Field(
        120_000, ge=0,
        description="Khoảng cách tối đa giữa hai bước liên tiếp; null = không giới hạn",
    )
    iou_threshold: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Hai chuỗi chồng nhau quá mức này bị coi là trùng lặp",
    )

    @model_validator(mode="after")
    def _check_weights(self):
        if self.weights is not None and len(self.weights) != len(self.stages):
            raise ValueError(
                f"weights phải có đúng một giá trị cho mỗi stage: cần "
                f"{len(self.stages)}, nhận {len(self.weights)}"
            )
        return self
    
    @model_validator(mode="after")
    def _check_seed(self):
        if self.seed >= len(self.stages):
            raise ValueError(f"seed phải nằm trong 0..{len(self.stages) - 1}")
        return self


class TemporalMatch(Item):
    stage: int
    query: str
    score: float
    rank: int # vị trí của kf này trong result của query
    timestamp_ms: int

class TemporalItem(BaseModel):
    rank: int
    video_id: str
    score: float # RRF score
    length: int # số query chuỗi khớp
    matches: List[TemporalMatch] = Field(default_factory=list)
    skipped_stages: List[int] = Field(default_factory=list)

class TemporalQueryResponse(BaseModel):
    mode: Literal["temporal"] = "temporal"
    results: List[TemporalItem] = Field(default_factory=list)

class DetectObject(BaseModel):
    phrase: str = Field(..., description="Tên object")
    min_count: int = Field(1, ge=1, description="Số lượng tối thiểu trong khung hình")
    min_score: float = Field(0.25, ge=0.0, le=1.0)
    region: Region | None = Field(None, description="Vị trí trong khung hình")
    min_area: float = Field(0.0, ge=0.0, le=1.0, description="Tỉ lệ diện tích tối thiểu")
    max_area: float = Field(1.0, ge=0.0, le=1.0)

class DetectPrefilter(BaseModel):
    query: str = Field(..., description="Câu truy vấn ngữ nghĩa chạy trước")
    model: KeyframeModel = "siglip2"
    limit: int = Field(2000, ge=1, description="Số keyframe đưa vào bước detect")

class DetectItem(Item):
    counts: List[int] = Field(default_factory=list, description="Theo thứ tự objects")
    boxes: List[List[List[float]]] = Field(
        default_factory=list, description="[object][box] = [x0, y0, x1, y1, score]")

class DetectQueryRequest(BaseModel):
    objects: List[DetectObject] = Field(..., min_length=1)
    limit: int = Field(100, ge=1)
    model: DetectModel = "owlv2-base"
    nms_iou: float = Field(0.5, ge=0.0, le=1.0)
    prefilter: DetectPrefilter | None = Field(
        None, description="Bỏ trống = quét toàn bộ corpus (chậm hơn nhiều)")

class DetectQueryResponse(QueryResponse):
    mode: Literal["detect"] = "detect"
    model: DetectModel
    results: List[DetectItem] = Field(default_factory=list)


# ---

class DetectStageRequest(BaseModel):
    objects: List[DetectObject] = Field(..., min_length=1,
                                        description="Tất cả object phải xuất hiện cùng khung hình")

class TemporalDetectQueryRequest(BaseModel):
    stages: List[DetectStageRequest] = Field(..., min_length=2,
                                             description="Chuỗi sự kiện, đúng thứ tự")
    seed: int = Field(0, ge=0, description="Stage search toàn corpus")
    limit: int = Field(100, ge=1, description="Số video trả về")
    model: DetectModel = "owlv2-base"
    nms_iou: float = Field(0.5, ge=0.0, le=1.0)
    r: int = Field(5000, ge=1, description="Độ sâu recall của stage 1 (S_R)")
    rrf_k: float = 60.0
    weights: List[float] | None = None
    max_gap_ms: int | None = Field(120_000, ge=0)
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0)
    chains_per_video: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _check_weights(self):
        if self.weights is not None and len(self.weights) != len(self.stages):
            raise ValueError(f"weights cần đúng {len(self.stages)} giá trị")
        return self

    @model_validator(mode="after")
    def _check_seed(self):
        if self.seed >= len(self.stages):
            raise ValueError(f"seed phải nằm trong 0..{len(self.stages) - 1}")
        return self


class TemporalDetectQueryResponse(QueryResponse):
    mode: Literal["temporal_detect"] = "temporal_detect"
    model: DetectModel
    results: List[TemporalItem] = Field(default_factory=list)
