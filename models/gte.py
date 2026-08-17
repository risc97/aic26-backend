from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
MAX_LENGTH = 8192


class Gte:
    EMBED_DIM = 768

    def __init__(self, device: str = "cuda", amp: bool = True):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model.to(self.device).eval()
        self.amp = amp and self.device.type == "cuda"

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp)

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(
            texts, max_length=MAX_LENGTH, padding=True,
            truncation=True, return_tensors="pt",
        ).to(self.device)
        with self._autocast():
            embeddings = self.model(**tokens).last_hidden_state[:, 0]
        return F.normalize(embeddings.float(), dim=-1).cpu().numpy()
