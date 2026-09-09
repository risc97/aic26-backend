from __future__ import annotations
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

class Owlv2TextEncoder:
    def __init__(self, repo: str, device: str = "cuda"):
        processor = Owlv2Processor.from_pretrained(repo)
        self.tokenizer = processor.tokenizer
        model = Owlv2ForObjectDetection.from_pretrained(repo)
        self.text_model = model.owlv2.text_model.to(device).eval()
        self.text_projection = model.owlv2.text_projection.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def encode(self, phrases: list[str]) -> torch.Tensor:
        """Phrases -> (D, n) float32 on CPU, L2-normalised, ready to matmul."""
        tokens = self.tokenizer(phrases, padding="max_length", truncation=True,
                                return_tensors="pt").to(self.device)
        pooled = self.text_model(**tokens)[1]
        embeds = self.text_projection(pooled).float()
        embeds = embeds / (embeds.norm(dim=-1, keepdim=True) + 1e-6)
        return embeds.cpu().T.contiguous()