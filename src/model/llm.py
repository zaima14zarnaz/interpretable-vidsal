"""Utilities for loading multimodal LLMs used by explanation_generation.py.

Currently supported short names:
    - qwen3 -> Qwen/Qwen3-VL-8B-Instruct

The returned object contains both the Hugging Face model and processor because
vision-language generation needs both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class LLMHandle:
    """Small container for a loaded multimodal LLM and its processor."""

    model_name: str
    model: torch.nn.Module
    processor: Any


def load_llm(
    model_name: str = "qwen3",
    *,
    device_map: str | dict = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
) -> LLMHandle:
    """Load and return a multimodal LLM handle.

    Args:
        model_name: Short model key. Currently only "qwen3" is supported.
        device_map: Hugging Face device_map argument. Use "auto" for multi-GPU.
        torch_dtype: Optional dtype. Defaults to bfloat16 on CUDA, otherwise float32.
        attn_implementation: Optional attention implementation, e.g. "flash_attention_2".

    Returns:
        LLMHandle containing the loaded model and processor.

    Raises:
        ValueError: If an unsupported model_name is requested.
    """

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "transformers is required for LLM explanation generation. "
            "Install it with: pip install transformers"
        ) from exc

    model_name = model_name.lower().strip()
    if model_name != "qwen3":
        raise ValueError(
            f"Unsupported model_name={model_name!r}. Currently supported: 'qwen3'."
        )

    hf_model_id = "Qwen/Qwen3-VL-8B-Instruct"

    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True)

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForImageTextToText.from_pretrained(hf_model_id, **model_kwargs)
    model.eval()

    return LLMHandle(model_name=hf_model_id, model=model, processor=processor)
