"""Generation helpers for downstream RAG experiments."""

from __future__ import annotations

import os
from typing import List, Optional

import torch

_TOK = None
_GEN = None
_DEV = None
_MODEL_TYPE = None


def _detect_model_type(model_name: str) -> str:
    name = model_name.lower()
    if "medgemma" in name:
        return "medgemma"
    if "qwen3" in name:
        return "qwen3"
    if "qwen2" in name or "qwen-" in name:
        return "qwen2"
    if "llama" in name:
        return "llama"
    return "other"


def _require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Generation helpers require transformers. Install them with "
            "`pip install -e .[generation]`."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def load_generator(model_name: str, local_path: Optional[str] = None) -> None:
    """Load a generator model.

    Args:
        model_name: Hugging Face model id, or a local path if local_path is not
            set.
        local_path: Optional explicit local path. The environment variable
            FEDRAG_GENERATOR_PATH is also honored.
    """
    global _TOK, _GEN, _DEV, _MODEL_TYPE
    load_path = local_path or os.environ.get("FEDRAG_GENERATOR_PATH") or model_name
    _MODEL_TYPE = _detect_model_type(model_name)
    _DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading generator from: {load_path}")
    print(f"Detected model type: {_MODEL_TYPE}")

    if _MODEL_TYPE == "medgemma":
        from transformers import AutoModelForImageTextToText, AutoProcessor

        _TOK = AutoProcessor.from_pretrained(load_path)
        if torch.cuda.is_available():
            _GEN = AutoModelForImageTextToText.from_pretrained(
                load_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            _GEN = AutoModelForImageTextToText.from_pretrained(load_path)
            _GEN.to(_DEV)
    else:
        AutoModelForCausalLM, AutoTokenizer = _require_transformers()
        _TOK = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
        if getattr(_TOK, "pad_token", None) is None:
            _TOK.pad_token = _TOK.eos_token
        if torch.cuda.is_available():
            _GEN = AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            _GEN = AutoModelForCausalLM.from_pretrained(load_path, trust_remote_code=True)
            _GEN.to(_DEV)
    _GEN.eval()


def _build_prompt(query: str, docs: List[str], model_type: str) -> str:
    context = "\n\n".join([f"[{i + 1}] {doc}" for i, doc in enumerate(docs)])
    if model_type in {"qwen3", "qwen2"}:
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant. Answer the question based on the provided context.<|im_end|>\n"
            "<|im_start|>user\n"
            f"Context:\n{context}\n\nQuestion: {query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if model_type == "llama":
        return (
            "[INST] Use the provided context to answer the question.\n\n"
            f"Context:\n{context}\n\nQuestion: {query} [/INST]"
        )
    return (
        "Use the provided context to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )


def generate_answer(
    query: str,
    docs: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
    do_sample: bool = True,
) -> str:
    """Generate a RAG answer from a query and retrieved documents."""
    if _GEN is None or _TOK is None:
        raise RuntimeError("Generator not loaded. Call load_generator(...) first.")
    if _MODEL_TYPE == "medgemma":
        return _generate_medgemma(query, docs, max_new_tokens, temperature, top_p, top_k, repetition_penalty, do_sample)
    if _MODEL_TYPE == "qwen3":
        return _generate_qwen3(query, docs, max_new_tokens, temperature, top_p, top_k, repetition_penalty, do_sample)

    prompt = _build_prompt(query, docs, _MODEL_TYPE or "other")
    inputs = _TOK(prompt, return_tensors="pt").to(_GEN.device if hasattr(_GEN, "device") else _DEV)
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "top_k": top_k if do_sample else None,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": _TOK.pad_token_id,
        "eos_token_id": _TOK.eos_token_id,
    }
    generation_args = {k: v for k, v in generation_args.items() if v is not None}
    with torch.no_grad():
        outputs = _GEN.generate(**inputs, **generation_args)
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return _TOK.decode(generated, skip_special_tokens=True).strip()


def _generate_qwen3(
    query: str,
    docs: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    do_sample: bool,
) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer based on the provided context."},
        {"role": "user", "content": f"Context:\n{chr(10).join(docs)}\n\nQuestion: {query}"},
    ]
    prompt = _TOK.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _TOK(prompt, return_tensors="pt").to(_GEN.device if hasattr(_GEN, "device") else _DEV)
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "top_k": top_k if do_sample else None,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": _TOK.pad_token_id,
        "eos_token_id": _TOK.eos_token_id,
    }
    generation_args = {k: v for k, v in generation_args.items() if v is not None}
    with torch.no_grad():
        outputs = _GEN.generate(**inputs, **generation_args)
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return _TOK.decode(generated, skip_special_tokens=True).strip()


def _generate_medgemma(
    query: str,
    docs: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    do_sample: bool,
) -> str:
    context = "\n\n".join([f"[{i + 1}] {doc}" for i, doc in enumerate(docs)])
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Context:\n{context}\n\nQuestion: {query}"}],
        }
    ]
    prompt = _TOK.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    prompt = prompt.to(_GEN.device if hasattr(_GEN, "device") else _DEV)
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "top_k": top_k if do_sample else None,
        "repetition_penalty": repetition_penalty,
    }
    generation_args = {k: v for k, v in generation_args.items() if v is not None}
    with torch.no_grad():
        outputs = _GEN.generate(prompt, **generation_args)
    generated = outputs[0][prompt.shape[1] :]
    return _TOK.decode(generated, skip_special_tokens=True).strip()


def is_generator_loaded() -> bool:
    return _GEN is not None and _TOK is not None


def build_prompt_for_logging(query: str, docs: List[str]) -> str:
    return _build_prompt(query, docs, _MODEL_TYPE or "other")
