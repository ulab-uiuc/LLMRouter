"""
Routing Memory Bank
===================

Stores (query -> selected_model) routing history on disk and retrieves the most
similar past queries for the next routing decision.

This is intentionally simple:
- Persistence: JSONL at a configured path (append-only)
- Retrieval: dense retrieval using a Contriever encoder + cosine similarity
"""

from __future__ import annotations

import base64
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Handle both package and direct-script imports.
try:
    from .config import MemoryConfig
except ImportError:  # pragma: no cover
    from config import MemoryConfig


DEFAULT_RETRIEVER_MODEL = "facebook/contriever-msmarco"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode_f32_b64(vec: np.ndarray) -> str:
    raw = vec.astype(np.float32).tobytes()
    return base64.b64encode(raw).decode("ascii")


def _decode_f32_b64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64.encode("ascii"))
    return np.frombuffer(raw, dtype=np.float32)


def _normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec) + 1e-12)
    return (vec / denom).astype(np.float32)


class ContrieverEmbedder:
    """
    Dense encoder wrapper around a HuggingFace Contriever checkpoint.

    Lazy-loads the model on first use to avoid downloads/import cost when memory
    is disabled.
    """

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 256):
        self.model_name = model_name or DEFAULT_RETRIEVER_MODEL
        self.device = device or "cpu"
        self.max_length = int(max_length) if max_length else 256

        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)

        self._device = torch.device(self.device)
        self._model.to(self._device)
        self._model.eval()

    def embed(self, texts: List[str]) -> np.ndarray:
        self._ensure_loaded()

        torch = self._torch
        assert torch is not None

        # Tokenize
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model(**inputs)
            last_hidden = out.last_hidden_state
            mask = inputs.get("attention_mask")
            if mask is None:
                pooled = last_hidden[:, 0]
            else:
                mask_f = mask.unsqueeze(-1).type_as(last_hidden)
                summed = (last_hidden * mask_f).sum(dim=1)
                denom = mask_f.sum(dim=1).clamp(min=1e-6)
                pooled = summed / denom

            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        return pooled.detach().cpu().numpy().astype(np.float32)


class MemoryBank:
    """
    Append-only JSONL store + in-memory embedding index.

    File format (one JSON object per line):
    {
      "ts": "...",
      "query": "...",
      "model": "...",
      "strategy": "...",
      "user": "...",
      "emb_b64": "..."  # float32 bytes base64
    }
    """

    def __init__(self, cfg: MemoryConfig, config_dir: Optional[str] = None):
        self.cfg = cfg

        # Resolve path
        path = (cfg.path or "").strip()
        if not path:
            path = str(Path.home() / ".llmrouter" / "openclaw_memory.jsonl")
        else:
            # Allow "~" and environment variables in config.
            path = os.path.expanduser(os.path.expandvars(path))

        p = Path(path)
        if not p.is_absolute() and config_dir:
            p = Path(config_dir) / p
        self.path = p

        self._lock = threading.Lock()
        self._metas: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None  # shape (n, d), normalized

        self._embedder = ContrieverEmbedder(
            model_name=cfg.retriever_model or DEFAULT_RETRIEVER_MODEL,
            device=cfg.device or "cpu",
            max_length=cfg.max_length or 256,
        )

        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return

        metas: List[Dict[str, Any]] = []
        embs: List[np.ndarray] = []
        expected_dim: Optional[int] = None

        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    query = (obj.get("query") or "").strip()
                    model = (obj.get("model") or "").strip()
                    emb_b64 = obj.get("emb_b64")
                    if not query or not model or not emb_b64:
                        continue

                    try:
                        emb = _decode_f32_b64(str(emb_b64))
                        emb = _normalize(emb)
                    except Exception:
                        continue

                    if expected_dim is None:
                        expected_dim = int(emb.shape[0])
                    elif int(emb.shape[0]) != expected_dim:
                        # Skip entries created by a different retriever embedding size.
                        continue

                    meta = {
                        "ts": obj.get("ts"),
                        "query": query,
                        "model": model,
                        "strategy": obj.get("strategy"),
                        "user": obj.get("user"),
                    }
                    metas.append(meta)
                    embs.append(emb)
        except OSError:
            return

        with self._lock:
            self._metas = metas
            if embs:
                self._embeddings = np.stack(embs, axis=0).astype(np.float32)

    def _append_line(self, record: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def add(
        self,
        query: str,
        model: str,
        *,
        strategy: str = "",
        user: Optional[str] = None,
    ) -> None:
        q = (query or "").strip()
        m = (model or "").strip()
        if not q or not m:
            return

        # Keep the stored text bounded.
        max_chars = int(self.cfg.max_query_chars or 500)
        if max_chars > 0:
            q = q[:max_chars]

        emb = self._embedder.embed([q])[0]
        emb = _normalize(emb)

        record = {
            "ts": _utc_now_iso(),
            "query": q,
            "model": m,
            "strategy": (strategy or "").strip(),
            "user": (user or "").strip() or None,
            "emb_b64": _encode_f32_b64(emb),
        }

        with self._lock:
            if self._embeddings is not None and self._embeddings.shape[1] != emb.shape[0]:
                # Different embedding dimension: skip storing to keep index consistent.
                return

            self._metas.append({k: record.get(k) for k in ["ts", "query", "model", "strategy", "user"]})
            if self._embeddings is None:
                self._embeddings = emb.reshape(1, -1).astype(np.float32)
            else:
                self._embeddings = np.concatenate([self._embeddings, emb.reshape(1, -1)], axis=0).astype(np.float32)

        self._append_line(record)

    def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        strategy_filter: Optional[str] = None,
        user: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []

        with self._lock:
            if not self._metas or self._embeddings is None:
                return []
            metas = list(self._metas)
            # Safe to share the numpy array: we only ever replace it (no in-place mutation).
            mat = self._embeddings

        # Optional filters (applied outside the lock for simplicity).
        if strategy_filter:
            sf = strategy_filter.strip()
            keep = [i for i, m in enumerate(metas) if (m.get("strategy") or "") == sf]
            if not keep:
                return []
            metas = [metas[i] for i in keep]
            mat = mat[keep, :]

        if user and self.cfg.per_user:
            u = user.strip()
            keep = [i for i, m in enumerate(metas) if (m.get("user") or "") == u]
            if not keep:
                return []
            metas = [metas[i] for i in keep]
            mat = mat[keep, :]

        k = int(top_k if top_k is not None else (self.cfg.top_k or 10))
        if k <= 0:
            return []

        q_emb = self._embedder.embed([q])[0]
        q_emb = _normalize(q_emb)

        if mat.shape[1] != q_emb.shape[0]:
            return []

        # Cosine similarity via dot product (both normalized).
        scores = mat @ q_emb.reshape(-1, 1)
        scores = scores.reshape(-1)

        k = min(k, scores.shape[0])
        if k <= 0:
            return []

        # Top-k indices, highest score first.
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        results: List[Dict[str, Any]] = []
        for i in idx.tolist():
            meta = metas[i]
            results.append(
                {
                    "query": meta.get("query"),
                    "model": meta.get("model"),
                    "score": float(scores[i]),
                    "ts": meta.get("ts"),
                    "strategy": meta.get("strategy"),
                    "user": meta.get("user"),
                }
            )
        return results
