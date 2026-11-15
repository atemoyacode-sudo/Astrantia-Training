"""Shared pieces for the Qwen3.5-MoE pruning rehearsal: weight sources, layer-by-layer model pieces, data, KL."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import remote_st  # noqa: E402

_TORCH_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64}


class Source:
    """Weights from a Hugging Face repo (read over HTTP, tensor by tensor) or a local checkpoint directory."""

    def __init__(self, spec: str):
        self.spec = spec
        self.path = Path(spec)
        self.local = self.path.exists()
        if self.local:
            index = self.path / "model.safetensors.index.json"
            if index.exists():
                self.weight_map = json.loads(index.read_text())["weight_map"]
            else:
                from safetensors import safe_open

                with safe_open(self.path / "model.safetensors", "pt") as f:
                    self.weight_map = {k: "model.safetensors" for k in f.keys()}
            self.full_config = json.loads((self.path / "config.json").read_text())
        else:
            from huggingface_hub import hf_hub_download

            self.weight_map = remote_st.weight_map(spec)
            self.full_config = json.loads(Path(hf_hub_download(spec, "config.json")).read_text())
        self.prefix = "model.language_model." if any(k.startswith("model.language_model.") for k in self.weight_map) else "model."
        self._handles: dict[str, object] = {}

    @property
    def text_config(self) -> dict:
        return self.full_config.get("text_config", self.full_config)

    def get(self, name: str) -> torch.Tensor:
        """Return a tensor (CPU, checkpoint dtype). `name` is relative to the text model, e.g. 'layers.0.mlp.gate.weight'."""
        full = name if name == "lm_head.weight" else self.prefix + name
        if self.local:
            from safetensors import safe_open

            shard = self.weight_map[full]
            if shard not in self._handles:
                self._handles[shard] = safe_open(self.path / shard, "pt")
            return self._handles[shard].get_tensor(full)
        raw, dtype, shape = remote_st.read_raw(self.spec, full)
        return torch.frombuffer(bytearray(raw), dtype=_TORCH_DTYPES[dtype]).reshape(shape)


def build_config(text_config: dict):
    model_type = text_config.get("model_type", "")
    if "moe" in model_type:
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig as C
    else:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig as C
    keep = {k: v for k, v in text_config.items() if k not in ("architectures", "transformers_version", "model_type")}
    cfg = C(**keep)
    cfg._attn_implementation = "sdpa"
    return cfg


def layer_classes(cfg):
    if "moe" in cfg.model_type:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m

        return m.Qwen3_5MoeDecoderLayer, m.Qwen3_5MoeTextRotaryEmbedding, m.Qwen3_5MoeRMSNorm
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    return m.Qwen3_5DecoderLayer, m.Qwen3_5TextRotaryEmbedding, m.Qwen3_5RMSNorm


def load_layer(src: Source, cfg, idx: int, device: str, overrides: dict[str, torch.Tensor] | None = None):
    """Build decoder layer `idx` on `device` with weights from `src` (or `overrides` for some names)."""
    layer_cls, _, _ = layer_classes(cfg)
    with torch.device("meta"):
        layer = layer_cls(cfg, idx)
    sd = {}
    for k in layer.state_dict().keys():
        t = overrides[k] if overrides and k in overrides else src.get(f"layers.{idx}.{k}")
        sd[k] = t.to(device)
    layer.load_state_dict(sd, assign=True, strict=True)
    return layer.eval()


def positions(batch: int, seq: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Text-only positions: mrope ids (3, B, T) for the rotary embedding and plain ids (B, T)."""
    pos = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
    return pos.unsqueeze(0).expand(3, -1, -1), pos


def run_layer(layer, hidden: torch.Tensor, rotary, microbatch: int) -> torch.Tensor:
    out = torch.empty_like(hidden)
    for s in range(0, hidden.shape[0], microbatch):
        h = hidden[s : s + microbatch]
        mpos, tpos = positions(h.shape[0], h.shape[1], h.device)
        pe = rotary(h, mpos)
        with torch.no_grad():
            out[s : s + microbatch] = layer(h, position_embeddings=pe, attention_mask=None, position_ids=tpos)
    return out


# ---------------------------------------------------------------- data


def build_data(tokenizer_repo: str, seq_len: int, n_c: int, n_a: int, n_b: int, seed: int) -> dict[str, np.ndarray]:
    """One 512-token chunk per Wikipedia article, half Japanese and half English, split C / A / B by article."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_repo)
    need = n_c + n_a + n_b
    chunks: dict[str, list[list[int]]] = {"ja": [], "en": []}
    for lang in chunks:
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        for row in ds:
            ids = tok(row["text"], add_special_tokens=False)["input_ids"]
            if len(ids) >= seq_len:
                chunks[lang].append(ids[:seq_len])
            if len(chunks[lang]) >= (need + 1) // 2:
                break
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for name, n in (("C", n_c), ("A", n_a), ("B", n_b)):
        rows = [chunks["ja"].pop() for _ in range(n // 2)] + [chunks["en"].pop() for _ in range(n - n // 2)]
        arr = np.array(rows, dtype=np.int64)
        out[name] = arr[rng.permutation(len(arr))]
    return out


def build_extra(tokenizer_repo: str, seq_len: int, n: int, seed: int, exclude: list[np.ndarray]) -> np.ndarray:
    """More Wikipedia chunks (half Japanese, half English) for training, skipping any chunk already in `exclude`."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_repo)
    seen = {tuple(r[:64].tolist()) for arr in exclude for r in arr}
    rows: list[list[int]] = []
    for lang, want in (("ja", n // 2), ("en", n - n // 2)):
        got = 0
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
        for row in ds.shuffle(seed=seed, buffer_size=10_000):
            ids = tok(row["text"], add_special_tokens=False)["input_ids"]
            if len(ids) < seq_len or tuple(ids[:64]) in seen:
                continue
            seen.add(tuple(ids[:64]))
            rows.append(ids[:seq_len])
            got += 1
            if got >= want:
                break
    arr = np.array(rows, dtype=np.int64)
    return arr[np.random.default_rng(seed).permutation(len(arr))]


def random_data(vocab: int, seq_len: int, n_c: int, n_a: int, n_b: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {k: rng.integers(0, vocab, size=(n, seq_len), dtype=np.int64) for k, n in (("C", n_c), ("A", n_a), ("B", n_b))}


# ---------------------------------------------------------------- KL against a stored teacher


def teacher_topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    logp = F.log_softmax(logits.float(), dim=-1)
    vals, idx = logp.topk(k, dim=-1)
    return vals, idx


def kl_per_seq(student_logits: torch.Tensor, t_logp: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) per sequence, using the teacher's top-k entries plus one bucket for the rest."""
    logq = F.log_softmax(student_logits.float(), dim=-1)
    q_top = logq.gather(-1, t_idx.long())
    p_top = t_logp.float().exp()
    p_tail = (1 - p_top.sum(-1)).clamp_min(1e-8)
    q_tail = (1 - q_top.exp().sum(-1)).clamp_min(1e-8)
    kl = (p_top * (t_logp.float() - q_top)).sum(-1) + p_tail * (p_tail.log() - q_tail.log())
    return kl.mean(-1)


def ce_per_seq(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    return -logp.gather(-1, ids[:, 1:, None]).squeeze(-1).mean(-1)


# ---------------------------------------------------------------- resuming after a lost VM


class LayerState:
    """A checkpoint of a long pass, so the same command can resume after the VM is lost.

    Put the state (and the outputs) on Google Drive (`colab drivemount`). The state remembers the arguments it was made
    with and is ignored, with a message, if they differ. `save` writes a temporary file and renames it, so a crash
    mid-write leaves the previous checkpoint intact.
    """

    def __init__(self, path: Path, signature: dict):
        self.path = Path(path)
        self.signature = json.loads(json.dumps(signature, default=str))

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        state = torch.load(self.path, map_location="cpu", weights_only=False)
        if state.get("signature") != self.signature:
            print(f"ignoring {self.path}: made with different arguments", flush=True)
            return None
        return state

    def save(self, payload: dict) -> None:
        import os

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        torch.save({**payload, "signature": self.signature}, tmp)
        os.replace(tmp, self.path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


def to_cpu(tensors: dict) -> dict:
    return {k: (to_cpu(v) if isinstance(v, dict) else v.detach().cpu()) for k, v in tensors.items()}


def to_dev(tensors: dict, dev: str) -> dict:
    return {k: (to_dev(v, dev) if isinstance(v, dict) else v.to(dev)) for k, v in tensors.items()}
