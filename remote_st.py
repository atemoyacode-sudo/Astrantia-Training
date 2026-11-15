"""Read individual tensors (or row ranges) from a remote safetensors checkpoint without downloading the shards.

A safetensors file is an 8-byte header length, a JSON header (name -> dtype, shape, byte offsets), then raw bytes.
We fetch only the header and the byte ranges we need over HTTP, through huggingface_hub's HfFileSystem.
"""

from __future__ import annotations

import json
import struct
from functools import lru_cache

import numpy as np
from huggingface_hub import HfFileSystem, hf_hub_download

_FS = HfFileSystem()
_DTYPES = {"BF16": (2, None), "F16": (2, np.float16), "F32": (4, np.float32), "I64": (8, np.int64)}


@lru_cache(maxsize=None)
def weight_map(repo: str) -> dict[str, str]:
    path = hf_hub_download(repo, "model.safetensors.index.json")
    with open(path) as f:
        return json.load(f)["weight_map"]


@lru_cache(maxsize=None)
def _header(repo: str, shard: str) -> tuple[dict, int]:
    path = f"{repo}/{shard}"
    (n,) = struct.unpack("<Q", _FS.cat_file(path, start=0, end=8))
    header = json.loads(_FS.cat_file(path, start=8, end=8 + n))
    return header, 8 + n


def tensor_info(repo: str, name: str) -> tuple[str, list[int]]:
    shard = weight_map(repo)[name]
    header, _ = _header(repo, shard)
    meta = header[name]
    return meta["dtype"], meta["shape"]


def _decode(raw: bytes, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32)
    return np.frombuffer(raw, dtype=_DTYPES[dtype][1]).astype(np.float32)


def read_raw(repo: str, name: str) -> tuple[bytes, str, list[int]]:
    """Return the tensor's raw bytes, safetensors dtype string and shape."""
    shard = weight_map(repo)[name]
    header, base = _header(repo, shard)
    meta = header[name]
    begin, end = meta["data_offsets"]
    return _FS.cat_file(f"{repo}/{shard}", start=base + begin, end=base + end), meta["dtype"], meta["shape"]


def read(repo: str, name: str, rows: tuple[int, int] | None = None) -> np.ndarray:
    """Return the tensor as float32. `rows=(start, stop)` reads only those rows of the first dimension."""
    shard = weight_map(repo)[name]
    header, base = _header(repo, shard)
    meta = header[name]
    dtype, shape = meta["dtype"], meta["shape"]
    begin, end = meta["data_offsets"]
    itemsize = _DTYPES[dtype][0]
    if rows is not None:
        row_elems = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        start, stop = rows
        begin = begin + start * row_elems * itemsize
        end = begin + (stop - start) * row_elems * itemsize
        shape = [stop - start, *shape[1:]]
    raw = _FS.cat_file(f"{repo}/{shard}", start=base + begin, end=base + end)
    return _decode(raw, dtype).reshape(shape)
