"""mlx-lm's server with memory limits, so that one model cannot take the whole Mac down.

It did on 2026-11-12: a 4-bit model of 8.8 GB grew to 16.5 GB with its KV caches and prompt cache, mlx-lm wires
everything up to the recommended working set, wired memory reached 30 of 32 GB, and the kernel's watchdog rebooted the
machine. Here the wired limit is small (memory above it can be compressed or swapped instead), freed buffers go back
to the system, and the prompt cache (useless for one-off questions) is kept small by the caller's flags.

  MLX_WIRED_GB=11 MLX_CACHE_GB=1 MLX_MEM_GB=14 python mlx_safe_server.py --model <mlx model> [--adapter-path d] --port 8080
"""

from __future__ import annotations

import os

import mlx.core as mx

GB = 1 << 30


def main() -> None:
    import mlx_lm.server as server

    wired, cache, mem = (float(os.environ.get(k, d)) for k, d in (("MLX_WIRED_GB", 11), ("MLX_CACHE_GB", 1), ("MLX_MEM_GB", 14)))
    server.maybe_set_recommended_wired_limit = lambda: mx.set_wired_limit(int(wired * GB))
    mx.set_cache_limit(int(cache * GB))
    mx.set_memory_limit(int(mem * GB))
    print(f"memory limits: wired {wired:g} GB, buffer cache {cache:g} GB, memory {mem:g} GB", flush=True)
    server.main()


if __name__ == "__main__":
    main()
