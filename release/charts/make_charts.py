"""Charts for the model card and X (PNG, 1600x900 and 1600x1000).

  python3 release/charts/make_charts.py      -> release/charts/bench.png, size_vs_kl.png

bench.png reads bench.json (swap in the 2x T4 numbers when they are in). size_vs_kl.png is README 6a and the uneven
shapes of README 6h: KL to the original on 60 held-out chat windows, before any training.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OURS, OTHER = "#eb6834", "#b9b7b0"           # highlight: this model; the rest recede
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (validated, light)
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11, "axes.edgecolor": GRID, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def bench() -> None:
    d = json.loads((HERE / "bench.json").read_text())
    panels, models = d["panels"], d["models"]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 5.4), dpi=100, facecolor=SURFACE)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.70, bottom=0.17, wspace=0.2)
    for i, (ax, p) in enumerate(zip(axes, panels)):
        ax.set_facecolor(SURFACE)
        ys = list(range(len(models)))[::-1]
        vals = [v if v is not None else 0 for v in p["values"]]  # None = not measured
        ax.barh(ys, vals, height=0.56, color=[OURS] + [OTHER] * (len(models) - 1), zorder=2)
        for y, v in zip(ys, p["values"]):
            ax.text((v or 0) + p["of"] * 0.02, y, "not measured" if v is None else f"{v}", va="center", ha="left",
                    fontsize=12 if v is not None else 10, color=INK if v is not None else INK2,
                    fontweight="bold" if y == ys[0] else "normal")
        ax.set_xlim(0, p["of"] * 1.18)
        ax.set_yticks(ys, models if i == 0 else [""] * len(models), fontsize=12)
        ax.tick_params(axis="y", length=0)
        ax.set_xticks([0, p["of"]], ["0", f"{p['of']}"], fontsize=10)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_title(p["title"], loc="left", fontsize=14, fontweight="bold", pad=22)
        ax.text(0, 1.04, p["sub"], transform=ax.transAxes, fontsize=9.5, color=INK2)
    fig.text(0.105, 0.90, "A 9.4B cut of Qwen3.5-35B-A3B: close to 9B on competition math, behind on code and knowledge",
             fontsize=18, fontweight="bold")
    fig.text(0.105, 0.84, "bf16, thinking on, one sample per question. Higher is better except the last panel.", fontsize=11, color=INK2)
    fig.text(0.105, 0.04, d["source"], fontsize=9.5, color=INK2)
    fig.savefig(HERE / "bench.png", facecolor=SURFACE)
    print("wrote", HERE / "bench.png")


def size_vs_kl() -> None:
    uniform512 = [(14.5, 0.113, "96×512"), (18.5, 0.059, "128×512"), (22.6, 0.031, "160×512"), (26.6, 0.015, "192×512")]
    uniform256 = [(8.5, 0.268, "96×256"), (10.5, 0.214, "128×256"), (12.5, 0.187, "160×256"), (14.5, 0.171, "192×256"),
                  (18.5, 0.159, "256×256")]
    uneven = [(8.5, 0.182, "8.5B"), (9.4, 0.155, "9.4B (this model)")]
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100, facecolor=SURFACE)
    fig.subplots_adjust(left=0.08, right=0.84, top=0.82, bottom=0.12)
    ax.set_facecolor(SURFACE)
    for ref, lab in ((0.119, "Qwen3.5-9B (trained)"), (0.143, "Qwen3.5-4B (trained)")):
        ax.axhline(ref, color=INK2, lw=1, ls=(0, (4, 4)), zorder=1)
        ax.text(35.8, ref + 0.004, lab, va="bottom", ha="right", fontsize=11, color=INK2)
    for pts, color, name in ((uniform512, BLUE, "full-width experts, fewer of them"),
                             (uniform256, AQUA, "every expert at half width"),
                             (uneven, ORANGE, "uneven widths (each expert its own)")):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=color, lw=2, zorder=3, solid_capstyle="round")
        ax.scatter(xs, ys, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.text(xs[-1] + 0.45, ys[-1], name, color=INK, fontsize=12, va="center", ha="left")
    ax.scatter([35], [0.0016], s=64, color=INK, edgecolor=SURFACE, linewidth=2, zorder=4)
    ax.text(35, 0.014, "original 35B", fontsize=11, ha="center", color=INK)
    ax.annotate("9.4B uneven ≈ 18.5B uniform", xy=(9.4, 0.155), xytext=(10.6, 0.072), fontsize=12, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.set_xlim(7, 36)
    ax.set_ylim(0, 0.34)
    ax.set_xlabel("parameters kept (billions)", fontsize=12)
    ax.set_ylabel("KL to the original (lower = closer)", fontsize=12)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.08, 0.92, "Keep more, stay closer, and the shape of the cut matters as much as its size",
             fontsize=19, fontweight="bold")
    fig.text(0.08, 0.875, "Qwen3.5-35B-A3B cut to different sizes, before any training. KL is a measure of closeness, "
             "not a benchmark score.", fontsize=11.5, color=INK2)
    fig.text(0.08, 0.035, "KL on 60 held-out chat windows, Mac MPS float32. Dashed lines: KL of Qwen3.5-9B / 4B to the "
             "same original.", fontsize=9.5, color=INK2)
    fig.savefig(HERE / "size_vs_kl.png", facecolor=SURFACE)
    print("wrote", HERE / "size_vs_kl.png")


if __name__ == "__main__":
    bench()
    size_vs_kl()
