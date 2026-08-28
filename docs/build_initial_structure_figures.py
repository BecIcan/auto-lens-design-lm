"""Build the public benchmark figure from disclosure-safe aggregate data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "initial_structure_benchmark.json").read_text(encoding="utf-8"))
cases = DATA["cases"]
labels = [row["label"].replace(" · ", "\n") for row in cases]
x = np.arange(len(cases))
width = 0.24

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(12.5, 5.4))
fig.patch.set_facecolor("#07111f")
ax.set_facecolor("#07111f")
for offset, key, label, color in (
    (-width, "source_um", "Retrieved source", "#7895b2"),
    (0.0, "generated_um", "One-shot generated", "#39d98a"),
    (width, "reference_um", "Private reference", "#f6c85f"),
):
    bars = ax.bar(x + offset, [row[key] for row in cases], width, label=label, color=color, alpha=0.94)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#dce8f5",
        )
ax.axvline(2.5, color="#5e7185", linewidth=1)
ax.text(1, 74, "VISIBLE", ha="center", color="#8ba3bb", weight="bold")
ax.text(4, 74, "VISIBLE + NIR", ha="center", color="#8ba3bb", weight="bold")
ax.set_ylim(0, 80)
ax.set_ylabel("Mean RMS spot radius [µm]")
ax.set_xticks(x, labels)
ax.grid(axis="y", color="#294057", alpha=0.45, linewidth=0.7)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.legend(frameon=False, loc="upper left", ncols=3)
ax.set_title(
    "Private Seed Generator · Public Real-Ray Audit",
    loc="left",
    fontsize=18,
    weight="bold",
    pad=22,
)
ax.text(
    0,
    1.02,
    "One forward pass per candidate · no post-generation optimizer or paraxial solve",
    transform=ax.transAxes,
    color="#9fb3c8",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(
    ROOT / "assets" / "initial_structure_benchmark.png",
    dpi=180,
    facecolor=fig.get_facecolor(),
)
