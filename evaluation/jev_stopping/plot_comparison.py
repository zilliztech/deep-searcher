# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Plot verified adaptive stopping results from the paired replay report."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent / "results"
report = json.loads((ROOT / "report.json").read_text())
cost = json.loads((ROOT / "decision_cost_estimate.json").read_text())
methods = ["llm", "jev"]
colors = ["#6864B4", "#008C7C"]
labels = ["DeepSeek v4 Flash", "Jev"]
bg, ink, muted = "#FBFCFE", "#192B40", "#68798C"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "svg.fonttype": "none",
        "svg.hashsalt": "deepsearcher-stopping",
    }
)
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), facecolor=bg)
fig.subplots_adjust(left=0.085, right=0.955, top=0.74, bottom=0.10, hspace=0.62, wspace=0.32)
fig.text(
    0.065, 0.934, "DeepSearcher · When to Stop Searching", fontsize=26, weight="bold", color=ink
)
fig.text(
    0.065,
    0.885,
    "100 queries · Up to 7 rounds · Paired trajectory replay",
    fontsize=12,
    color=muted,
)
fig.legend(
    handles=[Patch(facecolor=color, label=label) for color, label in zip(colors, labels)],
    loc="upper left",
    bbox_to_anchor=(0.059, 0.851),
    ncol=2,
    frameon=False,
    fontsize=12,
    columnspacing=2.5,
)


def style(ax, title, unit):
    ax.set_facecolor(bg)
    ax.set_title(title, loc="left", fontsize=17, weight="bold", color=ink, pad=30)
    ax.text(0, 1.055, unit, transform=ax.transAxes, color=muted, fontsize=10)
    ax.yaxis.grid(True, color="#E4E9EF", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=8, labelcolor=muted)


ax = axes[0, 0]
style(ax, "Evidence recall", "% · Higher is better")
for j, (method, color) in enumerate(zip(methods, colors)):
    values = [report["policies"][method][key] * 100 for key in ["r2", "r5"]]
    xs = [i + (j - 0.5) * 0.28 for i in range(2)]
    ax.bar(xs, values, width=0.25, color=color, alpha=0.92)
    for x, value in zip(xs, values):
        ax.text(x, value + 2, f"{value:.2f}", ha="center", color=color, weight="bold", fontsize=12)
ax.set_xticks([0, 1], ["Recall@2", "Recall@5"])
ax.set_ylim(0, 110)
ax.set_yticks([0, 25, 50, 75, 100])

specs = [
    (
        axes[0, 1],
        "Search rounds",
        "Mean per query",
        [report["policies"][m]["round"] for m in methods],
        4,
        lambda v: f"{v:.2f}",
    ),
    (
        axes[1, 0],
        "Stop-decision latency",
        "Median API response · Seconds",
        [report["latency"][m]["successful_request_median_seconds"] for m in methods],
        3,
        lambda v: f"{v:.2f} s",
    ),
    (
        axes[1, 1],
        "Estimated decision cost",
        "USD / 100 queries · Public-rate estimate",
        [cost["llm"]["off_peak_usd"], cost["jev"]["usd"]],
        0.06,
        lambda v: f"${v:.4f}",
    ),
]
for ax, title, unit, values, upper, fmt in specs:
    style(ax, title, unit)
    ax.bar([0, 1], values, width=0.46, color=colors, alpha=0.92)
    for i, (value, color) in enumerate(zip(values, colors)):
        ax.text(
            i,
            value + upper * 0.035,
            fmt(value),
            ha="center",
            color=color,
            weight="bold",
            fontsize=15,
        )
    ax.set_xticks([0, 1], ["DeepSeek", "Jev"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, upper)
axes[0, 1].set_yticks([0, 1, 2, 3, 4])
axes[1, 0].set_yticks([0, 1, 2, 3])
ax = axes[1, 1]
ax.set_yticks([0, 0.02, 0.04, 0.06], ["$0", ".02", ".04", ".06"])
ax.text(
    0.5,
    -0.24,
    "DeepSeek: off-peak rate, no cache discount",
    transform=ax.transAxes,
    ha="center",
    color=muted,
    fontsize=9,
)
for ext in ["png", "svg", "pdf"]:
    fig.savefig(
        ROOT / f"deepsearcher-stopping-comparison.{ext}",
        dpi=210,
        facecolor=bg,
        metadata={"Date": None} if ext == "svg" else None,
    )
svg = ROOT / "deepsearcher-stopping-comparison.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
print(ROOT / "deepsearcher-stopping-comparison.png")
