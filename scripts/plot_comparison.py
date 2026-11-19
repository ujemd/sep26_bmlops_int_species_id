"""Accuracy vs latency chart for the extractor comparison, for the README and the slides.

Reads reports/extractor_comparison.csv (written by `make compare`) and writes
reports/extractor_comparison.png. The first row is the baseline.

Usage: uv run python scripts/plot_comparison.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"
LATENCY_TARGET_MS = 300
NOISE_F1 = 0.01  # differences smaller than this are noise on a ~400-photo test set

# Validated categorical palette (fixed order) and neutral tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def main():
    df = pd.read_csv(REPORTS / "extractor_comparison.csv")
    base = df.iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # Context first, so the data sits on top: the noise band around the baseline and the latency target
    ax.axhspan(base.test_macro_f1 - NOISE_F1, base.test_macro_f1 + NOISE_F1, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(150, base.test_macro_f1 - NOISE_F1 + 0.0008, f"±{NOISE_F1} F1 around the baseline: noise",
            color=MUTED, fontsize=8, va="bottom")
    ax.axvline(LATENCY_TARGET_MS, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(LATENCY_TARGET_MS - 6, 0.948, f"latency target\n{LATENCY_TARGET_MS} ms", color=MUTED, fontsize=8,
            va="top", ha="right")

    xmax = max(480, df.latency_ms_p50.max() * 1.4)  # room for the labels on the right
    for i, r in df.iterrows():
        ax.scatter(r.latency_ms_p50, r.test_macro_f1, s=130, color=SERIES[i], edgecolor=SURFACE, linewidth=2,
                   zorder=3, label=f"{r.extractor} ({r.params_millions:.1f} M params)")
        gain = r.test_macro_f1 - base.test_macro_f1
        note = "baseline" if i == 0 else f"{gain:+.3f} F1, ×{r.latency_ms_p50 / base.latency_ms_p50:.1f} latency"
        right_side = r.latency_ms_p50 > 0.8 * xmax  # keep labels inside the plot
        ax.annotate(f"{r.extractor}\n{note}", (r.latency_ms_p50, r.test_macro_f1), xytext=(-10 if right_side else 10, 10),
                    textcoords="offset points", ha="right" if right_side else "left", fontsize=8.5, color=INK_2, zorder=4)

    ax.set_xlim(0, xmax)
    ax.set_ylim(0.89, 0.95)
    ax.set_xlabel("latency of one prediction, median (ms, CPU)", color=INK_2, fontsize=9)
    ax.set_ylabel("macro F1 on the test set", color=INK_2, fontsize=9)
    ax.set_title("Bigger extractors cost more latency for no significant gain", color=INK, fontsize=11,
                 loc="left", pad=12)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    legend = ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=INK_2)
    for handle in legend.legend_handles:
        handle.set_sizes([60])
    fig.tight_layout()
    out = REPORTS / "extractor_comparison.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
