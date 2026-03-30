#!/usr/bin/env python3
"""Generate benchmark chart images for the tokie README.

Matches the original chart style: light cream background, olive-green tokie bars,
gray competitor bars, horizontal bars with "Nx slower" annotations.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Theme — matches original charts
# ---------------------------------------------------------------------------
BG = "#fdf6e3"        # Light cream/beige
TOKIE_COLOR = "#8b8b00"  # Olive/dark yellow-green
GRAY_COLOR = "#c0c0c0"   # Light gray for HF/tiktoken
GOLD_COLOR = "#b8860b"   # Dark goldenrod for kitoken
TEXT_COLOR = "#333333"
CAPTION_COLOR = "#999999"
DPI = 200
ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS, exist_ok=True)


def _style(ax, title):
    """Apply the original minimal style."""
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT_COLOR)
    ax.spines["bottom"].set_color(TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    # No grid — clean and minimal
    ax.set_title(title, color=TEXT_COLOR, fontsize=16, fontweight="bold", pad=12)


def _save(fig, name):
    for ext in ("png", "svg"):
        path = os.path.join(ASSETS, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  saved {path}")
    plt.close(fig)


def _horiz_speed_chart(title, bars, xlabel, caption, fname, figsize=None):
    """Create a horizontal bar chart in the original style.

    bars: list of (label, value, color, annotation_text)
    """
    n = len(bars)
    if figsize is None:
        figsize = (10, max(2.5, n * 0.85 + 0.8))
    fig, ax = plt.subplots(figsize=figsize)
    _style(ax, title)

    labels = [b[0] for b in bars]
    values = [b[1] for b in bars]
    colors = [b[2] for b in bars]
    annots = [b[3] for b in bars]

    bar_height = 0.55
    y_pos = list(range(n))

    ax.barh(y_pos, values, height=bar_height, color=colors, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.invert_yaxis()

    # Annotations after each bar
    max_val = max(values)
    for i, (val, annot) in enumerate(zip(values, annots)):
        if annot:
            ax.text(
                val + max_val * 0.02, i,
                annot, va="center", ha="left",
                color=TEXT_COLOR, fontsize=11,
            )

    # Caption at bottom right
    if caption:
        ax.text(
            1.0, -0.08, caption,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, fontstyle="italic", color=CAPTION_COLOR,
        )

    _save(fig, fname)


# ---------------------------------------------------------------------------
# 1. Overview benchmark (GPT-2 encoder, like the original)
# ---------------------------------------------------------------------------
def chart_overview():
    # Show tokie vs HF on representative models at 900KB
    # Using Python-level MB/s (900KB text / median_ms)
    # tokie: 9.42ms -> 900/9.42 = 95.5 MB/s; HF: 209.2ms -> 900/209.2 = 4.3 MB/s
    _horiz_speed_chart(
        "Tokenization speed",
        [
            ("tokie",          95.5,  TOKIE_COLOR, ""),
            ("HF tokenizers",  4.3,   GRAY_COLOR,  "22x slower"),
        ],
        "MB/s",
        "GPT-2 encoder, enwik8 900KB, Apple M3 Pro",
        "benchmark",
    )


# ---------------------------------------------------------------------------
# 2. BPE models
# ---------------------------------------------------------------------------
def chart_bpe():
    # GPT-2 at 900KB: tokie=9.42ms (95.5 MB/s), HF=209.2ms (4.3 MB/s)
    _horiz_speed_chart(
        "BPE encoding speed",
        [
            ("tokie",          95.5,  TOKIE_COLOR, ""),
            ("HF tokenizers",  4.3,   GRAY_COLOR,  "22x slower"),
        ],
        "MB/s",
        "GPT-2 encoder, enwik8 900KB, Apple M3 Pro",
        "benchmark_bpe",
    )


# ---------------------------------------------------------------------------
# 3. WordPiece
# ---------------------------------------------------------------------------
def chart_wordpiece():
    # BERT at 900KB: tokie=9.84ms (91.5 MB/s), HF=280.6ms (3.2 MB/s)
    _horiz_speed_chart(
        "WordPiece encoding speed",
        [
            ("tokie",          91.5,  TOKIE_COLOR, ""),
            ("HF tokenizers",  3.2,   GRAY_COLOR,  "29x slower"),
        ],
        "MB/s",
        "BERT-base-uncased, enwik8 900KB, Apple M3 Pro",
        "benchmark_wordpiece",
    )


# ---------------------------------------------------------------------------
# 4. SentencePiece
# ---------------------------------------------------------------------------
def chart_sentencepiece():
    # Gemma 3 at 900KB: tokie=131ms (6.9 MB/s), HF=330ms (2.7 MB/s)
    _horiz_speed_chart(
        "SentencePiece BPE encoding speed",
        [
            ("tokie",          6.9,   TOKIE_COLOR, ""),
            ("HF tokenizers",  2.7,   GRAY_COLOR,  "2.5x slower"),
        ],
        "MB/s",
        "Gemma 3, enwik8 900KB, Apple M3 Pro",
        "benchmark_sentencepiece",
    )


# ---------------------------------------------------------------------------
# 5. tiktoken — vertical bars, dark background (matches original)
# ---------------------------------------------------------------------------
def chart_tiktoken():
    fig, ax = plt.subplots(figsize=(7, 5.5))

    dark_bg = "#1a1a2e"
    ax.set_facecolor(dark_bg)
    fig.set_facecolor(dark_bg)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444466")
    ax.spines["bottom"].set_color("#444466")
    ax.tick_params(colors="#cccccc", labelsize=10)
    ax.xaxis.label.set_color("#cccccc")
    ax.yaxis.label.set_color("#cccccc")
    ax.set_title("OpenAI Tokenizer Speed (900 KB)",
                 color="#e0e0e0", fontsize=14, fontweight="bold", pad=10)

    # cl100k and o200k at 900KB
    models = ["cl100k\n(GPT-4)", "o200k\n(GPT-4o)"]
    tokie_vals = [9.63, 9.83]
    tt_vals = [45.67, 81.47]
    speedups = ["4.7x faster", "8.3x faster"]

    x = [0, 1]
    width = 0.32
    tokie_color = "#2ec4b6"   # Teal
    tt_color = "#b8a9e8"      # Light purple

    bars_tokie = ax.bar([xi - width/2 for xi in x], tokie_vals, width,
                        color=tokie_color, label="tokie", zorder=3)
    bars_tt = ax.bar([xi + width/2 for xi in x], tt_vals, width,
                     color=tt_color, label="tiktoken", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Time (ms) \u2014 lower is better", fontsize=10, color="#cccccc")

    # Speedup annotations on tokie bars
    for i, sp in enumerate(speedups):
        ax.text(
            x[i] - width/2, tokie_vals[i] + 2,
            sp, ha="center", va="bottom",
            color=tokie_color, fontsize=10, fontweight="bold",
        )

    ax.legend(loc="upper left", facecolor="#22223a", edgecolor="#444466",
              labelcolor="#cccccc", fontsize=9)

    _save(fig, "benchmark_tiktoken")


# ---------------------------------------------------------------------------
# 6. Loading times
# ---------------------------------------------------------------------------
def chart_loading():
    # cl100k: tokie=54.6ms, tiktoken=173.2ms, HF=173.2ms
    _horiz_speed_chart(
        "Tokenizer loading time",
        [
            ("tokie (.tkz)",   12.4,   TOKIE_COLOR, ""),
            ("tiktoken",       173.2,  GRAY_COLOR,  "3x slower"),
            ("HF tokenizers",  100.8,  GRAY_COLOR,  "8x slower"),
        ],
        "ms",
        "BERT tokenizer, cold load, Apple M3 Pro",
        "benchmark_loading",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating benchmark charts...")
    chart_overview()
    chart_bpe()
    chart_wordpiece()
    chart_sentencepiece()
    chart_tiktoken()
    chart_loading()
    print("Done.")
