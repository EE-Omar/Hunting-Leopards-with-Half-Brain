"""
Render the v3 comparison figures from comprehensive_comparison_v3.csv.

Three configurations -- baseline, split only, split + learned bottleneck -- all at
the layer-3 split point, all sequential (one frame at a time), so every panel is
a like-for-like comparison.

  A  avg payload size per frame
  B  avg transmission time per frame
  C  throughput
  D  frame time breakdown (Pi Zero compute / network transfer / Pi 3 compute)
  E  accuracy on 1,212 validation images
  F  precision & recall on 1,212 validation images

Each panel is a function, drawn either into the six-panel composite or into its
own standalone figure -- one source of truth, so the two can never disagree.

Panels A/B label the transmitted tensor's dtype under each bar: the split-only arm
was measured sending fp16, the bottleneck uint8. Both encode the SAME fp32
layer-3 activation; those are the configurations actually run, so payload and
timing describe the same measurement.

Usage:
    python scripts/bottleneck/plot_comprehensive.py              # composite + individual
    python scripts/bottleneck/plot_comprehensive.py --composite-only
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Validated categorical palette (see plot_results.py). Colour follows the
# configuration and is held fixed across every panel.
BASE, SPLIT, BNECK = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
SURFACE, RED = "#fcfcfb", "#e34948"

IDS = ["baseline", "split_only", "split_bottleneck"]
NAMES = ["baseline\nno split", "split only\nlayer 3", "split + bottleneck\nlayer 3"]
DTYPES = ["", "fp16 tensor", "uint8 tensor"]
COLOURS = [BASE, SPLIT, BNECK]

PI_CAP = "Layer-3 split, one frame at a time  ·  Raspberry Pi Zero 2W -> Pi 3 over a hotspot LAN"
VAL_CAP = "1,212 validation images  ·  deployed ONNX artifacts, real uint8 round-trip"


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_axisbelow(True)


def panel_title(ax, title, sub=""):
    ax.set_title(title, fontsize=12, color=INK, weight="bold", loc="left", pad=24)
    if sub:
        ax.text(0, 1.030, sub, transform=ax.transAxes, fontsize=8.5, color=INK_2)


def dtype_labels(ax):
    """Tensor dtype under each x tick -- which representation was transmitted."""
    for i, d in enumerate(DTYPES):
        if d:
            ax.text(i, -0.135, d, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color=INK_2, style="italic")


# --------------------------------------------------------------------- panels

def draw_a(ax, col, title, sub):
    style(ax)
    vals = col("payload_kb")
    bars = ax.bar(NAMES, vals, 0.5, color=COLOURS, linewidth=0)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.025,
                "0" if v == 0 else f"{v:,.1f} KB", ha="center", va="bottom",
                fontsize=10, color=INK, weight="bold")
    ax.set_ylim(0, max(vals) * 1.20)
    ax.set_ylabel("KB per frame", fontsize=9.5, color=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    dtype_labels(ax)
    panel_title(ax, title, sub)


def draw_b(ax, col, title, sub):
    style(ax)
    vals = col("wire_ms")
    bars = ax.bar(NAMES, vals, 0.5, color=COLOURS, linewidth=0)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.025, f"{v:.0f} ms",
                ha="center", va="bottom", fontsize=10, color=INK, weight="bold")
    ax.set_ylim(0, max(vals) * 1.20)
    ax.set_ylabel("milliseconds per frame", fontsize=9.5, color=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    dtype_labels(ax)
    panel_title(ax, title, sub)


def draw_c(ax, col, title, sub):
    style(ax)
    vals, deltas = col("throughput_fps"), col("fps_vs_baseline_pct")
    bars = ax.bar(NAMES, vals, 0.5, color=COLOURS, linewidth=0)
    for i, (b, v) in enumerate(zip(bars, vals)):
        txt = f"{v:.3f} fps" if i == 0 else f"{v:.3f} fps\n{deltas[i]:+.1f}%"
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, txt, ha="center",
                va="bottom", fontsize=9.5, color=INK, weight="bold")
    ax.axhline(vals[0], color=BASE, lw=1.0, ls=(0, (4, 3)), zorder=0)
    ax.set_ylim(0, max(vals) * 1.40)
    ax.set_ylabel("frames per second", fontsize=9.5, color=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    panel_title(ax, title, sub)


def draw_d(ax, col, title, sub):
    style(ax)
    head, wire, tail = col("head_ms"), col("wire_ms"), col("tail_ms")
    y, h = np.arange(3), 0.46
    ax.barh(y, head, h, color=BASE, linewidth=0, label="Pi Zero compute")
    ax.barh(y, wire, h, left=np.array(head) + 8, color=RED, linewidth=0,
            label="network transfer")
    ax.barh(y, tail, h, left=np.array(head) + np.array(wire) + 16, color=BNECK,
            linewidth=0, label="Pi 3 compute")
    for yi, (a, b_, c) in enumerate(zip(head, wire, tail)):
        ax.text(a + b_ + c + 70, yi, f"{a + b_ + c:,.0f} ms", va="center",
                fontsize=9.5, color=INK, weight="bold")
    ax.set_yticks(y, ["baseline\n(all on Pi Zero)", "split only", "split + bottleneck"],
                  color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 3550)
    ax.set_xlabel("milliseconds per frame", fontsize=9.5, color=INK_2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    panel_title(ax, title, sub)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncols=3, loc="upper left",
              bbox_to_anchor=(-0.012, -0.17), handlelength=1.1, columnspacing=1.6,
              borderpad=0)


def _paired_bars(ax, col, keys, labels, fmt, title, sub, deltas=False):
    """Two grouped series per configuration (used by E and F)."""
    style(ax)
    v1, v2 = col(keys[0]), col(keys[1])
    x, w = np.arange(3), 0.34
    b1 = ax.bar(x - w / 2 - 0.012, v1, w, color=COLOURS, linewidth=0, label=labels[0])
    b2 = ax.bar(x + w / 2 + 0.012, v2, w, color=COLOURS, alpha=0.45, linewidth=0,
                label=labels[1])
    for bars, vs in ((b1, v1), (b2, v2)):
        for b, v in zip(bars, vs):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.014, fmt.format(v),
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    if deltas:
        for i, d in enumerate(col("mAP50_95_delta")):
            ax.text(i, -0.135, "reference" if i == 0 else f"{d:+.4f} mAP50-95",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=8.5, color=INK_2 if d == 0 else RED)
    ax.set_xticks(x, NAMES, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score", fontsize=9.5, color=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    panel_title(ax, title, sub)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncols=2, loc="upper left",
              bbox_to_anchor=(-0.012, -0.245 if deltas else -0.175),
              handlelength=1.1, borderpad=0)


def draw_e(ax, col, title, sub):
    _paired_bars(ax, col, ("mAP50_95", "mAP50"), ("mAP50-95", "mAP50"),
                 "{:.4f}", title, sub, deltas=True)


def draw_f(ax, col, title, sub):
    # Same 1,212-image run as panel E (not a second sample) -- precision and
    # recall say WHICH kind of error the bottleneck introduces.
    _paired_bars(ax, col, ("precision", "recall"), ("precision", "recall"),
                 "{:.3f}", title, sub)


# axes rect for the standalone figures; D is a horizontal bar chart whose y tick
# labels ("split + bottleneck") need a wider left margin than the vertical panels.
RECT_V = [0.135, 0.215, 0.835, 0.635]
RECT_H = [0.225, 0.235, 0.745, 0.615]

PANELS = [
    ("a", "payload_size",         draw_a, "Avg payload size per frame",        PI_CAP,  RECT_V),
    ("b", "transmission_time",    draw_b, "Avg transmission time per frame",   PI_CAP,  RECT_V),
    ("c", "throughput",           draw_c, "Throughput",                        PI_CAP,  RECT_V),
    ("d", "frame_time_breakdown", draw_d, "Frame time breakdown",              PI_CAP,  RECT_H),
    ("e", "accuracy",             draw_e, "Accuracy",                          VAL_CAP, RECT_V),
    ("f", "precision_recall",     draw_f, "Precision & recall",                VAL_CAP, RECT_V),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(ROOT / "results/benchmarks/comprehensive_comparison_v3.csv"))
    ap.add_argument("--out", default=str(ROOT / "results/charts/comprehensive_comparison_v3.png"))
    ap.add_argument("--indiv-dir", default=str(ROOT / "results/charts/v3_panels"))
    ap.add_argument("--composite-only", action="store_true")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    by = {r["config_id"]: r for r in csv.DictReader(open(args.csv, newline=""))}
    col = lambda key: [fnum(by[i][key]) for i in IDS]

    # ------------------------------------------------------------- composite
    fig = plt.figure(figsize=(15.5, 13.2), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, hspace=0.80, wspace=0.24,
                          left=0.075, right=0.975, top=0.860, bottom=0.070)
    fig.text(0.075, 0.962, "YOLO26-Large at 320x320: baseline vs split vs learned bottleneck",
             fontsize=18, color=INK, weight="bold")
    fig.text(0.075, 0.936,
             "Layer-3 split, one frame at a time (sequential)  ·  "
             "accuracy on 1,212 validation images  ·  "
             "latency on Raspberry Pi Zero 2W -> Pi 3 over a hotspot LAN",
             fontsize=10.5, color=INK_2)

    for n, (key, _slug, fn, title, _cap, _rect) in enumerate(PANELS):
        ax = fig.add_subplot(gs[n // 2, n % 2])
        # In the composite the header carries the context, so panels get no caption.
        fn(ax, col, f"{key.upper()} · {title}", "")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"[+] {out}")

    if args.composite_only:
        return

    # ------------------------------------------------------ individual panels
    # Each standalone figure carries its own caption so it is self-contained in
    # a report, where it will sit under the report's own figure numbering --
    # hence no "A ·" prefix here.
    outdir = Path(args.indiv_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for key, slug, fn, title, cap, rect in PANELS:
        fig = plt.figure(figsize=(7.6, 5.6), facecolor=SURFACE)
        ax = fig.add_axes(rect)
        fn(ax, col, title, cap)
        for ext, kw in ((".svg", {}), (".png", {"dpi": args.dpi})):
            path = outdir / f"{key}_{slug}{ext}"
            fig.savefig(path, facecolor=SURFACE, **kw)
        plt.close(fig)
        print(f"[+] {outdir.name}/{key}_{slug}.svg  +  .png")


if __name__ == "__main__":
    main()
