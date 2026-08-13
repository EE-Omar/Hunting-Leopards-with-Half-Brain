"""
Render the v3 three-way comparison figure from the benchmark CSV.

Reads results/benchmarks/accuracy_three_way_v3.csv (the single source of truth for
accuracy and wire size) and measures the compute split directly from the ONNX graphs,
so the figure cannot drift from the artifacts it describes.

Usage:
    python scripts/bottleneck/plot_results.py
    python scripts/bottleneck/plot_results.py --csv ... --out ...
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Categorical slots 1-3 of the validated reference palette. Validated all-pairs in
# light mode: worst CVD dE 9.2, worst normal-vision dE 24.0. Colour follows the
# CONFIG (entity), never its rank, and is held fixed across panels.
FULL, SPLIT, BNECK = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
SURFACE = "#fcfcfb"

# Measured on 150 val images resized to 320x320 (cv2.imencode, quality 75). The
# 40 KB "JPEG baseline" hardcoded in verify_bottleneck.py is an assumption, not this.
JPEG_Q75_KB = 23.2


def conv_macs(path):
    """Multiply-accumulates in an ONNX graph, from Conv/MatMul output shapes."""
    m = shape_inference.infer_shapes(onnx.load(str(path)))
    shapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
              for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output)}
    init = {i.name: numpy_helper.to_array(i).shape for i in m.graph.initializer}
    total = 0
    for n in m.graph.node:
        if n.op_type == "Conv":
            w, out = init.get(n.input[1]), shapes.get(n.output[0])
            if w and out and len(out) == 4:
                total += out[1] * out[2] * out[3] * (w[1] * w[2] * w[3])
        elif n.op_type == "MatMul":
            a, b = shapes.get(n.input[0]), shapes.get(n.input[1])
            if a and b and len(a) >= 2 and len(b) >= 2:
                total += a[-2] * a[-1] * b[-1]
    return total


def style(ax):
    """Hairline recessive chrome — no top/right spines, solid hairline y-grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description="Plot the v3 three-way comparison figure")
    ap.add_argument("--csv", default=str(ROOT / "results/benchmarks/accuracy_three_way_v3.csv"))
    ap.add_argument("--split-dir", default=str(ROOT / "models/split"))
    ap.add_argument("--full-dir", default=str(ROOT / "models/full"))
    ap.add_argument("--out", default=str(ROOT / "results/charts/bottleneck_comparison_v3.png"))
    ap.add_argument("--imgsz", type=int, default=320)
    args = ap.parse_args()

    rows = {r["config"]: r for r in csv.DictReader(open(args.csv, newline=""))}
    f = lambda cfg, k: float(rows[cfg][k])

    fig = plt.figure(figsize=(13, 9.8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.24,
                          left=0.075, right=0.975, top=0.845, bottom=0.085)

    fig.text(0.075, 0.958, "YOLO26-Large split inference + learned bottleneck",
             fontsize=17, color=INK, weight="bold")
    fig.text(0.075, 0.928,
             f"Layer-3 split at {args.imgsz}px, 256→16 channel bottleneck, uint8 wire  ·  "
             "1,212-image validation split, deployed ONNX artifacts",
             fontsize=10.5, color=INK_2)

    def legend_below(ax, ncols):
        """Legends sit under the axes so they never cover a mark."""
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, ncols=ncols,
                  loc="upper left", bbox_to_anchor=(-0.012, -0.155), handlelength=1.1,
                  handleheight=0.9, columnspacing=1.4, borderpad=0)

    # --- A: accuracy by config -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    style(ax)
    metrics = [("mAP50-95", "mAP50-95"), ("mAP50", "mAP50")]
    cfgs = [("full", "full ONNX", FULL), ("split", "split, no bottleneck", SPLIT),
            ("bottleneck", "split + bottleneck", BNECK)]
    w, x = 0.20, np.arange(len(metrics))
    for i, (key, label, colour) in enumerate(cfgs):
        vals = [f(key, col) for _, col in metrics]
        off = (i - 1) * (w + 0.018)      # surface gap between adjacent bars
        bars = ax.bar(x + off, vals, w, label=label, color=colour, linewidth=0)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [m for m, _ in metrics], color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score", fontsize=9, color=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_title("Accuracy — the split is exactly lossless", fontsize=11.5,
                 color=INK, weight="bold", loc="left", pad=24)
    ax.text(0, 1.028, "full and split bars are identical to 6 decimals; "
            "the bottleneck costs −0.020 mAP50-95",
            transform=ax.transAxes, fontsize=8.5, color=INK_2)
    legend_below(ax, 3)

    # --- B: per-class AP50-95 --------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    style(ax)
    classes = [k.replace("AP50-95_", "") for k in rows["full"] if k.startswith("AP50-95_")]
    order = np.argsort([f("full", f"AP50-95_{c}") for c in classes])
    classes = [classes[i] for i in order]
    y, h = np.arange(len(classes)), 0.36
    full_v = [f("full", f"AP50-95_{c}") for c in classes]
    bn_v = [f("bottleneck", f"AP50-95_{c}") for c in classes]
    ax.barh(y + h / 2 + 0.006, full_v, h, label="full ONNX", color=FULL, linewidth=0)
    ax.barh(y - h / 2 - 0.006, bn_v, h, label="split + bottleneck", color=BNECK, linewidth=0)
    for yi, (a, b) in enumerate(zip(full_v, bn_v)):
        ax.text(max(a, b) + 0.015, yi, f"{b - a:+.3f}", va="center", fontsize=8,
                color=INK_2 if abs(b - a) < 0.05 else "#e34948")
    ax.set_yticks(y, classes, color=INK)
    ax.set_xlim(0, 1.16)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel("AP50-95", fontsize=9, color=INK_2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_title("Per-class accuracy — leopard holds", fontsize=11.5, color=INK,
                 weight="bold", loc="left", pad=24)
    ax.text(0, 1.028, "delta labelled at each pair; hyena (n=12) is the only large drop",
            transform=ax.transAxes, fontsize=8.5, color=INK_2)
    legend_below(ax, 2)

    # --- C: wire payload -------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    style(ax)
    raw_kb = f("split", "wire_bytes") / 1024
    bn_kb = f("bottleneck", "wire_bytes") / 1024
    labels = ["raw layer-3 tensor\n(fp32)", "bottleneck\n(uint8)", "JPEG q75\n(measured)"]
    vals, colours = [raw_kb, bn_kb, JPEG_Q75_KB], [SPLIT, BNECK, GRID]
    bars = ax.bar(labels, vals, 0.5, color=colours, linewidth=0)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 26, f"{v:.1f} KB",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.text(1.5, raw_kb * 0.42, f"{raw_kb / bn_kb:.0f}× smaller\nthan fp32", ha="center",
            fontsize=11, color=INK, weight="bold", linespacing=1.4)
    ax.set_ylim(0, raw_kb * 1.18)
    ax.set_ylabel("KB per frame", fontsize=9, color=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_title("Wire payload — 64× smaller, but not smaller than JPEG",
                 fontsize=11.5, color=INK, weight="bold", loc="left", pad=24)
    ax.text(0, 1.028, "at q75 a JPEG of the same frame is 23.2 KB — bandwidth alone "
            "does not justify the split",
            transform=ax.transAxes, fontsize=8.5, color=INK_2)

    # --- D: compute split ------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    style(ax)
    full_m = conv_macs(Path(args.full_dir) / f"best_{args.imgsz}.onnx")
    enc_m = conv_macs(Path(args.split_dir) / f"backbone_enc_{args.imgsz}.onnx")
    dec_m = conv_macs(Path(args.split_dir) / f"dec_head_{args.imgsz}.onnx")
    edge_pct, head_pct = 100 * enc_m / full_m, 100 * dec_m / full_m

    ax.barh(1, 100, 0.34, color=FULL, linewidth=0, label="full model, one device")
    ax.text(50, 1, "100%", ha="center", va="center", fontsize=9.5, color="white", weight="bold")
    ax.barh(0, edge_pct, 0.34, color=SPLIT, linewidth=0, label="Pi Zero (backbone + encoder)")
    ax.barh(0, head_pct, 0.34, left=edge_pct + 0.35, color=BNECK, linewidth=0,
            label="Pi 3 (decoder + head)")
    ax.text(edge_pct / 2, 0, f"{edge_pct:.1f}%", ha="center", va="center",
            fontsize=9, color="white", weight="bold")
    ax.text(edge_pct + head_pct / 2, 0, f"{head_pct:.1f}%", ha="center", va="center",
            fontsize=9.5, color="white", weight="bold")
    ax.axvline(100, color=INK_2, lw=0.9)
    ax.text(edge_pct + head_pct + 1.5, 0.34, f"+{edge_pct + head_pct - 100:.1f}%\noverhead",
            va="center", fontsize=8.5, color="#e34948", linespacing=1.3)
    ax.set_yticks([0, 1], ["split", "unsplit"], color=INK)
    ax.set_xlim(0, 122)
    ax.set_xlabel("% of full-model GMACs", fontsize=9, color=INK_2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_title("Compute — the head still does 85% of the work", fontsize=11.5,
                 color=INK, weight="bold", loc="left", pad=24)
    ax.text(0, 1.028, f"total exceeds 100% — the decoder adds "
            f"{edge_pct + head_pct - 100:.1f}%; MACs are not latency, Pi timings decide",
            transform=ax.transAxes, fontsize=8.5, color=INK_2)
    legend_below(ax, 3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"[+] {out}")
    print(f"[*] compute: full {full_m/1e9:.3f} GMACs | edge {edge_pct:.1f}% | head {head_pct:.1f}%")


if __name__ == "__main__":
    main()
