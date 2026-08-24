"""
Build the master v3 comparison table: baseline vs split-only vs split+bottleneck.

Three configurations, all at the layer-3 split point, all sequential (K=1), so
every column is directly comparable. Sources:

  ACCURACY   results/benchmarks/accuracy_three_way_v3.csv
             1,212 validation images, deployed ONNX artifacts, real uint8
             round-trip. Accuracy is device-independent (identical graphs give
             identical outputs), so this is authoritative for every arm.

  LATENCY    results/benchmarks/pi_latency_raw_split_v3.csv    (split only, fp16)
             results/benchmarks/pi_latency_bottleneck_v3.csv   (split + bottleneck)
             Real Pi Zero 2W -> Pi 3 over the 172.20.10.x hotspot LAN.

  EXACTNESS  constants below, from scripts/verify_split.py and
             scripts/bottleneck/verify_bottleneck.py (138 images each).

Layer-4 cuts and the K=2/K=4 pipelined runs exist in the raw CSVs but are left
out here: this project deploys the layer-3 split, and mixing cut depths or
pipeline depths into the headline table makes it unreadable.

Usage:
    python scripts/bottleneck/build_comparison.py
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- secondary sanity check on a DIFFERENT, smaller sample -------------------
# scripts/verify_split.py / verify_bottleneck.py, over data/images/ (138 files,
# not the 1,212-image validation set). Kept ONLY as a labeled footnote in
# `notes` below -- never as the headline detection-correctness figure, so a
# reader never has to reconcile two different sample sizes in one place.
SPLIT_EXACT_N = 138          # verify_split.py: 138/138 PASS, max abs err 0.0
BNECK_CHECK_N = 138          # verify_bottleneck.py, conf>=0.3
BNECK_REF_DETS = 114         # detections found within those 138 images (not a image count)
BNECK_MATCHED = 110
BNECK_MEAN_IOU = 0.9674
BNECK_CLIP_PCT = 0.166

# B1 reference from the Pi campaign (pipeline_v6.log). Independently reproduced
# at 2642.4 ms by our own run's reference pass -- 0.5 % apart.
B1_E2E_MS = 2629.6
B1_REF_IMAGES = 20

# Measured from the ONNX graphs (scripts/bottleneck/plot_results.py)
GMACS_FULL, GMACS_EDGE, GMACS_TAIL = 10.768, 2.567, 9.157

# The split-only arm was shipped fp16 on the Pis and fp32 in the accuracy pass.
# fp16 is what the latency columns describe; fp32 is noted for reference only.
SPLIT_FP32_KB = 1600.0

COLUMNS = [
    "config_id", "label", "arm", "devices", "split_layer", "wire_dtype", "inflight_k",
    "payload_bytes", "payload_kb", "wire_ms", "head_ms", "tail_ms",
    "e2e_ms", "e2e_ms_sd", "throughput_fps", "fps_vs_baseline_pct",
    "mAP50_95", "mAP50", "precision", "recall", "mAP50_95_delta",
    "accuracy_n_images", "accuracy_n_preds", "accuracy_n_targets", "accuracy_source",
    "latency_n_runs", "latency_n_steady_frames", "latency_n_unique_images",
    "latency_source", "detection_verdict", "box_max_px", "measured", "notes",
]


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def agg(rows, key):
    vals = [fnum(r.get(key)) for r in rows if fnum(r.get(key)) is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acc", default=str(ROOT / "results/benchmarks/accuracy_three_way_v3.csv"))
    ap.add_argument("--raw", default=str(ROOT / "results/benchmarks/pi_latency_raw_split_v3.csv"))
    ap.add_argument("--bneck", default=str(ROOT / "results/benchmarks/pi_latency_bottleneck_v3.csv"))
    ap.add_argument("--out", default=str(ROOT / "results/benchmarks/comprehensive_comparison_v3.csv"))
    args = ap.parse_args()

    acc = {r["config"]: r for r in csv.DictReader(open(args.acc, newline=""))}
    raw = list(csv.DictReader(open(args.raw, newline="")))
    bn = list(csv.DictReader(open(args.bneck, newline="")))

    # layer-3 cut (cut_idx 10), sequential K=1 -- the arm the bottleneck is compared against
    split = [r for r in raw if r["cut_idx"] == "10" and r["inflight_k"] == "1"]
    if not split:
        raise SystemExit("no layer-3 K=1 rows found in the raw split CSV")

    base_fps = 1000.0 / B1_E2E_MS
    ACC_SRC = "accuracy_three_way_v3.csv (1,212 val images, deployed ONNX, real uint8 round-trip)"
    PI_SRC = "Pi Zero 2W -> Pi 3, 172.20.10.x hotspot LAN, sequential K=1"

    def acc_cols(key):
        r = acc[key]
        return {
            "mAP50_95": round(fnum(r["mAP50-95"]), 6),
            "mAP50": round(fnum(r["mAP50"]), 6),
            "precision": round(fnum(r["precision"]), 6),
            "recall": round(fnum(r["recall"]), 6),
            "mAP50_95_delta": round(fnum(r["mAP50-95"]) - fnum(acc["full"]["mAP50-95"]), 6),
            "accuracy_n_images": r["n_images"],
            "accuracy_n_preds": r["n_preds"],
            "accuracy_n_targets": r["n_targets"],
            "accuracy_source": ACC_SRC,
        }

    rows = []

    # ---- 1. baseline: full model on one device, nothing transmitted -----------
    rows.append({
        "config_id": "baseline", "label": "Baseline - full model, single Pi Zero",
        "arm": "baseline", "devices": "Pi Zero 2W only", "split_layer": "",
        "wire_dtype": "", "inflight_k": 1,
        "payload_bytes": 0, "payload_kb": 0.0, "wire_ms": 0.0,
        "head_ms": round(B1_E2E_MS, 1), "tail_ms": 0.0,
        "e2e_ms": round(B1_E2E_MS, 1), "e2e_ms_sd": "",
        "throughput_fps": round(base_fps, 4), "fps_vs_baseline_pct": 0.0,
        **acc_cols("full"),
        "latency_n_runs": 2, "latency_n_steady_frames": "",
        "latency_n_unique_images": B1_REF_IMAGES, "latency_source": PI_SRC,
        "detection_verdict": "REFERENCE", "box_max_px": 0.0, "measured": "yes",
        "notes": "nothing crosses a network; whole model runs on the Pi Zero",
    })

    # ---- 2. split only, layer 3 ----------------------------------------------
    # fps is 1000/e2e, not the harness's throughput_fps_steady: that column
    # measures completion-timestamp spacing, which INCLUDES SD-card read time
    # (~140 ms/frame). e2e_ms starts its clock after cv2.imread (matching the
    # baseline's own definition), so deriving fps from it keeps all three rows
    # on one consistent basis -- otherwise the baseline looks artificially fast.
    e2e_s = agg(split, "e2e_ms_mean")
    fps_s = 1000.0 / e2e_s
    rows.append({
        "config_id": "split_only", "label": "Split only - layer 3, raw tensor",
        "arm": "split", "devices": "Pi Zero 2W -> Pi 3", "split_layer": 3,
        "wire_dtype": "float16", "inflight_k": 1,
        "payload_bytes": int(agg(split, "bytes_up_mean")),
        "payload_kb": round(agg(split, "kb_per_frame"), 2),
        "wire_ms": round(agg(split, "wire_ms_mean"), 1),
        "head_ms": round(agg(split, "head_ms_mean"), 1),
        "tail_ms": round(agg(split, "tail_ms_mean"), 1),
        "e2e_ms": round(e2e_s, 1), "e2e_ms_sd": round(agg(split, "e2e_ms_sd"), 1),
        "throughput_fps": round(fps_s, 4),
        "fps_vs_baseline_pct": round(100 * (fps_s / base_fps - 1), 1),
        **acc_cols("split"),
        "latency_n_runs": len(split),
        "latency_n_steady_frames": sum(int(r["n_steady"]) for r in split),
        "latency_n_unique_images": 20, "latency_source": PI_SRC,
        "detection_verdict": f"EXACT: {acc['split']['n_preds']}/{acc['full']['n_preds']} predictions match (1,212 images)",
        "box_max_px": max(fnum(r["box_max_px"]) for r in split), "measured": "yes",
        "notes": (f"fp16 on the wire ({SPLIT_FP32_KB:.0f} KB if sent fp32); bit-exact vs the "
                  f"full model on every accuracy metric (+0.0000). Separately, "
                  f"scripts/verify_split.py checked {SPLIT_EXACT_N}/{SPLIT_EXACT_N} images "
                  f"from a different, smaller sample (data/images/) and found max abs "
                  f"error 0.0 -- a secondary sanity check, not this row's accuracy figure."),
    })

    # ---- 3. split + learned bottleneck, layer 3 ------------------------------
    e2e_b = agg(bn, "e2e_ms_mean")
    fps_b = 1000.0 / e2e_b
    tail_b = agg(bn, "tail_ms_mean")
    rows.append({
        "config_id": "split_bottleneck", "label": "Split + learned bottleneck - layer 3, uint8",
        "arm": "bottleneck", "devices": "Pi Zero 2W -> Pi 3", "split_layer": 3,
        "wire_dtype": "uint8", "inflight_k": 1,
        "payload_bytes": int(agg(bn, "bytes_up_mean")),
        "payload_kb": round(agg(bn, "kb_per_frame"), 2),
        "wire_ms": round(agg(bn, "wire_ms_mean"), 1),
        "head_ms": round(agg(bn, "head_ms_mean"), 1), "tail_ms": round(tail_b, 1),
        "e2e_ms": round(e2e_b, 1), "e2e_ms_sd": round(agg(bn, "e2e_ms_sd"), 1),
        "throughput_fps": round(fps_b, 4),
        "fps_vs_baseline_pct": round(100 * (fps_b / base_fps - 1), 1),
        **acc_cols("bottleneck"),
        "latency_n_runs": len(bn),
        "latency_n_steady_frames": sum(int(r["n_steady"]) for r in bn),
        "latency_n_unique_images": 20, "latency_source": PI_SRC,
        "detection_verdict": (f"{acc['bottleneck']['n_preds']}/{acc['full']['n_preds']} predictions "
                              f"({100*fnum(acc['bottleneck']['n_preds'])/fnum(acc['full']['n_preds']):.1f}%) "
                              f"(1,212 images)"),
        "box_max_px": max(fnum(r["box_max_px"]) for r in bn), "measured": "yes",
        "notes": (f"256->16 channels, uint8 wire; 25 KB tensor + protocol header; "
                  f"quantizer clipping {BNECK_CLIP_PCT}% of values. Separately, "
                  f"scripts/bottleneck/verify_bottleneck.py checked {BNECK_CHECK_N} images "
                  f"from a different, smaller sample (data/images/) and matched "
                  f"{BNECK_MATCHED}/{BNECK_REF_DETS} detections, mean IoU {BNECK_MEAN_IOU} "
                  f"-- a secondary sanity check, not this row's accuracy figure."),
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"[+] {out}  ({len(rows)} configurations)")
    print(f"[*] compute: full {GMACS_FULL} GMACs | Pi Zero {100*GMACS_EDGE/GMACS_FULL:.1f}% "
          f"| Pi 3 {100*GMACS_TAIL/GMACS_FULL:.1f}%")
    hdr = f"{'config':<20}{'KB/frame':>10}{'wire ms':>9}{'e2e ms':>9}{'fps':>8}{'mAP50-95':>10}"
    print(hdr)
    for r in rows:
        print(f"{r['config_id']:<20}{r['payload_kb']:>10}{r['wire_ms']:>9}"
              f"{r['e2e_ms']:>9}{r['throughput_fps']:>8}{r['mAP50_95']:>10}")


if __name__ == "__main__":
    main()
