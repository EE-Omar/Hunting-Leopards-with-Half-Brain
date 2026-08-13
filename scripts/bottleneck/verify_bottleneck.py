"""
End-to-end check of the deployed bottleneck path, including the real uint8 wire round-trip.

Compares, per image:
    full model                        best_320.onnx
    bottleneck path   backbone_enc_320.onnx -> uint8 -> dequant -> dec_head_320.onnx

Unlike scripts/verify_split.py, this does NOT assert numerical equivalence — the
bottleneck is lossy by construction and a `max abs error < 1e-3` test would fail by
design. The gates here are wire size and detection agreement; mAP is the real gate and
lives in eval_bottleneck.py.

Usage:
    python scripts/bottleneck/verify_bottleneck.py data/images/ --imgsz 320
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.bottleneck.modules import SPLIT_CHANNELS, dequantize_uint8, quantize_uint8  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def load(path):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name, [o.name for o in sess.get_outputs()]


def run(bundle, tensor):
    sess, inp, outs = bundle
    return sess.run(outs, {inp: tensor})[0]


def preprocess(path, imgsz):
    """Matches BackboneServer.preprocess()."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(img, (2, 0, 1)), 0)


def detections(raw, conf_thresh):
    """Decode (1,300,6) -> [x1,y1,x2,y2,conf,cls]. Output is already corner format."""
    dets = raw[0]
    return dets[dets[:, 4] >= conf_thresh]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(ref, test, iou_thresh=0.5):
    """Greedy IoU matching between two detection sets of the same image."""
    used, ious = set(), []
    for r in ref:
        best_j, best_iou = -1, 0.0
        for j, t in enumerate(test):
            if j in used or int(round(t[5])) != int(round(r[5])):
                continue
            v = iou(r[:4], t[:4])
            if v > best_iou:
                best_j, best_iou = j, v
        if best_j >= 0 and best_iou >= iou_thresh:
            used.add(best_j)
            ious.append(best_iou)
    return ious, len(ref) - len(ious), len(test) - len(used)


def main():
    parser = argparse.ArgumentParser(description="Verify the bottleneck path end to end")
    parser.add_argument("images", help="Folder of test images")
    parser.add_argument("--full", default=None, help="Full model (default: models/full/best_<imgsz>.onnx)")
    parser.add_argument("--split-dir", default=str(ROOT / "models/split"))
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    meta = json.loads((split_dir / f"bottleneck_{args.imgsz}.json").read_text())
    scale, zero_point = meta["scale"], meta["zero_point"]

    full_path = args.full or ROOT / f"models/full/best_{args.imgsz}.onnx"
    full = load(full_path)
    backbone_enc = load(split_dir / meta["backbone_enc"])
    dec_head = load(split_dir / meta["dec_head"])

    print(f"[*] full        : {Path(full_path).name}")
    print(f"[*] backbone+enc: {meta['backbone_enc']}")
    print(f"[*] dec+head    : {meta['dec_head']}")
    print(f"[*] wire        : uint8, scale={scale:.6g} zero_point={zero_point}")

    folder = Path(args.images)
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"[!] No images found in {folder}")
    print(f"[*] {len(images)} images\n")

    raw_bytes = SPLIT_CHANNELS * (args.imgsz // 8) ** 2 * 4
    wire_bytes = None
    all_ious, total_missed, total_extra, total_ref = [], 0, 0, 0
    clipped_total, clipped_count = 0, 0

    for path in images:
        x = preprocess(path, args.imgsz)
        if x is None:
            continue

        ref = detections(run(full, x), args.conf)

        z = run(backbone_enc, x)
        q = quantize_uint8(z, scale, zero_point)
        wire_bytes = q.nbytes
        # Track saturation: heavy clipping means the calibrated range is wrong.
        clipped_total += int(np.sum((q == 0) | (q == 255)))
        clipped_count += q.size
        z_hat = dequantize_uint8(q, scale, zero_point)
        test = detections(run(dec_head, z_hat), args.conf)

        ious, missed, extra = match(ref, test)
        all_ious.extend(ious)
        total_missed += missed
        total_extra += extra
        total_ref += len(ref)

        print(f"  {path.name:<44} ref={len(ref):<3} bneck={len(test):<3} "
              f"matched={len(ious):<3} missed={missed:<3} extra={extra}")

    print(f"\n{'=' * 66}")
    print("WIRE")
    print(f"{'=' * 66}")
    print(f"  fp32 layer-3 tensor : {raw_bytes / 1024:>8.1f} KB/frame")
    print(f"  uint8 bottleneck    : {wire_bytes / 1024:>8.1f} KB/frame  "
          f"({raw_bytes / wire_bytes:.0f}x smaller)")
    print(f"  JPEG frame baseline : {40.0:>8.1f} KB/frame  "
          f"({'PASS' if wire_bytes / 1024 < 40 else 'FAIL'} — must be under this to be worth splitting)")
    print(f"  quantizer clipping  : {100 * clipped_total / max(1, clipped_count):>8.3f} % of values at 0 or 255")

    print(f"\n{'=' * 66}")
    print(f"DETECTION AGREEMENT vs full model (conf >= {args.conf})")
    print(f"{'=' * 66}")
    print(f"  reference detections: {total_ref}")
    print(f"  matched             : {len(all_ious)}"
          f"  ({100 * len(all_ious) / max(1, total_ref):.1f} % recall vs full model)")
    print(f"  missed              : {total_missed}")
    print(f"  spurious            : {total_extra}")
    if all_ious:
        print(f"  mean IoU (matched)  : {np.mean(all_ious):.4f}")
    print(f"{'=' * 66}")
    print("\nNote: numerical equivalence is NOT expected here — the bottleneck is lossy.")
    print("      mAP from eval_bottleneck.py is the real accuracy gate.")


if __name__ == "__main__":
    main()
