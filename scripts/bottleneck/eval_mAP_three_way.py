"""
Three-way accuracy comparison of the ACTUAL DEPLOYED ONNX ARTIFACTS.

    full        models/full/best_256.onnx
    split       backbone_256.onnx -> head_256.onnx
    bottleneck  backbone_enc_256.onnx -> uint8 round-trip -> dec_head_256.onnx

Unlike eval_bottleneck.py (which evaluates the PyTorch model with the bottleneck wrapped
into net.model[4]), this runs the exact ONNX files that ship to the Pis, including the
real uint8 wire quantization. That closes the gap flagged in bottleneck_handoff.md §9.

Accuracy is device-independent — identical ONNX graphs produce identical outputs on Pi or
laptop (two uint8 Pi runs were previously shown to be bit-identical) — so this runs on the
laptop only.

All three configs share ONE preprocessing path (plain cv2.resize, matching
backbone_server.py) so the comparison is apples-to-apples. Note this differs from
Ultralytics' letterbox, so absolute numbers will not exactly match `model.val()`; the
three-way DELTA is what this script is for.

Metrics come from ultralytics.utils.metrics.ap_per_class, the same function model.val()
uses, so mAP50 / mAP50-95 / P / R / F1 are computed identically.

Usage:
    python scripts/bottleneck/eval_three_way.py
    python scripts/bottleneck/eval_three_way.py --limit 100        # quick smoke test
    python scripts/bottleneck/eval_three_way.py --configs full bottleneck
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics.utils.metrics import ap_per_class, box_iou

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.bottleneck.modules import dequantize_uint8, quantize_uint8  # noqa: E402

CLASS_NAMES = {
    0: "leopard", 1: "cheetah", 2: "hyena", 3: "nubian_ibex",
    4: "camel", 5: "cat", 6: "dog", 7: "person",
}

# 10 IoU thresholds 0.50:0.05:0.95 — the COCO/Ultralytics convention.
IOUV = torch.linspace(0.5, 0.95, 10)


def build_configs(split_dir, full_dir, imgsz):
    meta_path = split_dir / f"bottleneck_{imgsz}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    cfgs = {
        "full": dict(
            label="full ONNX (unmodified)",
            stages=[full_dir / f"best_{imgsz}.onnx"],
            quant=None,
        ),
        "split": dict(
            label="split (no bottleneck)",
            stages=[split_dir / f"backbone_{imgsz}.onnx", split_dir / f"head_{imgsz}.onnx"],
            quant=None,
        ),
        "bottleneck": dict(
            label="split + bottleneck (uint8)",
            stages=[split_dir / meta["backbone_enc"], split_dir / meta["dec_head"]] if meta else [],
            quant=(meta["scale"], meta["zero_point"]) if meta else None,
        ),
    }
    return cfgs


class Pipeline:
    """One or two ONNX stages, with an optional uint8 round-trip between them."""

    def __init__(self, stages, quant=None):
        self.sessions = []
        for path in stages:
            if not Path(path).exists():
                raise SystemExit(f"[!] Missing model: {path}")
            s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.sessions.append((s, s.get_inputs()[0].name, [o.name for o in s.get_outputs()]))
        self.quant = quant
        self.wire_bytes = None

    def __call__(self, x):
        for i, (sess, inp, outs) in enumerate(self.sessions):
            x = sess.run(outs, {inp: x})[0]
            is_boundary = i == 0 and len(self.sessions) > 1
            if is_boundary:
                if self.quant is not None:
                    scale, zero_point = self.quant
                    q = quantize_uint8(x, scale, zero_point)
                    self.wire_bytes = q.nbytes
                    x = dequantize_uint8(q, scale, zero_point)
                else:
                    self.wire_bytes = x.astype(np.float32).nbytes
        return x


def preprocess(path, imgsz):
    """Matches BackboneServer.preprocess() — the deployment path."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(img, (2, 0, 1)), 0)


def load_labels(label_path, imgsz):
    """YOLO `cls cx cy w h` (normalised) -> (cls, xyxy) in imgsz pixel space.

    Plain-resize squashes the image, so normalised coords scale directly with no
    letterbox offset to undo.
    """
    if not label_path.exists():
        return np.zeros(0), np.zeros((0, 4))
    rows = [r.split() for r in label_path.read_text().strip().splitlines() if r.strip()]
    if not rows:
        return np.zeros(0), np.zeros((0, 4))
    arr = np.array(rows, dtype=np.float32)
    cls, cx, cy, w, h = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    xyxy = np.stack([(cx - w / 2), (cy - h / 2), (cx + w / 2), (cy + h / 2)], 1) * imgsz
    return cls, xyxy


def match(pred_cls, pred_xyxy, true_cls, true_xyxy):
    """Replicates BaseValidator.match_predictions: (N, 10) bool of correct-at-IoU."""
    correct = np.zeros((len(pred_cls), len(IOUV)), dtype=bool)
    if len(true_cls) == 0 or len(pred_cls) == 0:
        return correct

    iou = box_iou(torch.from_numpy(true_xyxy).float(), torch.from_numpy(pred_xyxy).float())
    iou = (iou * (torch.from_numpy(true_cls)[:, None] == torch.from_numpy(pred_cls))).numpy()

    for i, thr in enumerate(IOUV.tolist()):
        m = np.array(np.nonzero(iou >= thr)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def evaluate(pipe, images, labels_dir, imgsz, conf_thresh):
    stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
    t_total, n = 0.0, 0

    for k, img_path in enumerate(images, 1):
        x = preprocess(img_path, imgsz)
        if x is None:
            continue

        t0 = time.perf_counter()
        raw = pipe(x)[0]  # (300, 6)
        t_total += time.perf_counter() - t0
        n += 1

        keep = raw[:, 4] >= conf_thresh
        det = raw[keep]
        pred_xyxy, pred_conf = det[:, :4], det[:, 4]
        pred_cls = np.round(det[:, 5]).astype(int)

        true_cls, true_xyxy = load_labels(labels_dir / f"{img_path.stem}.txt", imgsz)

        stats["tp"].append(match(pred_cls, pred_xyxy, true_cls, true_xyxy))
        stats["conf"].append(pred_conf)
        stats["pred_cls"].append(pred_cls)
        stats["target_cls"].append(true_cls)

        if k % 200 == 0:
            print(f"      {k}/{len(images)}", flush=True)

    tp = np.concatenate(stats["tp"]) if stats["tp"] else np.zeros((0, 10), bool)
    conf = np.concatenate(stats["conf"]) if stats["conf"] else np.zeros(0)
    pred_cls = np.concatenate(stats["pred_cls"]) if stats["pred_cls"] else np.zeros(0)
    target_cls = np.concatenate(stats["target_cls"]) if stats["target_cls"] else np.zeros(0)

    _, _, p, r, f1, ap, cls_idx, *_ = ap_per_class(
        tp, conf, pred_cls, target_cls, plot=False, names=CLASS_NAMES
    )

    return {
        "mAP50": float(ap[:, 0].mean()),
        "mAP50-95": float(ap.mean()),
        "precision": float(p.mean()),
        "recall": float(r.mean()),
        "f1": float(f1.mean()),
        "per_class": {int(c): (float(ap[i, 0]), float(ap[i].mean()), float(p[i]), float(r[i]))
                      for i, c in enumerate(cls_idx)},
        "n_images": n,
        "n_preds": int(len(conf)),
        "n_targets": int(len(target_cls)),
        "ms_per_image": 1000 * t_total / max(n, 1),
        "wire_bytes": pipe.wire_bytes,
    }


def main():
    ap_ = argparse.ArgumentParser(description="Three-way ONNX accuracy comparison")
    ap_.add_argument("--data", default=str(ROOT / "data/v2_yolo_dataset/valid"))
    ap_.add_argument("--full-dir", default=str(ROOT / "models/full"))
    ap_.add_argument("--split-dir", default=str(ROOT / "models/split"))
    ap_.add_argument("--out", default=str(ROOT / "results/benchmarks/accuracy_three_way.csv"))
    ap_.add_argument("--imgsz", type=int, default=256)
    ap_.add_argument("--conf", type=float, default=0.001,
                     help="Low threshold so the full PR curve is captured (val convention)")
    ap_.add_argument("--limit", type=int, default=None, help="Smoke-test on N images")
    ap_.add_argument("--configs", nargs="+", default=["full", "split", "bottleneck"])
    args = ap_.parse_args()

    data = Path(args.data)
    images_dir, labels_dir = data / "images", data / "labels"
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"[!] No images in {images_dir}")

    cfgs = build_configs(Path(args.split_dir), Path(args.full_dir), args.imgsz)
    print(f"[*] {len(images)} val images @ {args.imgsz}px, conf>={args.conf}")
    print(f"[*] Preprocessing: plain cv2.resize (deployment-matched, identical for all configs)\n")

    results = {}
    for name in args.configs:
        cfg = cfgs[name]
        print(f"[*] {name}: {cfg['label']}")
        for s in cfg["stages"]:
            print(f"      {Path(s).name}")
        pipe = Pipeline(cfg["stages"], cfg["quant"])
        results[name] = evaluate(pipe, images, labels_dir, args.imgsz, args.conf)
        print(f"      done — {results[name]['ms_per_image']:.1f} ms/img\n")

    # ---- comparison table -----------------------------------------------------
    w = 26
    print("=" * 92)
    print("OVERALL  (validation split, all classes)")
    print("=" * 92)
    print(f"{'config':<{w}}{'mAP50-95':>10}{'mAP50':>9}{'precision':>11}{'recall':>9}"
          f"{'F1':>8}{'wire KB':>10}{'ms/img':>9}")
    base = results.get("full")
    for name in args.configs:
        r = results[name]
        kb = f"{r['wire_bytes'] / 1024:.1f}" if r["wire_bytes"] else "-"
        print(f"{cfgs[name]['label']:<{w}}{r['mAP50-95']:>10.4f}{r['mAP50']:>9.4f}"
              f"{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1']:>8.4f}{kb:>10}{r['ms_per_image']:>9.1f}")

    if base and len(args.configs) > 1:
        print(f"\n{'delta vs full ONNX':<{w}}{'mAP50-95':>10}{'mAP50':>9}{'precision':>11}{'recall':>9}{'F1':>8}")
        for name in args.configs:
            if name == "full":
                continue
            r = results[name]
            print(f"{cfgs[name]['label']:<{w}}{r['mAP50-95'] - base['mAP50-95']:>+10.4f}"
                  f"{r['mAP50'] - base['mAP50']:>+9.4f}{r['precision'] - base['precision']:>+11.4f}"
                  f"{r['recall'] - base['recall']:>+9.4f}{r['f1'] - base['f1']:>+8.4f}")

    # ---- per class ------------------------------------------------------------
    print(f"\n{'=' * 92}\nPER-CLASS mAP50-95\n{'=' * 92}")
    print(f"{'class':<14}" + "".join(f"{n:>22}" for n in args.configs))
    for cid in sorted(CLASS_NAMES):
        cells = ""
        for name in args.configs:
            pc = results[name]["per_class"].get(cid)
            cells += f"{pc[1]:>22.4f}" if pc else f"{'-':>22}"
        print(f"{CLASS_NAMES[cid]:<14}{cells}")

    # ---- csv ------------------------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["config", "label", "mAP50-95", "mAP50", "precision", "recall", "f1",
                      "wire_bytes", "ms_per_image", "n_images", "n_preds", "n_targets"]
                     + [f"AP50-95_{CLASS_NAMES[c]}" for c in sorted(CLASS_NAMES)])
        for name in args.configs:
            r = results[name]
            wtr.writerow([name, cfgs[name]["label"], f"{r['mAP50-95']:.6f}", f"{r['mAP50']:.6f}",
                          f"{r['precision']:.6f}", f"{r['recall']:.6f}", f"{r['f1']:.6f}",
                          r["wire_bytes"], f"{r['ms_per_image']:.2f}", r["n_images"],
                          r["n_preds"], r["n_targets"]]
                         + [f"{r['per_class'][c][1]:.6f}" if c in r["per_class"] else ""
                            for c in sorted(CLASS_NAMES)])
    print(f"\n[+] {out}")


if __name__ == "__main__":
    main()
