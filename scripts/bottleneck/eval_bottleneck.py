"""
Evaluate detection accuracy with and without the learned bottleneck.

The v2 model was trained at 640, so its mAP at 256 is not a known quantity. Measure the
unmodified 256 baseline FIRST — it is the only fair comparison target for the bottleneck.

Usage:
    # Baseline: unmodified model at 256 (run this before anything else)
    python scripts/bottleneck/eval_bottleneck.py --baseline-only --imgsz 256

    # With a trained bottleneck injected at layer 4
    python scripts/bottleneck/eval_bottleneck.py --ckpt models/bottleneck/bneck_16ch.pt --imgsz 256

    # Skip the uint8 simulation to isolate how much loss comes from quantization
    python scripts/bottleneck/eval_bottleneck.py --ckpt ... --no-quant
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.bottleneck.modules import load_bottleneck, wrap_layer4  # noqa: E402


def evaluate(weights, data, imgsz, batch, device, split, ckpt=None, quantize=True):
    """Run Ultralytics validation, optionally with the bottleneck wrapped in at layer 4."""
    from ultralytics import YOLO

    model = YOLO(str(weights))

    if ckpt is not None:
        block = load_bottleneck(ckpt)
        if not quantize:
            block.quantize = False
        wrap_layer4(model.model, block)
        scale, zero_point = block.quant.qparams()
        print(f"[*] Bottleneck injected at layer 4")
        print(f"    Channels     : {block.ch_in} -> {block.ch_bottleneck}")
        print(f"    Wire (uint8) : {block.wire_bytes(imgsz // 8) / 1024:.1f} KB/frame")
        print(f"    Quantization : {'uint8 simulated' if block.quantize else 'DISABLED'}")
        if block.quantize:
            print(f"    scale={scale.item():.6g}  zero_point={zero_point.item():.0f}")
    else:
        raw_kb = 128 * (imgsz // 8) ** 2 * 4 / 1024
        print(f"[*] Baseline — no bottleneck (wire would be {raw_kb:.0f} KB/frame fp32)")

    results = model.val(
        data=str(data),
        imgsz=imgsz,
        batch=batch,
        device=device,
        split=split,
        plots=False,
        verbose=True,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate the layer-4 bottleneck's effect on mAP")
    parser.add_argument("--weights", default=str(ROOT / "models/full/best_v2.pt"),
                        help="PyTorch weights (default: models/full/best_v2.pt)")
    parser.add_argument("--data", default=str(ROOT / "data/v2_yolo_dataset/data.yaml"),
                        help="Dataset yaml")
    parser.add_argument("--ckpt", default=None, help="Bottleneck checkpoint (.pt)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Evaluate the unmodified model (ignores --ckpt)")
    parser.add_argument("--no-quant", action="store_true",
                        help="Disable the simulated uint8 round-trip")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    ckpt = None if args.baseline_only else args.ckpt
    if ckpt is None and not args.baseline_only:
        print("[!] Pass --ckpt <bottleneck.pt> or --baseline-only")
        sys.exit(1)

    results = evaluate(
        weights=args.weights,
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        ckpt=ckpt,
        quantize=not args.no_quant,
    )

    box = results.box
    print(f"\n{'=' * 60}")
    print(f"RESULT  ({'baseline' if ckpt is None else Path(ckpt).name}) @ {args.imgsz}px")
    print(f"{'=' * 60}")
    print(f"  mAP50-95 : {box.map:.4f}")
    print(f"  mAP50    : {box.map50:.4f}")
    print(f"  precision: {box.mp:.4f}")
    print(f"  recall   : {box.mr:.4f}")
    try:
        # Class 0 is leopard — the class that actually matters for deployment.
        print(f"  leopard AP50-95: {box.maps[0]:.4f}")
    except (IndexError, TypeError):
        pass
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
