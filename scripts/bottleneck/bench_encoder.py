"""
Measure what the encoder actually costs on the field device.

The whole premise of the learned bottleneck is that the encoder is nearly free: layers
0-4 are only ~75K params but are FLOP-heavy because they run at high spatial resolution,
whereas a 1x1 conv at the 32x32 grid is ~2.1M MACs. This script checks that claim on
real hardware instead of trusting the arithmetic.

Run this ON THE PI ZERO:
    python3 scripts/bottleneck/bench_encoder.py --imgsz 256 --runs 30

Reports backbone alone vs backbone+encoder. The delta is the price of the bottleneck;
compare it against the bytes it saves.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]


def bench(path, input_shape, runs, warmup):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    x = np.random.rand(*input_shape).astype(np.float32)

    for _ in range(warmup):
        sess.run(outs, {name: x})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        y = sess.run(outs, {name: x})[0]
        times.append((time.perf_counter() - t0) * 1000)
    return times, y.shape


def report(label, times):
    print(f"  {label:<26} mean {statistics.mean(times):7.2f} ms   "
          f"min {min(times):7.2f}   max {max(times):7.2f}")
    return statistics.mean(times)


def main():
    parser = argparse.ArgumentParser(description="Benchmark the encoder's cost on the field device")
    parser.add_argument("--split-dir", default=str(ROOT / "models/split"))
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    backbone = split_dir / f"backbone_{args.imgsz}.onnx"
    backbone_enc = split_dir / f"backbone_enc_{args.imgsz}.onnx"

    for p in (backbone, backbone_enc):
        if not p.exists():
            raise SystemExit(f"[!] Missing {p}\n    Run scripts/bottleneck/export_bottleneck.py first")

    shape = (1, 3, args.imgsz, args.imgsz)
    print(f"[*] imgsz {args.imgsz}, {args.runs} runs ({args.warmup} warmup)\n")

    t_backbone, shape_b = bench(backbone, shape, args.runs, args.warmup)
    t_combined, shape_c = bench(backbone_enc, shape, args.runs, args.warmup)

    print("LATENCY")
    mean_b = report("backbone (layers 0-4)", t_backbone)
    mean_c = report("backbone + encoder", t_combined)
    delta = mean_c - mean_b
    print(f"  {'encoder overhead':<26} {delta:+7.2f} ms   "
          f"({100 * delta / mean_b:+.1f} % of backbone)")

    bytes_b = int(np.prod(shape_b)) * 4
    bytes_c = int(np.prod(shape_c))  # uint8 on the wire
    print(f"\nWIRE")
    print(f"  {'without bottleneck':<26} {bytes_b / 1024:7.1f} KB/frame  fp32 {shape_b}")
    print(f"  {'with bottleneck (uint8)':<26} {bytes_c / 1024:7.1f} KB/frame  uint8 {shape_c}")
    print(f"  {'reduction':<26} {bytes_b / bytes_c:7.0f}x")

    saved_kb = (bytes_b - bytes_c) / 1024
    print(f"\nVERDICT")
    print(f"  Costs {delta:+.2f} ms on the field device to remove {saved_kb:.0f} KB "
          f"from every frame.")

    # Below the run-to-run spread the overhead is not measurable, and the break-even
    # link speed it implies would be meaningless.
    noise = (max(t_backbone) - min(t_backbone)) / 2
    if delta <= noise:
        print(f"  Overhead is within measurement noise (+/-{noise:.2f} ms) — free at this "
              f"resolution.")
    else:
        breakeven = saved_kb / delta
        print(f"  Worth it whenever the link is slower than {breakeven:.1f} KB/ms "
              f"({breakeven * 8 / 1000:.1f} Mbps).")


if __name__ == "__main__":
    main()
