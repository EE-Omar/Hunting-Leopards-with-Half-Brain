# Session 4 Handoff — YOLO26-Large Split + Bottleneck (COMPLETE)

**Date:** August 13, 2026
**Status:** Split + bottleneck pipeline done, verified, and benchmarked at 320px. Nothing committed.
**Next Action:** Deploy to Pi Zero + Pi 3 and measure real latency.

---

## What Was Done

Executed the full split + bottleneck pipeline on YOLO26-Large (`models/full/best_v3.pt`),
at **320px only** (decision below). All six bottleneck scripts were restored from the
unmerged `feat/learned-bottleneck` branch and retargeted to layer 3 — 88 insertions /
85 deletions across 10 files, almost entirely constants and defaults.

**Everything now lives on `main`. The branch holds no unique code** (only archived v2
model binaries, which already exist under `models/v2/`). Branch deletion is the user's call.

---

## Corrections to Session 3's Handoff

Session 3 stated Layer 3 was `200x200x128`, 5 KB fp32. **This was wrong by ~1000x.**
Verified from the actual ONNX graph:

| | Session 3 claimed | Actual (measured) |
|---|---|---|
| shape @640 | 200x200x128 | **80x80x256** |
| fp32 size @640 | 5.0 KB | **6.25 MB/frame** |
| shape @320 | — | **40x40x256** (1600 KB fp32) |

Layer 3 is stride-8 (640/8 = 80), not 200x200, and it is 256 channels, not 128.

**Consequence:** at 640, a 16-channel bottleneck is 100 KB/frame, which fails the wire
gate. This is why 320 was chosen. **The 640 path remains unbuilt and unresolved.**

---

## Split Point: Layer 3 (confirmed correct)

```
Tensor:   /model.3/act/Mul_output_0
Shape:    (1, 256, 40, 40) at 320px
Layers:   0-3, all f == -1 (straight chain, clean single cut)
Params:   840,000
```

Layer 3 is the last clean single cut — layer 4 onward feeds the P3 concat skip, so a
deeper split would have to ship more than one tensor.

**Verified lossless:** `scripts/verify_split.py` — 138/138 images PASS, max error
`0.00000000`. Confirmed again at the mAP level (see benchmarks: `+0.0000` on every metric).

---

## Bottleneck: 256 -> 16 channels

```
Encoder (Pi Zero) : Conv2d(256->16, k=1)              4,112 params
Decoder (Pi 3)    : Conv2d(16->256) + Conv2d(256->256, k=3) residual   594,432 params
Quantization      : uint8 affine, 99.9th-percentile calibrated
Wire params       : scale=0.05108868330717087  zero_point=133
Wire payload      : 25.0 KB/frame  (vs 1600 KB fp32 = 64x)
```

**Training:** 3000 cached train images, 40 epochs, reconstruction MSE only, QAT with
straight-through estimator. Converged to val relative L2 **0.2729**, cosine **0.9659**.
Plateaued from epoch ~20 — that is a 16-channel capacity limit, not undertraining.
More epochs will not help; more channels would.

---

## Results

### Accuracy (1,212-image val split, deployed ONNX, real uint8 round-trip)

| config | mAP50-95 | mAP50 | precision | recall | wire KB |
|---|---|---|---|---|---|
| full ONNX | 0.6535 | 0.8323 | 0.8293 | 0.7854 | — |
| split, no bottleneck | 0.6535 | 0.8323 | 0.8293 | 0.7854 | 1600.0 |
| split + bottleneck | 0.6338 | 0.8121 | 0.9014 | 0.7075 | **25.0** |

Bottleneck cost: **-0.0197 mAP50-95 / -0.0202 mAP50**. Well inside the 0.05 gate, so
16 channels stands; no need for the 24-channel fallback.

Cross-checked against the PyTorch-level eval (`eval_bottleneck.py`): -0.025 mAP50-95,
same conclusion. Absolute numbers differ between the two evals (0.6535 vs 0.6924) purely
because the three-way script uses deployment-matched `cv2.resize` and Ultralytics uses
letterbox. **The deltas are the comparable quantity, not the absolutes.**

### Detection agreement vs full model (138 images, conf >= 0.3)

96.5% recall, mean IoU 0.9674, 4 missed / 5 spurious of 114. Quantizer clipping 0.166%.

### v2 vs v3 comparison

Benchmarks saved:
- `results/benchmarks/accuracy_three_way_v3.csv` (current — nano numbers were overwritten and regenerated)
- `results/benchmarks/accuracy_three_way_v2.csv` (v2 nano@256, layer-4, for reference only)

| | v2 nano@256 | v3 large@320 |
|---|---|---|
| full mAP50-95 | 0.5822 | **0.6535** |
| bottleneck mAP50-95 | 0.5727 | **0.6338** |
| bottleneck delta | -0.0095 | -0.0197 |
| wire | 16.0 KB (32x) | 25.0 KB (**64x**) |
| bottleneck compute overhead | **+37.3%** | **+8.9%** |
| full model, laptop CPU | 5.2 ms/img | **300.5 ms/img** |

**v3 with the bottleneck (0.6338) beats v2 without it (0.5822).** The compression tax is
repaid several times over by the larger model.

Caveat when citing the accuracy gap: v3 changes model size *and* resolution (256->320)
simultaneously, so +0.071 mAP50-95 cannot be attributed to capacity alone.

---

## OPEN ISSUES — read before writing the report

### 1. "Smaller than sending a JPEG" is NOT a safe claim

The 40 KB JPEG baseline hardcoded in `verify_bottleneck.py` is a hand-written assumption,
not a measurement. Measured on actual val images at 320x320:

| JPEG quality | mean KB | vs 25.0 KB bottleneck |
|---|---|---|
| q95 | 51.8 | bottleneck 2.1x smaller |
| q85 | 30.6 | bottleneck 1.2x smaller |
| **q75** | **23.2** | **JPEG smaller** |
| q50 | 15.6 | JPEG 1.6x smaller |

At ordinary quality, shipping the JPEG costs the same or less. zlib on the tensor gets it
to 20.2 KB, still not decisive. **Defensible claims instead:** compute offload, privacy
(the wire carries a tensor, not a viewable image), and no double degradation.

### 2. The head still does 85% of the work

Measured from the ONNX graphs at 320px:

| artifact | GMACs | % of full |
|---|---|---|
| full `best_320.onnx` | 10.768 | 100% |
| `backbone_enc_320.onnx` (Pi Zero) | 2.567 | **23.8%** |
| `dec_head_320.onnx` (Pi 3) | 9.157 | **85.0%** |

Layer 3 is early, and the decoder adds ~9% back. If Pi Zero 2W and Pi 3 have comparable
cores (both 4x A53), pipelined throughput may improve only ~1.15x, and single-frame
latency could be *worse* than sending a JPEG and running everything on the Pi 3.

MACs are not latency — early layers are memory-bandwidth-bound and often underperform
their MAC share, and the 98 MB head may not fit a Pi Zero-class memory budget at all.
**Pi measurements are the decider.**

**The design knob if Pi numbers disappoint:** the bottleneck compresses to 16 channels
*regardless of cut depth*, which decouples wire size from split depth. Layer 3 was chosen
to minimise payload, but the bottleneck already solves payload. A deeper cut would offload
much more compute at the *same or smaller* wire cost (16ch at 20x20 = 6.4 KB). The catch:
deeper cuts cross the P3 concat skip and would need to ship more than one tensor.

### 3. Recall drops 7.8 points

Precision rises (+0.072) while recall falls (-0.078) — the bottleneck acts as a confidence
filter that strips marginal detections. **For leopard monitoring this is the wrong
direction** (a missed leopard is worse than a false alarm). Cheap fix: lower the deployment
confidence threshold, trading back some of the precision gain.

Note this trade *flips direction* between versions — v2 lost precision and held recall.
It is configuration-specific, not inherent to the approach.

### 4. `models/bottleneck/` is gitignored

`.gitignore:222`. The trained `bneck_16ch.pt` will not be committable where it lands.
The 2.3 GB feature cache *should* stay ignored, but the checkpoint needs a negation rule.

---

## Artifacts

```
models/split/
├── backbone_320.onnx        3.4 MB    plain split, verified exact
├── head_320.onnx           95.9 MB
├── backbone_enc_320.onnx    3.4 MB    <- Pi Zero: images -> bottleneck (1,16,40,40)
├── dec_head_320.onnx       98.2 MB    <- Pi 3: bottleneck -> output0 (1,300,6)
├── bottleneck_320.json                scale/zero_point for the wire
└── _bottleneck_parts/                 intermediate encoder.onnx / decoder.onnx
models/bottleneck/
├── bneck_16ch.pt                      trained weights (gitignored)
└── feats_320.npy           2.3 GB     feature cache (gitignored, regenerable)
results/benchmarks/
├── accuracy_three_way_v3.csv
└── accuracy_three_way_v2.csv
```

**Nothing is committed.** `git status` shows the restored `scripts/bottleneck/` (staged),
modified `create_split.py` / `backbone_server.py` / `head_client.py`, and untracked
`models/split/` and the two CSVs.

---

## Reproducing the Pipeline

```bash
python scripts/create_split.py models/full/best_320.onnx models/split 320
python scripts/verify_split.py models/full/best_320.onnx models/split/backbone_320.onnx models/split/head_320.onnx data/images 320
python scripts/bottleneck/cache_features.py --imgsz 320 --limit 3000
python scripts/bottleneck/train_bottleneck.py --channels 16 --epochs 40
python scripts/bottleneck/export_bottleneck.py --ckpt models/bottleneck/bneck_16ch.pt
python scripts/bottleneck/verify_bottleneck.py data/images --imgsz 320
python scripts/bottleneck/eval_mAP_three_way.py --imgsz 320
```

On Windows, prefix with `KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8` — the OpenMP
duplicate-runtime error and the cp1252 console codec both bite otherwise.

Timings on this laptop (CPU): cache ~15 min, train ~65 min, three-way eval ~45 min.

---

## Next Session

1. **Deploy** — `backbone_enc_320.onnx` to Pi Zero, `dec_head_320.onnx` to Pi 3. The uint8
   wire path is restored in `scripts/backbone_server.py` (`--quant models/split/bottleneck_320.json`)
   and `scripts/head_client.py`.
2. **Measure real latency** — this settles open issue #2, which is the project's core claim.
3. **Baseline** — full model on the Pi alone, for the "split inference is necessary" story.
4. **Live demo wiring** — recommendation was to send detections *back* to the Pi Zero
   (<1 KB) and draw there, rather than uplinking a JPEG. Keeps the uplink bottleneck-only
   and avoids the "why not just send the image" objection. Not yet implemented.
5. **Report** — mind the four open issues above.
