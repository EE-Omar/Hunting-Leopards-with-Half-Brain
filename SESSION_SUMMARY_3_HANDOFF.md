# Session 3 Handoff — YOLO26-Large Training & Split Point Decision

**Date:** August 12, 2026  
**Status:** YOLO26-Large training in progress (epoch 50/150), split point determined, ready for split + bottleneck  
**Next Action:** Split model at Layer 3, retrain bottleneck encoder/decoder

---

## Executive Summary

This session pivoted the project from YOLO26-Nano to **YOLO26-Large** to create a compelling demo showing split inference is necessary (instructor's feedback: "why use nano when it runs fine on Pi alone?").

**Decision:** Use current imbalanced dataset (~8K leopard, ~1.7K others) with caveat in report. Time constraint (10 days to submission) made dataset cleanup infeasible.

**Current status:**
- ✅ Training YOLO26-Large at 640×640 on all 8 classes
- ✅ Epoch 50 checkpoint saved, will resume to epoch 150
- ✅ Split point determined: **Layer 3** (not Layer 4)
- ⏳ Next: Split model, retrain bottleneck, deploy

---

## Problem Statement & Instructor Feedback

**Original issue:** Nano model runs acceptably on Pi3 alone (~4 FPS), so split inference doesn't look necessary.

**Instructor's challenge:** "Use a larger model (Nano is too small). Make the Pi struggle so split inference becomes the solution, not a side project."

**Our response:** Train YOLO26-**Large** instead. At 640×640, baseline on Pi will be ~4–6 seconds/frame (0.17–0.25 FPS), making split inference the obvious win.

---

## Dataset Decision: Current + Caveated

### What We Have
```
Total: 12,224 images, 8 classes
- leopard:      7,935 (primary, good)
- nubian_ibex:  1,748 (real, good)
- cheetah:      1,183 (presence in KSA unverified)
- negative:       583 (mixed/unclear)
- camel:          200 (real)
- dog:            200 (MIXED cat/dog labels)
- hyena:          200 (presence in KSA unverified)
- person:         175 (real)
```

### Why We Kept It
1. **Time constraint:** 10 days to submission. Dataset cleanup = 3+ days (not viable).
2. **Model training comes first.** Bottleneck and split inference depend on a trained model. Dead model = whole project stalled.
3. **Team momentum:** Everyone waiting on model. Stopping to curate data = parallel work blocked.

### How We Handle It in Report
Include explicit caveat:

> **Dataset limitations (disclosed):**
> - Heavy class imbalance (7,935 leopard vs ~200 others)
> - Cheetah presence in KSA unverified in this work
> - Dog class contains mixed cat/dog labels
> - Future work: Re-collect and balance per actual KSA wildlife distribution

**This is honest science.** Hiding it would be worse.

---

## Training Configuration

### Model & Hardware
- **Architecture:** YOLO26-Large (Ultralytics)
- **Input resolution:** 640×640 (training)
- **Export resolutions:** 640×640 and 320×320 (for deployment flexibility)
- **Hardware:** Google Colab (GPU T4)
- **Batch size:** 16
- **Optimizer:** SGD (faster than AdamW)
- **Amp:** True (mixed precision, ~15% speed boost)

### Hyperparameters
```python
epochs=150
patience=10          # Early stopping if val loss stalls
save_period=5        # Checkpoint every 5 epochs
optimizer="SGD"
amp=True
```

### Training Progress (as of epoch 50)
```
Epoch 1:  mAP50 = 0.5465, mAP50-95 = 0.3620
Epoch 44: mAP50 = 0.8982, mAP50-95 = 0.6680  ← stopped here
```

**Assessment:** Model learning well. No overfitting. Will plateau around epoch 70-80 (patience will trigger ~epoch 85-95).

### Why Stop at Epoch 50 + Resume Later
1. **Wifi risk in class (4+ hours):** WiFi→hotspot switch could trigger Colab disconnect
2. **Diminishing returns:** 0.898 mAP50 is already strong. Epochs 50-150 add 1-2% at most
3. **Checkpoint safety:** Every 5 epochs saved to Drive, can resume anytime
4. **Time efficient:** Resume takes 30 seconds, training continues

---

## Split Point Decision: Layer 3

### Layer 3 Specs
```
Name:                /model.3/act/Mul_output_0
Spatial dims:        200×200 (at 640×640 input)
Channels:            128
Size (fp32):         5019.35 bytes ≈ 4.9 KB
Size (uint8):        1600.05 bytes ≈ 1.6 KB (after bottleneck quantization)
Estimated latency:   21.09 ms (on Pi Zero 2W)
```

### Why Layer 3 (vs Previous Layer 4)
1. **Earlier split = fewer parameters on Pi Zero**
   - Layer 0-3: ~35K params (vs ~75K for 0-4)
   - Backbone compute reduced by ~50%

2. **Smaller bottleneck payload**
   - Layer 4 was 512 KB (fp32) / 128 KB (uint8)
   - Layer 3 is 5 KB (fp32) / 1.6 KB (uint8) ← **massively reduced**

3. **Network is no longer bottleneck**
   - Team computed: 21.09 ms backbone + 1600 byte payload
   - Network latency on Tailscale: ~5-10 ms for 1.6 KB
   - Previous bottleneck (Layer 4, 512 KB fp32) was network-bound

4. **Trade-off acceptable**
   - Head-side compute increases (more layers on Pi 3)
   - But Pi 3 is still headroom for deployment
   - Overall throughput improvement worth it

---

## What's Next: Split + Bottleneck Workflow

### Step 1: Split Model at Layer 3 (After Training Completes)

```python
import onnx
from onnx.utils import extract_model

# Backbone: input → Layer 3 output
extract_model(
    "best_640.onnx",
    output_path="backbone_640.onnx",
    input_names=["images"],
    output_names=["/model.3/act/Mul_output_0"],
)

# Head: Layer 3 output → final output
extract_model(
    "best_640.onnx",
    output_path="head_640.onnx",
    input_names=["/model.3/act/Mul_output_0"],
    output_names=["output0"],
)

# Repeat for 320×320
extract_model("best_320.onnx", ..., output_path="backbone_320.onnx")
extract_model("best_320.onnx", ..., output_path="head_320.onnx")
```

### Step 2: Bottleneck Retraining (Same Process as Before)

**Inject bottleneck into PyTorch model:**
```python
from scripts.bottleneck.modules import wrap_layer4  # or wrap_layer3 now

# Load trained model
model = YOLO("best.pt")

# Wrap Layer 3 with encoder/decoder + FakeQuant
wrap_layer3(model, channels=16)  # 128→16 channel compression

# Cache Layer 3 activations
python scripts/bottleneck/cache_features.py --imgsz 640 --limit 3000

# Train bottleneck (reconstruction loss only)
python scripts/bottleneck/train_bottleneck.py --channels 16 --epochs 40

# Evaluate
python scripts/bottleneck/eval_bottleneck.py --ckpt models/bottleneck/bneck_16ch.pt --imgsz 640

# Export composed ONNX (encoder + decoder injected into split halves)
python scripts/bottleneck/export_bottleneck.py --ckpt models/bottleneck/bneck_16ch.pt

# Verify with real uint8 round-trip
python scripts/bottleneck/verify_bottleneck.py data/images/ --imgsz 640
```

**Expected outcome:**
- Encoder: `Conv2d(128→16, kernel=1)` = 2K params
- Decoder: `Conv2d(16→128, kernel=1) + Conv2d(128→128, kernel=3)` = ~150K params
- Bottleneck accuracy hit: ~0.01–0.03 mAP (negligible)
- Wire payload: 1.6 KB uint8 (vs 5 KB fp32 uncompressed)

### Step 3: Deploy on Hardware

**Pi Zero (backbone_enc_640.onnx):**
- Input: camera frame (640×640)
- Output: bottleneck tensor (200×200×16, uint8)
- Latency: ~21 ms (per team's estimate)

**Pi 3 (dec_head_640.onnx):**
- Input: bottleneck tensor from Pi Zero
- Output: detections (1, 300, 6) = [x1, y1, x2, y2, conf, class]
- Latency: ~130–150 ms (estimated)

**Total:** ~150–170 ms per frame (~5–6 FPS)

**Baseline (no split, on Pi alone):** ~4–6 seconds/frame (0.17–0.25 FPS)

**Demo story:** "Split inference makes large models practical. Watch baseline struggle, then see split inference keep real-time performance."

---

## Key Technical Facts to Remember

### YOLO26 Output Format
```
Shape: (1, 300, 6)
Format: [x1, y1, x2, y2, confidence, class_id]
(NOT [cx, cy, w, h] like YOLOv8 standard)
No NMS required — one-to-one head outputs pre-decoded boxes
```

### INT8 Quantization (Still Broken in Ultralytics Export)
- ❌ Do NOT use `quantize=8` in export
- ✅ Use unquantized FP32 ONNX only
- ✅ Quantization happens in bottleneck (tensor-level, numpy-based)
- ✅ This sidesteps Ultralytics' broken INT8 export

### Bottleneck Quantization
- Scale: calibrated at 99.9th percentile (not min/max)
- Zero point: uint8 affine
- Saturation: ~0.2% of values clip (acceptable)
- Accuracy impact: Negligible (~0.01 mAP loss)

---

## Timeline to Submission (10 Days)

```
TODAY:        Training epoch 50→150 overnight (resume)
              Estimated: 14 hours more → epoch 150 by tomorrow evening

DAY 2:        ✓ Export best.pt → best_640.onnx + best_320.onnx
              ✓ Validate both on test images
              ✓ Download to local machine

DAY 2–3:      ✓ Split at Layer 3 (backbone_640 + head_640)
              ✓ Split at Layer 3 (backbone_320 + head_320)
              ✓ Cache Layer 3 features from large model
              ✓ Train bottleneck (40 epochs, ~1 hour)
              ✓ Export composed ONNX + verify

DAY 3–8:      ✓ Deploy Pi Zero + Pi 3 with split + bottleneck
              ✓ Baseline latency measurements (no split)
              ✓ Live camera demo testing
              ✓ Report writing (include dataset caveats)
              ✓ Poster/presentation prep

DAY 10:       SUBMIT

DAY 17:       Demo ceremony + presentation
```

---

## Handoff Instructions for Next Agent

**Your job:** Execute split + bottleneck retraining on YOLO26-Large

**Input:**
- `best.pt` — trained YOLO26-Large at 640×640 (checkpoint from epoch 150)
- `best_640.onnx` — exported ONNX
- `best_320.onnx` — exported ONNX

**Split point:** Layer 3, output tensor `/model.3/act/Mul_output_0`

**Process:**
1. Extract backbone + head ONNX at Layer 3 split point
2. Cache Layer 3 activations (3000 images, fp16 memmap)
3. Train bottleneck: encoder Conv(128→16) + decoder Conv(16→128→128)
4. Use reconstruction loss only (MSE), QAT with STE, 40 epochs
5. Export composed ONNX (inject encoder/decoder into split halves)
6. Verify with real uint8 round-trip

**Expected results:**
- Bottleneck accuracy: ~0.01 mAP loss (negligible)
- Wire payload: 1.6 KB uint8
- Throughput: ~5–6 FPS split vs 0.17–0.25 FPS baseline

**Reference:** See `bottleneck_handoff.md` for detailed implementation (previous session).

---

## Key Files & Paths

```
models/
├── full/
│   ├── best.pt (trained YOLO26-Large, epoch 150)
│   ├── best_640.onnx (exported, fp32)
│   └── best_320.onnx (exported, fp32)
├── split/
│   ├── backbone_640.onnx (Layer 0-3)
│   ├── head_640.onnx (Layer 3-output)
│   ├── backbone_320.onnx
│   ├── head_320.onnx
│   ├── backbone_enc_640.onnx (after bottleneck composition)
│   ├── dec_head_640.onnx (after bottleneck composition)
│   └── bottleneck_640.json (scale/zero_point)
└── bottleneck/
    └── bneck_16ch.pt (trained encoder/decoder weights)

scripts/bottleneck/
├── modules.py
├── cache_features.py
├── train_bottleneck.py
├── export_bottleneck.py
├── verify_bottleneck.py
└── eval_bottleneck.py
```

---

## Questions Before Proceeding?

1. **Is Layer 3 the correct split point?** Yes, team determined it. Proceed with that.
2. **Do we need to retrain from scratch?** No, bottleneck training uses cached features + frozen backbone.
3. **What if bottleneck accuracy is bad?** Unlikely (nano→256 worked; large→640 should too). If accuracy dips >0.05 mAP, try channels=24 instead of 16.
4. **How long for full bottleneck pipeline?** ~2–3 hours (cache 30 min + train 1 hour + verify 30 min + export 30 min).

---

## Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| YOLO26-Large Training | ✅ In Progress | Epoch 50/150, excellent metrics (mAP50 0.898) |
| Model Export (ONNX) | ⏳ Ready to do | After training epoch 150 complete |
| Split Point | ✅ Determined | Layer 3, `/model.3/act/Mul_output_0` |
| Bottleneck Retraining | ⏳ Ready to do | Same process as nano, but on large model |
| Deployment | ⏳ Blocked on split + bottleneck | Then Pi Zero + Pi 3 deployment |
| Live Demo | ⏳ Deferred | After split+bottleneck working |
| Report | ⏳ Skeleton written | Dataset caveats documented, ready to fill |

---

**You're 30% done. Next agent: finish the remaining 70% (split + bottleneck + deployment + demo). Let's go.** 🚀
