# Hunting Leopards with Half a Brain

> Real-time Arabian Leopard detection on constrained edge hardware using split neural network inference.

## Table of Contents
- [The Problem](#the-problem)
- [Our Approach](#our-approach)
- [How It Works](#how-it-works)
- [Model](#model)
- [Repo Structure](#repo-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running](#running)
- [Benchmarks](#benchmarks)
- [Debug Tools](#debug-tools)
- [Backlog](#backlog)

---

## [The Problem](#table-of-contents)

The Arabian Leopard is one of the most endangered big cats in the world, with fewer than 200 individuals remaining in the wild. Monitoring them requires deploying detection systems in remote, off-grid locations — places with no internet, no power grid, and extreme heat.

This rules out cloud-based AI entirely. Everything has to run locally on battery-powered edge hardware.

The challenge: a full YOLO model won't run in real-time on a single low-power device. A Raspberry Pi Zero — the kind of hardware you can realistically deploy in the wild — doesn't have enough compute to run inference fast enough to be useful.

---

## [Our Approach](#table-of-contents)

We implement **split inference**: the neural network is divided into two parts that run on two separate devices simultaneously.

- The **backbone device** (low-power, field-deployed) handles early-layer feature extraction
- The **head device** (more powerful, nearby) handles the rest of the network and outputs detections
- Only a small intermediate tensor (~0.52 MB) is sent between them — not raw video frames

This lets us deploy the backbone on ultra-low-power hardware while offloading the heavy computation to a slightly more capable device nearby.

---

## [How It Works](#table-of-contents)

```
[Camera]
   ↓
[Backbone Device]
   - Preprocesses frame (resize, normalize)
   - Runs backbone layers (Layer 0 → split point)
   - Sends intermediate tensor over TCP
   ↓ (~0.52 MB per frame over Tailscale)
[Head Device]
   - Receives tensor
   - Runs head layers (split point → output)
   - Decodes detections
   - Filters for target class (leopard)
   ↓
[Detections: x1, y1, x2, y2, confidence, class]
```

### [Split Points](#table-of-contect)

The model can be split at different layers depending on hardware constraints and network bandwidth. We tested two split points:

| Split Point | Tensor Shape | Tensor Size | Notes |
|-------------|-------------|-------------|-------|
| Layer 3 | `(1, 64, ?, ?)` | ~0.26 MB | Less backbone compute, more head compute |
| Layer 4 | `(1, 128, ?, ?)` | ~0.52 MB | Best balance — currently used |

The split layer is determined by the model files you use. Currently tested configurations:

| Resolution | Split | Backbone file | Head file |
|------------|-------|--------------|-----------|
| 256×256 | Layer 4 | `backbone_256.onnx` | `head_256.onnx` |
| 320×320 | Layer 3 or 4 | `backbone_320.onnx` | `head_320.onnx` |
| 640×640 | Layer 4 | `backbone_640.onnx` | `head_640.onnx` |

No raw images are transmitted — only the intermediate tensor.

---

## [Model](#table-of-contect)

- **Architecture:** YOLO26 Nano (End-to-End, One-to-One Head)
- **Training framework:** Ultralytics
- **Export format:** ONNX (unquantized FP32)
- **Output shape:** `(1, 300, 6)` → `[x1, y1, x2, y2, confidence, class_id]`
- **No NMS needed** — the One-to-One Head outputs at most 300 detections directly

> ⚠️ INT8 quantization via Ultralytics export is currently broken for this model — use unquantized ONNX only.

### Classes

| ID | Class |
|----|-------|
| 0 | leopard ← **target** |
| 1 | cheetah |
| 2 | hyena |
| 3 | nubian_ibex |
| 4 | camel |
| 5 | cat |
| 6 | dog |
| 7 | person |

---

## [Repo Structure](#table-of-contect)

```
project/
├── run_backbone.py          ← entry point (backbone device)
├── run_head.py              ← entry point (head device)
├── config_backbone.yaml     ← backbone device config
├── config_head.yaml         ← head device config
│
├── models/
│   ├── full/                ← full unquantized ONNX models
│   ├── split/               ← split backbone + head models
│   ├── ncnn/                ← ncnn exports (attempted, see results)
│   └── v1/                  ← v1 single-class baseline
│
├── notebooks/               ← training notebooks (v1, v2)
│
├── results/
│   ├── benchmarks/          ← latency reports
│   ├── charts/              ← benchmark graphs
│   ├── detections/          ← sample detection outputs
│   └── training/
│       └── v2/              ← confusion matrices, PR curves, training plots
│
└── scripts/
    ├── backbone_server.py   ← backbone inference + TCP stream
    ├── head_client.py       ← receive tensor, head inference, decode
    ├── create_split.py      ← split a full ONNX model into backbone + head
    ├── verify_split.py      ← verify split is mathematically identical to full model
    ├── draw_detections.py   ← draw bounding boxes on test images
    ├── inspect_model.py     ← inspect ONNX model structure and tensor names
    └── profile_latency.py   ← measure per-component inference latency
```

---

## [Setup](#table-of-contents)

### Requirements

```bash
pip install onnxruntime opencv-python numpy pyyaml ultralytics
```

### Generate split models (run once)

```bash
# 256×256
python scripts/create_split.py models/full/best_256.onnx models/split 256

# 320×320
python scripts/create_split.py models/full/best_320.onnx models/split 320

# 640×640
python scripts/create_split.py models/full/best_640.onnx models/split 640
```

### Verify the split

```bash
python scripts/verify_split.py models/full/best_256.onnx \
    models/split/backbone_256.onnx \
    models/split/head_256.onnx \
    data/images/ 256
```

Expected output: `PASS: N, FAIL: 0`

---

## [Configuration](#table-of-contents)

### `config_backbone.yaml` — backbone device

```yaml
network:
  head_ip: <HEAD_DEVICE_TAILSCALE_IP>
  port: 5000

model:
  backbone: models/split/backbone_256.onnx
  imgsz: 256                  # must match head config

source:
  type: images                # images | camera
  images_path: data/images/
  camera_device: 0
  fps: 2
```

### `config_head.yaml` — head device

```yaml
network:
  port: 5000

model:
  head: models/split/head_256.onnx
  imgsz: 256                  # must match backbone config

detection:
  conf_threshold: 0.5
  target_class: 0             # 0=leopard  |  -1=all classes

output:
  save: false                 # save annotated images to results/detections/
  display: false              # show live window (requires display)
```

**To change resolution:** update `imgsz` and model paths in both configs to match (e.g. both to 320).

---

## [Running](#table-of-contents)

```bash
# 1. Start head device first (it waits for connection)
python run_head.py

# 2. Start backbone device
python run_backbone.py
```

To test locally without a second device:

```bash
python run_head.py --test-images data/images/
```

---

## [Benchmarks](#table-of-contents)

Full latency data in `results/benchmarks/`. Summary:

| Resolution | Backbone | Send | Head | Total | FPS | Bottleneck |
|------------|---------|------|------|-------|-----|-----------|
| 640×640 | 718 ms | 4,273 ms | 218 ms | ~5,200 ms | 0.19 | Network (82%) |
| 256×256 | 134 ms | 55 ms | 58 ms | ~246 ms | ~2.0 | Backbone (54%) |

At 640×640 the network dominates. At 256×256 the bottleneck shifts to backbone compute. See `results/charts/` for visual breakdown.

---

## Debug Tools

```bash
# Draw detections on an image (full model)
python scripts/draw_detections.py models/full/best_256.onnx "data/images/leopard.jpg" --save

# Draw detections (split model)
python scripts/draw_detections.py models/split/head_256.onnx "data/images/leopard.jpg" \
    --backbone models/split/backbone_256.onnx --save

# Inspect model structure
python scripts/inspect_model.py models/full/best_256.onnx

# Profile latency breakdown
python scripts/profile_latency.py models/full/best_256.onnx data/images/ 256
```

---

## [What's Not in This Repo](#table-of-contents)

| Path | Reason |
|------|--------|
| `data/` | Datasets are gitignored. Download script coming soon. |
| `venv/` | Python environment |
| `notebooks/runs/` | Training run outputs |

---

## Backlog

- [ ] Data download script (`setup_data.py`)
- [ ] Live camera testing end-to-end
- [ ] Compress tensor before TCP (Bottleneck)
- [ ] Retrain with balanced dataset (current: ~9K leopard vs ~300-400 other classes)
- [ ] YOLO26-small + FP16 quantization
- [ ] Systemd service for auto-start on boot
