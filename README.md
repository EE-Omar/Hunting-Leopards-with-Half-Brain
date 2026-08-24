# Hunting Leopards with Half a Brain

Split-inference detection of the Arabian leopard on off-grid edge hardware.

A single YOLO26-Large is cut in two. The first three layers run on a Raspberry Pi Zero 2 W in
the field; the remaining layers and the detection head run on one shared Raspberry Pi 3 nearby.
A learned encoder/decoder pair straddles the cut, so what crosses the wireless link is 25 KB per
frame instead of 1,600 KB. The result is a large model running on hardware that cannot hold it,
at a 0.0197 mAP50-95 accuracy drop.

---

## Team

| | |
|---|---|
| Omar Alharbi | omar.alharbi@kaust.edu.sa |
| Abdullah Alhindi | abdullah.alhindi@kaust.edu.sa |
| Abdullah Alghanim | abdullah.alghanim@kaust.edu.sa |
| Rasheed Hamidaddin | rasheed.hamidaddin@kaust.edu.sa |
| Basil Alshareef | basilahmed.alshareef@gmail.com |
| Basil Alshehri | basil.alshehri55@gmail.com |

---

## Results

Layer-3 split, 320x320, Pi Zero 2 W to Pi 3 over a hotspot LAN. Accuracy is measured on 1,212
validation images through the deployed ONNX artifacts, including the real uint8 wire round-trip.

| | Baseline (whole model on Pi Zero) | Split only | Split + learned bottleneck |
|---|---|---|---|
| Payload per frame | — | 800.3 KB (fp16) | **25.26 KB** (uint8) |
| Transmission time | — | 522.5 ms | **73.0 ms** |
| mAP50-95 | 0.6535 | 0.6535 | 0.6338 |
| mAP50 | 0.8323 | 0.8323 | 0.8121 |
| Leopard AP50-95 | 0.6267 | 0.6267 | 0.6156 |
| Throughput | 0.380 fps | 0.346 fps | 0.386 fps |

**Accuracy drop from the bottleneck:** 0.0197 mAP50-95 across all eight classes (3.0% relative),
and 0.0111 on the leopard alone. The split without the bottleneck is bit-exact: 4528/4528
predictions match the unmodified model.

**Compression:** 64x fewer bytes on the wire (1,600 KB fp32 to 25 KB uint8), 7.2x faster
transmission. The encoder narrows the layer-3 feature map from 256 channels to 16 and costs
~4,000 parameters on the camera.

**Reconstruction vs. accuracy:** the decoder's output differs from the original feature map by
41% relative L2 error, yet mAP moves by 0.0197. The discarded information was not information the
detection head was using.

**Pipelining:** overlapping camera and server so two frames are in flight raises the split arm
from 0.329 to 0.499 fps, 52% faster on identical hardware (mean of 5 runs each, fp16 wire).

Source data: [`results/benchmarks/comprehensive_comparison_v3.csv`](results/benchmarks/comprehensive_comparison_v3.csv),
figures in [`results/charts/`](results/charts/).

---

## Architecture

```
[Camera]  Raspberry Pi Zero 2 W          |          Raspberry Pi 3  (shared)
   |                                     |
   +-- preprocess (320x320)              |
   +-- YOLO26-L layers 0-3               |    +-- decoder  16 -> 256 ch
   |     3.4% of parameters              |    +-- YOLO26-L layers 4-23 + head
   +-- encoder  256 -> 16 ch  ~4K params |    +-- decode detections
   +-- quantize uint8 ------------------>|
         25 KB/frame over Wi-Fi          |          [x1 y1 x2 y2 conf class]
```

Adding coverage costs one more camera, not one more server. If the link drops, the camera falls
back to a smaller model it runs alone.

**Classes:** `0 leopard` (target), `1 cheetah`, `2 hyena`, `3 nubian_ibex`, `4 camel`, `5 cat`,
`6 dog`, `7 person`. The seven non-target classes share the leopard's habitat and teach the model
what a leopard is not.

---

## Repo layout

```
run_backbone.py / run_head.py     entry points for the two devices
config_backbone.yaml / config_head.yaml

models/
  full/     best_v3.pt, best_320.onnx, best_640.onnx    unmodified YOLO26-L
  split/    backbone_320.onnx, head_320.onnx            layer-3 split, no bottleneck
            backbone_enc_320.onnx                       camera side, with encoder
            bottleneck_320.json                         uint8 scale / zero-point
            (dec_head_320.onnx is 98 MB and NOT committed --
             rebuild it with export_bottleneck.py, step 5 below)
  bottleneck/bneck_16ch.pt                              trained encoder/decoder

scripts/
  create_split.py         split a full ONNX into backbone + head
  verify_split.py         check the split is numerically identical to the full model
  backbone_server.py      camera-side inference + TCP stream
  head_client.py          server-side receive, inference, decode
  bottleneck/             cache features, train, export, evaluate, plot

results/
  benchmarks/   accuracy and latency CSVs
  charts/       figures
  training/v3/  PR curves, confusion matrices, results.csv

notebooks/v3_train.ipynb  training
```

---

## Setup

```bash
pip install onnxruntime opencv-python numpy pyyaml ultralytics torch gdown tqdm
```

```bash
python download_data.py
```

The scripts print status with `[OK]`-style Unicode marks. On a Windows console this raises
`UnicodeEncodeError`, so set the encoding first (Linux and the Pis need nothing):

```bash
set PYTHONIOENCODING=utf-8
```

---

## Reproducing the pipeline

Every step below writes into `models/` or `results/` and can be run from the repo root. The
committed artifacts are the output of exactly these commands.

**1. Split the full model at layer 3**

```bash
python scripts/create_split.py models/full/best_320.onnx models/split 320
```

**2. Verify the split is numerically identical**

```bash
python scripts/verify_split.py models/full/best_320.onnx models/split/backbone_320.onnx models/split/head_320.onnx data/v2_yolo_dataset/valid/images 320
```

**3. Cache layer-3 activations for bottleneck training**

```bash
python scripts/bottleneck/cache_features.py --imgsz 320 --limit 3000
```

**4. Train the 16-channel bottleneck**

```bash
python scripts/bottleneck/train_bottleneck.py --channels 16 --epochs 30
```

**5. Export it onto the split artifacts**

```bash
python scripts/bottleneck/export_bottleneck.py --ckpt models/bottleneck/bneck_16ch.pt
```

**6. Three-way accuracy comparison on the deployed ONNX**

```bash
python scripts/bottleneck/eval_mAP_three_way.py
```

**7. Build the comparison table and figures**

```bash
python scripts/bottleneck/build_comparison.py
```

```bash
python scripts/bottleneck/plot_comprehensive.py
```

---

## Running the two devices

Start the head first; it waits for the connection.

```bash
python run_head.py
```

```bash
python run_backbone.py
```

Set `head_ip` in `config_backbone.yaml` to the server's address, and keep `imgsz` and the model
paths matching in both config files.

---

## Not in this repo

`data/` and `venv/` are gitignored. Latency measurements on the Pis were collected with a
separate benchmarking harness; the CSVs it produced are committed under `results/benchmarks/`.
