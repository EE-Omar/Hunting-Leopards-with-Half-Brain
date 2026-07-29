# START HERE - Run These In Order

| Order | File | What It Does | Run Where | Command |
|-------|------|-------------|-----------|---------|
| **1** | `create_split_model.py` | Splits best_640.onnx into backbone + head | Pi or Laptop | `python3 create_split_model.py models/best_640.onnx models/split_640 640` |
| **2** | `verify_split_new.py` | Checks if split works correctly (100% match?) | Pi or Laptop | `python3 verify_split_new.py models/best_640.onnx models/split_640/backbone_640.onnx models/split_640/head_640.onnx images/ 640` |
| **3** | `laptop_head_client.py` | Runs head inference locally (test first) | Laptop | `python3 laptop_head_client.py models/split_640/head_640.onnx --test-images images/` |
| **4a** | `laptop_head_client.py` | Starts head server (waits for Pi) | Laptop | `python3 laptop_head_client.py models/split_640/head_640.onnx --port 5000 --imgsz 640` |
| **4b** | `pi_backbone_server.py` | Runs backbone, sends tensors to laptop | Pi | `python3 pi_backbone_server.py models/split_640/backbone_640.onnx <LAPTOP_IP> 5000 640 --images images/` |

---

## What Each Does (One Line)

- **create_split_model.py** - Splits model into two parts
- **verify_split_new.py** - Makes sure split is correct
- **laptop_head_client.py** - Runs the head part, outputs detections
- **pi_backbone_server.py** - Runs the backbone part, sends tensor to laptop
- **inspect_model.py** - Debug: see inside the ONNX model
- **profile_latency.py** - Debug: measure how fast each part is
- **README.md** - Background info (read if confused)
- **QUICK_REFERENCE.md** - Copy-paste commands for everything
- **SPLIT_INFERENCE_GUIDE.md** - Detailed troubleshooting

---

## Actual Quick Start

```bash
# Step 1: Split (run once on Pi or laptop with models/)
cd ~/Second_Try
python3 create_split_model.py models/best_640.onnx models/split_640 640

# Step 2: Verify
python3 verify_split_new.py models/best_640.onnx models/split_640/backbone_640.onnx models/split_640/head_640.onnx images/ 640

# Step 3: Test locally first (see if it works)
python3 laptop_head_client.py models/split_640/head_640.onnx --test-images images/

# Step 4a: When ready for network - Terminal 1 (Laptop)
python3 laptop_head_client.py models/split_640/head_640.onnx --port 5000 --imgsz 640

# Step 4b: Terminal 2 (Pi) - replace IP with laptop's
python3 pi_backbone_server.py models/split_640/backbone_640.onnx 100.118.245.12 5000 640 --images images/
```

That's it. Everything else is optional.
