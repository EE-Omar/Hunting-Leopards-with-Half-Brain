# YOLO26 Split Inference - Quick Reference

## One-Shot Setup (Copy-Paste)

### On Pi3 or any machine with models/

```bash
cd ~/Second_Try

# 1. Split the model (once)
python3 create_split_model.py models/best_640.onnx models/split_640 640

# 2. Verify it worked
python3 verify_split_new.py models/best_640.onnx models/split_640/backbone_640.onnx models/split_640/head_640.onnx images/ 640
```

Expected output: `PASS: 136, FAIL: 0`

---

## Running Split Inference

### Terminal 1: Laptop (Head Server)

```bash
cd ~/Second_Try
python3 laptop_head_client.py models/split_640/head_640.onnx --port 5000 --imgsz 640
```

**Or test locally (no Pi needed):**
```bash
python3 laptop_head_client.py models/split_640/head_640.onnx --port 5000 --imgsz 640 --test-images images/
```

### Terminal 2: Pi3 (Backbone Server)

First, get laptop IP:
```bash
# On laptop
tailscale status | grep -i laptop
# Example: 100.118.245.12
```

Then on Pi:
```bash
cd ~/Second_Try

# With images (reproducible)
python3 pi_backbone_server.py models/split_640/backbone_640.onnx 100.118.245.12 5000 640 --images images/ --fps 2

# With camera
python3 pi_backbone_server.py models/split_640/backbone_640.onnx 100.118.245.12 5000 640 --camera 0
```

Replace `100.118.245.12` with actual laptop IP.

---

## Inspecting Models

### View model structure
```bash
python3 inspect_model.py models/best_640.onnx
```

### Verbose (show every layer)
```bash
python3 inspect_model.py models/best_640.onnx --verbose
```

### Visualize with Netron
```bash
pip install netron
netron models/best_640.onnx  # Opens browser, shows interactive diagram
```

---

## Common Issues

### "Connection refused"
**Problem:** Pi can't reach laptop
- [ ] Check laptop Tailscale IP is correct
- [ ] Check laptop server is running (Terminal 1)
- [ ] Check firewall: `sudo ufw status`

### "No images found"
**Problem:** Test images not in right folder
```bash
# Copy images to project
mkdir -p ~/Second_Try/images
cp ~/your/leopard/images/* ~/Second_Try/images/
```

### "FAIL" from verify_split_new.py
**Problem:** Split tensors don't match
- [ ] Are you using `best_640.onnx` (NOT INT8)?
- [ ] Are both backbone and head models present?
- [ ] Check `inspect_model.py` output for layer 4 name

### Very low FPS on Pi3
**Problem:** Backbone inference taking >300ms
- [ ] Switch to 256x256: use `best_256.onnx` instead
- [ ] Check if camera input is bottleneck (try `--images`)
- [ ] Monitor Pi CPU: `top` (should be ~100% CPU usage)

### Tensor size huge (~2.1 MB)
**Normal!** Backbone output (1, 128, 32, 32) float32 = 2,097,152 bytes
- Can optimize later with quantization

---

## Performance Targets

### Latency (640x640, no optimization)
```
Full model on Pi:     ~250 ms/frame
Split inference:      ~250 ms/frame (similar now)
                      ↓ (optimize backbone → ~180 ms)
```

### Latency (256x256, no optimization)
```
Full model on Pi:     ~60-80 ms/frame
Split inference:      ~60-80 ms/frame
                      ↓ (optimize backbone → ~40-50 ms)
```

### Throughput
```
640x640: ~4 FPS (limited by Pi backbone)
256x256: ~12-15 FPS (limited by Pi backbone)
Laptop alone: 30+ FPS (head is fast)
```

---

## File Locations

After split:
```
~/Second_Try/
├── models/
│   ├── best_640.onnx            (full model)
│   ├── best_256.onnx            (full model)
│   ├── split_640/
│   │   ├── backbone_640.onnx    ← Deploy to Pi
│   │   └── head_640.onnx        ← Deploy to Laptop
│   └── split_256/
│       ├── backbone_256.onnx
│       └── head_256.onnx
├── images/                       (test images)
├── create_split_model.py
├── verify_split_new.py
├── pi_backbone_server.py
├── laptop_head_client.py
└── inspect_model.py
```

---

## Debugging Commands

### Check model file sizes
```bash
ls -lh models/
du -sh models/split_640/
```

### Test ONNX runtime
```bash
python3 -c "import onnxruntime; print(onnxruntime.get_device())"
```

### Monitor Pi CPU during inference
```bash
# Terminal 1: start inference
python3 pi_backbone_server.py ...

# Terminal 2 (on Pi): monitor
watch -n1 'top -bn1 | head -20'
```

### Verify network connectivity
```bash
# On Pi
ping 100.118.245.12  # Laptop IP

# On Laptop
ping 100.64.24.5     # Pi IP
```

### View Tailscale IPs
```bash
tailscale status
# or
ip addr show tailscale0
```

---

## Testing Strategy

1. **Local test first** (no network)
   ```bash
   python3 laptop_head_client.py ... --test-images images/
   ```

2. **Images over network** (reproducible)
   ```bash
   # Terminal 1 (Laptop)
   python3 laptop_head_client.py ...
   
   # Terminal 2 (Pi)
   python3 pi_backbone_server.py ... --images images/ --fps 1
   ```

3. **Live camera** (final test)
   ```bash
   python3 pi_backbone_server.py ... --camera 0
   ```

---

## Next Steps (Post-Verification)

### Add detection filtering
```python
# In laptop_head_client.py decode_detections()
if int(class_val) == 0:  # Only leopards
    detections.append(...)
```

### Add visualization
```python
# Draw boxes on dummy image or save results
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite("detections.jpg", img)
```

### Quantize backbone (Pi only)
```bash
# Quantize only backbone for 20-30% speedup
python3 -c "
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='onnx', imgsz=256, quantize=8)  # Now safe to quantize backbone only
"
```

### Deploy to production
- Copy backbone to Pi filesystem
- Copy head to Laptop filesystem
- Run as systemd service
- Add logging/error handling

---

## Key Metrics to Log

```
Frame#, Backbone_ms, Send_ms, Receive_ms, Head_ms, Decode_ms, Detections, FPS
1,      234.2,      3.1,     2.8,        12.3,    1.2,       3,           1.0
2,      231.5,      3.2,     2.9,        12.1,    1.1,       2,           2.0
...
```

Use this to identify bottlenecks:
- Backbone > 250ms? Pi issue
- Send/Receive > 5ms? Network issue
- Head > 20ms? Laptop issue

---

## Commands Cheat Sheet

```bash
# Setup (one time)
python3 create_split_model.py models/best_640.onnx models/split_640 640
python3 verify_split_new.py models/best_640.onnx models/split_640/backbone_640.onnx models/split_640/head_640.onnx images/ 640

# Testing
python3 laptop_head_client.py models/split_640/head_640.onnx --test-images images/

# Running (separate terminals)
# Terminal 1 (Laptop)
python3 laptop_head_client.py models/split_640/head_640.onnx --port 5000 --imgsz 640

# Terminal 2 (Pi)
python3 pi_backbone_server.py models/split_640/backbone_640.onnx 100.118.245.12 5000 640 --images images/

# Inspect
python3 inspect_model.py models/best_640.onnx
python3 inspect_model.py models/split_640/backbone_640.onnx
python3 inspect_model.py models/split_640/head_640.onnx

# Monitor
tailscale status
top -u $USER
```

---

## Emergency Reset

If something breaks:

```bash
# Remove old split models
rm -rf models/split_640 models/split_256

# Re-split from original
python3 create_split_model.py models/best_640.onnx models/split_640 640

# Re-verify
python3 verify_split_new.py models/best_640.onnx models/split_640/backbone_640.onnx models/split_640/head_640.onnx images/ 640
```

---

**Ready to go! Start with the One-Shot Setup above.** 🚀
