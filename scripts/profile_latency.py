"""
Profile YOLO26 latency breakdown: full model vs split model.

Usage:
    python3 profile_latency.py <model_path> <test_images_folder> [imgsz]
    
Examples:
    python3 profile_latency.py models/best_640.onnx images/ 640
    python3 profile_latency.py models/best_256.onnx images/ 256
    
    # Compare with split models
    python3 profile_latency.py models/best_640.onnx images/ 640 \
        --backbone models/split_640/backbone_640.onnx \
        --head models/split_640/head_640.onnx
"""

import onnxruntime as ort
import cv2
import numpy as np
from pathlib import Path
import sys
import time
import argparse
from statistics import mean, stdev

class LatencyProfiler:
    def __init__(self, model_path, imgsz=640):
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
    
    def preprocess(self, image):
        """Preprocess image"""
        img = cv2.resize(image, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
    
    def run_inference(self, image, warmup=True):
        """Run inference with timing"""
        img = self.preprocess(image)
        
        # Warmup run (not timed)
        if warmup:
            self.session.run(self.output_names, {self.input_name: img})
        
        # Timed run
        t0 = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: img})
        elapsed = time.perf_counter() - t0
        
        return elapsed, outputs[0]

def profile_full_model(full_model_path, images, imgsz=640, warmup=True):
    """Profile full model inference"""
    print(f"\n[*] Profiling full model: {full_model_path}")
    print(f"    Image size: {imgsz}x{imgsz}")
    print(f"    Samples: {len(images)}")
    
    profiler = LatencyProfiler(str(full_model_path), imgsz)
    
    latencies = []
    
    for i, image_path in enumerate(images):
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        
        elapsed, _ = profiler.run_inference(img, warmup=(i==0))
        latencies.append(elapsed * 1000)  # Convert to ms
        
        print(f"  [{i+1:3d}] {elapsed*1000:7.2f} ms")
    
    return latencies

def profile_split_model(backbone_path, head_path, images, imgsz=640):
    """Profile split model (backbone + head)"""
    print(f"\n[*] Profiling split model")
    print(f"    Backbone: {backbone_path}")
    print(f"    Head: {head_path}")
    print(f"    Image size: {imgsz}x{imgsz}")
    print(f"    Samples: {len(images)}")
    
    backbone_profiler = LatencyProfiler(str(backbone_path), imgsz)
    head_profiler = LatencyProfiler(str(head_path), imgsz)
    
    backbone_latencies = []
    head_latencies = []
    total_latencies = []
    
    for i, image_path in enumerate(images):
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        
        # Backbone
        t_backbone, backbone_out = backbone_profiler.run_inference(img, warmup=(i==0))
        backbone_latencies.append(t_backbone * 1000)
        
        # Head
        t_head, _ = head_profiler.run_inference(img, warmup=(i==0))
        head_latencies.append(t_head * 1000)
        
        total = (t_backbone + t_head) * 1000
        total_latencies.append(total)
        
        print(f"  [{i+1:3d}] Backbone: {t_backbone*1000:7.2f} ms | "
              f"Head: {t_head*1000:7.2f} ms | "
              f"Total: {total:7.2f} ms")
    
    return {
        'backbone': backbone_latencies,
        'head': head_latencies,
        'total': total_latencies,
    }

def print_stats(latencies, name):
    """Print latency statistics"""
    if not latencies:
        return
    
    latencies = list(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    avg_lat = mean(latencies)
    
    if len(latencies) > 1:
        std_lat = stdev(latencies)
    else:
        std_lat = 0
    
    fps = 1000 / avg_lat if avg_lat > 0 else 0
    
    print(f"\n{name}")
    print(f"  Min:        {min_lat:7.2f} ms  ({1000/max_lat:5.2f} FPS)")
    print(f"  Max:        {max_lat:7.2f} ms  ({1000/min_lat:5.2f} FPS)")
    print(f"  Mean:       {avg_lat:7.2f} ms  ({fps:5.2f} FPS)")
    print(f"  Std Dev:    {std_lat:7.2f} ms")
    print(f"  Samples:    {len(latencies)}")

def main():
    parser = argparse.ArgumentParser(
        description="Profile YOLO26 latency: full model vs split model"
    )
    parser.add_argument("model_path", help="Path to full ONNX model")
    parser.add_argument("images_folder", help="Folder with test images")
    parser.add_argument("imgsz", type=int, nargs='?', default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--backbone", help="Path to backbone model (optional)")
    parser.add_argument("--head", help="Path to head model (optional)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Limit number of samples (default: all)")
    
    args = parser.parse_args()
    
    # Load images
    images_folder = Path(args.images_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = []
    for ext in image_extensions:
        images.extend(images_folder.glob(f"*{ext}"))
        images.extend(images_folder.glob(f"*{ext.upper()}"))
    
    images = sorted(list(set(images)))
    
    if not images:
        print(f"[!] No images found in {images_folder}")
        sys.exit(1)
    
    if args.samples:
        images = images[:args.samples]
    
    print(f"[*] Found {len(images)} test images")
    
    # Profile full model
    full_latencies = profile_full_model(args.model_path, images, args.imgsz)
    
    # Profile split model if provided
    split_latencies = None
    if args.backbone and args.head:
        split_latencies = profile_split_model(args.backbone, args.head, images, args.imgsz)
    
    # Print statistics
    print(f"\n{'='*80}")
    print("LATENCY STATISTICS")
    print(f"{'='*80}")
    
    print_stats(full_latencies, "Full Model")
    
    if split_latencies:
        print_stats(split_latencies['backbone'], "\nSplit: Backbone")
        print_stats(split_latencies['head'], "Split: Head")
        print_stats(split_latencies['total'], "Split: Total (Backbone + Head)")
        
        # Comparison
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        
        full_mean = mean(full_latencies)
        split_mean = mean(split_latencies['total'])
        difference = split_mean - full_mean
        percent_diff = (difference / full_mean) * 100
        
        print(f"\nFull model mean:  {full_mean:7.2f} ms")
        print(f"Split model mean: {split_mean:7.2f} ms")
        print(f"Difference:       {difference:+7.2f} ms ({percent_diff:+6.2f}%)")
        
        if abs(difference) < 1:
            print(f"\n✓ Models are mathematically equivalent (difference < 1ms)")
        else:
            print(f"\n! Warning: Models differ by {difference:.2f}ms")

if __name__ == "__main__":
    main()
