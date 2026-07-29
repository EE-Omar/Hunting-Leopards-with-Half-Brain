"""
Verify YOLO26 split model: Compare full model output vs (backbone + head) output.
Tests on all images in a folder to ensure mathematical equivalence.

Usage: python3 verify_split_new.py <full_model> <backbone> <head> <images_folder> [imgsz]
Example: python3 verify_split_new.py models/best_640.onnx models/split_640/backbone_640.onnx models/split_640/head_640.onnx images/ 640
"""

import onnxruntime as ort
import cv2
import numpy as np
from pathlib import Path
import sys

class YOLO26Model:
    """Wrapper for YOLO26 ONNX inference"""
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
    
    def predict(self, image):
        """Run inference, returns output tensor"""
        outputs = self.session.run(self.output_names, {self.input_name: image})
        return outputs[0]

def preprocess_image(image_path, imgsz):
    """Preprocess image for YOLO26 model (256x256 or 640x640)"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Resize to target size
    img = cv2.resize(img, (imgsz, imgsz))
    
    # Convert BGR to RGB and normalize (0-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension: (H, W, C) -> (1, C, H, W)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img

def verify_split(full_model_path, backbone_path, head_path, images_folder, imgsz=640):
    """
    Verify split model by comparing outputs on all test images.
    """
    full_model_path = Path(full_model_path)
    backbone_path = Path(backbone_path)
    head_path = Path(head_path)
    images_folder = Path(images_folder)
    
    print(f"[*] Loading models...")
    full_model = YOLO26Model(str(full_model_path))
    backbone = YOLO26Model(str(backbone_path))
    head = YOLO26Model(str(head_path))
    print(f"[✓] Models loaded")
    
    # Get all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = []
    for ext in image_extensions:
        images.extend(images_folder.glob(f"*{ext}"))
        images.extend(images_folder.glob(f"*{ext.upper()}"))
    
    if not images:
        print(f"[!] No images found in {images_folder}")
        return
    
    images = sorted(list(set(images)))  # Remove duplicates
    print(f"[*] Found {len(images)} images")
    
    print(f"\n[*] Running verification on {imgsz}x{imgsz} images...")
    print(f"{'Image':<40} {'Match':<10} {'Max Error':<15} {'Status':<10}")
    print("-" * 75)
    
    passes = 0
    fails = 0
    max_error_global = 0.0
    
    for image_path in images:
        # Preprocess
        img = preprocess_image(image_path, imgsz)
        if img is None:
            print(f"{image_path.name:<40} {'SKIP':<10} {'N/A':<15} {'No img':<10}")
            continue
        
        # Full model inference
        full_output = full_model.predict(img)
        
        # Split inference: backbone -> head
        backbone_output = backbone.predict(img)
        split_output = head.predict(backbone_output)
        
        # Compare
        max_error = np.max(np.abs(full_output - split_output))
        match = "✓" if max_error < 1e-3 else "✗"
        status = "PASS" if max_error < 1e-3 else "FAIL"
        
        if max_error < 1e-3:
            passes += 1
        else:
            fails += 1
        
        max_error_global = max(max_error_global, max_error)
        
        print(f"{image_path.name:<40} {match:<10} {max_error:<15.8f} {status:<10}")
    
    print("-" * 75)
    print(f"\n[✓] RESULTS:")
    print(f"    Total images: {len(images)}")
    print(f"    PASS: {passes}")
    print(f"    FAIL: {fails}")
    print(f"    Max error (global): {max_error_global:.8f}")
    
    if fails == 0:
        print(f"\n[✓] Split model verified! Full model and split model are mathematically equivalent.")
    else:
        print(f"\n[!] WARNING: Split model does NOT match full model on {fails} images!")
        print(f"    Check tensor names and layer connections.")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <full_model> <backbone> <head> <images_folder> [imgsz]")
        print(f"Example: python3 {sys.argv[0]} models/best_640.onnx models/split_640/backbone_640.onnx models/split_640/head_640.onnx images/ 640")
        sys.exit(1)
    
    full_model = sys.argv[1]
    backbone = sys.argv[2]
    head = sys.argv[3]
    images_folder = sys.argv[4]
    imgsz = int(sys.argv[5]) if len(sys.argv) > 5 else 640
    
    verify_split(full_model, backbone, head, images_folder, imgsz)
