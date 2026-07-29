"""
Create split YOLO26 model: Backbone (Layer 0-4) + Head (Layer 4-output)
Usage: python3 create_split_model.py <full_onnx_model> <output_dir> [imgsz]
Example: python3 create_split_model.py models/best_640.onnx models/split_640 640
"""

import onnx
import onnx.utils
import sys
from pathlib import Path

def split_model(full_model_path, output_dir, imgsz=640):
    """
    Split YOLO26 ONNX model into backbone and head.
    
    Args:
        full_model_path: Path to full ONNX model (best_256.onnx or best_640.onnx)
        output_dir: Directory to save backbone.onnx and head.onnx
        imgsz: Image size (256 or 640) - used for naming/tracking
    """
    
    full_model_path = Path(full_model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Loading full model: {full_model_path}")
    full_model = onnx.load(str(full_model_path))
    
    # Inspect model structure
    print(f"[*] Full model inputs: {[inp.name for inp in full_model.graph.input]}")
    print(f"[*] Full model outputs: {[out.name for out in full_model.graph.output]}")
    
    # Extract backbone: images -> /model.4/cv2/act/Mul_output_0
    print(f"\n[*] Extracting backbone (Layer 0-4)...")
    backbone_input = "images"
    backbone_output = "/model.4/cv2/act/Mul_output_0"
    
    backbone_model = onnx.utils.extract_model(
        str(full_model_path),
        str(output_dir / f"backbone_{imgsz}.onnx"),
        [backbone_input],
        [backbone_output]
    )
    print(f"[✓] Backbone saved: {output_dir / f'backbone_{imgsz}.onnx'}")
    print(f"    Input: {backbone_input}")
    print(f"    Output: {backbone_output}")
    print(f"    Expected output shape: (1, 128, 32, 32) or similar")
    
    # Extract head: /model.4/cv2/act/Mul_output_0 -> output0
    print(f"\n[*] Extracting head (Layer 4-output)...")
    head_input = backbone_output
    head_output = "output0"
    
    head_model = onnx.utils.extract_model(
        str(full_model_path),
        str(output_dir / f"head_{imgsz}.onnx"),
        [head_input],
        [head_output]
    )
    print(f"[✓] Head saved: {output_dir / f'head_{imgsz}.onnx'}")
    print(f"    Input: {head_input}")
    print(f"    Output: {head_output}")
    print(f"    Expected output shape: (1, 300, 6)")
    
    print(f"\n[✓] Split complete!")
    print(f"    Backbone: {output_dir / f'backbone_{imgsz}.onnx'}")
    print(f"    Head: {output_dir / f'head_{imgsz}.onnx'}")
    print(f"\n[*] Next steps:")
    print(f"    1. Run verify_split_new.py to validate the split")
    print(f"    2. Deploy backbone to Pi3")
    print(f"    3. Deploy head to laptop")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <full_model_path> <output_dir> [imgsz]")
        print(f"Example: python3 {sys.argv[0]} models/best_640.onnx models/split_640 640")
        sys.exit(1)
    
    full_model_path = sys.argv[1]
    output_dir = sys.argv[2]
    imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 640
    
    split_model(full_model_path, output_dir, imgsz)
