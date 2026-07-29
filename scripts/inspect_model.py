"""
Inspect ONNX model structure - useful for debugging tensor names and shapes.

Usage:
    python3 inspect_model.py <model_path> [--verbose]
    
Examples:
    python3 inspect_model.py models/best_640.onnx
    python3 inspect_model.py models/best_640.onnx --verbose
"""

import onnx
import onnxruntime as ort
import sys
from pathlib import Path

def inspect_model(model_path, verbose=False):
    """Inspect ONNX model structure"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"[!] Model not found: {model_path}")
        sys.exit(1)
    
    print(f"[*] Loading model: {model_path}")
    print(f"    Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Load with ONNX
    model = onnx.load(str(model_path))
    
    # Load with onnxruntime
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    
    print(f"\n{'='*80}")
    print("INPUTS")
    print(f"{'='*80}")
    
    for inp in model.graph.input:
        print(f"\nName: {inp.name}")
        print(f"Type: {inp.type.tensor_type.elem_type}")  # Data type
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"Shape: {shape}")
    
    print(f"\n{'='*80}")
    print("OUTPUTS")
    print(f"{'='*80}")
    
    for out in model.graph.output:
        print(f"\nName: {out.name}")
        print(f"Type: {out.type.tensor_type.elem_type}")
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"Shape: {shape}")
    
    print(f"\n{'='*80}")
    print("NODES (Operations)")
    print(f"{'='*80}")
    
    if verbose:
        for i, node in enumerate(model.graph.node):
            print(f"\n[{i:3d}] Op: {node.op_type}")
            print(f"     Name: {node.name}")
            if node.input:
                print(f"     Input: {node.input}")
            if node.output:
                print(f"     Output: {node.output}")
    else:
        # Count by type
        op_counts = {}
        for node in model.graph.node:
            op = node.op_type
            op_counts[op] = op_counts.get(op, 0) + 1
        
        print("\nOperation counts:")
        for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
            print(f"  {op:<20} {count:3d}")
        
        print(f"\nTotal nodes: {len(model.graph.node)}")
    
    # List all intermediate tensor names
    print(f"\n{'='*80}")
    print("ALL INTERMEDIATE TENSORS (Layer-by-layer)")
    print(f"{'='*80}")
    
    # Get all value_info (intermediate tensors)
    value_infos = {}
    
    # From graph value_info
    for vi in model.graph.value_info:
        shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        value_infos[vi.name] = shape
    
    # From node outputs
    for node in model.graph.node:
        for output in node.output:
            if output not in value_infos:
                # Try to infer from node
                value_infos[output] = "Unknown shape"
    
    # Sort by name
    for name in sorted(value_infos.keys()):
        shape = value_infos[name]
        print(f"  {name:<60} {str(shape):<30}")
    
    # Show layer 4 output specifically (important for split)
    print(f"\n{'='*80}")
    print("LAYER 4 OUTPUT (Split Point)")
    print(f"{'='*80}")
    
    layer4_candidates = [
        "/model.4/cv2/act/Mul_output_0",  # Expected
        "/model.4/act/Mul_output_0",       # Alternative
    ]
    
    for candidate in layer4_candidates:
        if candidate in value_infos:
            print(f"\n✓ Found: {candidate}")
            print(f"  Shape: {value_infos[candidate]}")
            break
    else:
        print(f"\n! Candidates not found directly in value_info")
        print(f"  Searching for '/model.4' entries...")
        layer4_entries = [k for k in value_infos.keys() if "/model.4" in k and "act" in k]
        if layer4_entries:
            print(f"  Found {len(layer4_entries)} candidates:")
            for entry in sorted(layer4_entries):
                print(f"    - {entry}")
                print(f"      Shape: {value_infos[entry]}")
        else:
            print(f"  No /model.4 entries found")
    
    # Runtime info
    print(f"\n{'='*80}")
    print("RUNTIME INFO")
    print(f"{'='*80}")
    
    print(f"\nInput to session:")
    for inp in session.get_inputs():
        print(f"  Name: {inp.name}")
        print(f"  Shape: {inp.shape}")
        print(f"  Type: {inp.type}")
    
    print(f"\nOutput from session:")
    for out in session.get_outputs():
        print(f"  Name: {out.name}")
        print(f"  Shape: {out.shape}")
        print(f"  Type: {out.type}")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_path> [--verbose]")
        print(f"Example: python3 {sys.argv[0]} models/best_640.onnx")
        sys.exit(1)
    
    model_path = sys.argv[1]
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    inspect_model(model_path, verbose)

if __name__ == "__main__":
    main()
