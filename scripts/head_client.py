"""
Laptop Head Client
Receives backbone tensors over TCP from Pi3, runs head inference, decodes detections.

Usage (Laptop):
    python3 laptop_head_client.py <head_model> [port] [imgsz]
    
Examples:
    # Standard (wait for Pi to connect)
    python3 laptop_head_client.py models/split_640/head_640.onnx --port 5000 --imgsz 640
    
    # Test mode (uses local images)
    python3 laptop_head_client.py models/split_640/head_640.onnx --port 5000 --imgsz 640 --test-images data/images/
"""

import onnxruntime as ort
import numpy as np
import socket
import struct
import sys
import time
from pathlib import Path
import argparse
import cv2

class HeadClient:
    def __init__(self, head_path, imgsz=640):
        self.head_path = Path(head_path)
        self.imgsz = imgsz
        
        print(f"[*] Loading head model: {self.head_path}")
        self.session = ort.InferenceSession(
            str(self.head_path),
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        print(f"[✓] Head loaded")
        print(f"    Input: {self.input_name}")
        print(f"    Output: {self.output_names[0]}")
        
        output_shape = self.session.get_outputs()[0].shape
        print(f"    Output shape: {output_shape}")
    
    def run_head(self, backbone_tensor):
        """Run head inference on backbone output"""
        outputs = self.session.run(self.output_names, {self.input_name: backbone_tensor})
        return outputs[0]
    
    def decode_detections(self, raw_output, conf_threshold=0.3, target_class=0):
        detections = []
        
        for detection in raw_output[0]:
            x1, y1, x2, y2, confidence, class_val = detection
            
            if confidence < conf_threshold:
                continue
            
            cls_id = int(round(float(class_val)))
            if target_class != -1 and cls_id != target_class:
                continue
            
            x1 = max(0, min(float(x1), self.imgsz))
            y1 = max(0, min(float(y1), self.imgsz))
            x2 = max(0, min(float(x2), self.imgsz))
            y2 = max(0, min(float(y2), self.imgsz))
            
            if (x2 - x1) > 1 and (y2 - y1) > 1:
                detections.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': float(confidence),
                    'class_value': cls_id,
                })
        
        return detections

    def print_detections(self, detections):
        if not detections:
            print("  No detections")
            return
        
        for i, det in enumerate(detections):
            print(f"  [{i}] Box: ({det['x1']:.1f}, {det['y1']:.1f}) -> ({det['x2']:.1f}, {det['y2']:.1f}) | "
                  f"Conf: {det['confidence']:.3f} | "
                  f"Class: {det['class_value']}")
    
    def print_detections(self, detections):
        """Pretty-print detections"""
        if not detections:
            print("  No detections")
            return
        
        for i, det in enumerate(detections):
            print(f"  [{i}] Box: ({det['x1']:.1f}, {det['y1']:.1f}) -> ({det['x2']:.1f}, {det['y2']:.1f}) | "
                  f"Conf: {det['confidence']:.3f} | "
                  f"Class: {det['class_value']}")

def recv_exact(sock, n):
    """Read exactly n bytes, or return None if the peer hangs up.

    TCP does not guarantee recv() returns everything asked for. The payload loop below
    always handled that, but the header reads did not — and short reads get more likely,
    not less, as the bottleneck shrinks frames toward a single segment.
    """
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def receive_tensor(sock):
    """
    Receive tensor from Pi over TCP.

    Format: [dtype(1)] [num_dims(1)] [shape(4 each)] {[scale(4)] [zero_point(4)]} [data]

    dtype 1 = float32 (raw layer-4 tensor)
    dtype 2 = uint8   (bottleneck tensor, dequantized here using the header's params)
    """
    try:
        header = recv_exact(sock, 2)
        if header is None:
            return None
        dtype_byte, num_dims = struct.unpack('BB', header)

        shape_bytes = recv_exact(sock, 4 * num_dims)
        if shape_bytes is None:
            return None
        shape = struct.unpack(f'{num_dims}I', shape_bytes)

        if dtype_byte == 1:
            dtype, itemsize, quant = np.float32, 4, None
        elif dtype_byte == 2:
            qp = recv_exact(sock, 8)
            if qp is None:
                return None
            quant = struct.unpack('ff', qp)
            dtype, itemsize = np.uint8, 1
        else:
            print(f"[!] Unknown wire dtype: {dtype_byte}")
            return None

        total_elements = 1
        for s in shape:
            total_elements *= s

        data = recv_exact(sock, total_elements * itemsize)
        if data is None:
            print("[!] Connection closed by Pi")
            return None

        tensor = np.frombuffer(data, dtype=dtype).reshape(shape)
        if quant is not None:
            scale, zero_point = quant
            tensor = (tensor.astype(np.float32) - zero_point) * scale
        return tensor

    except Exception as e:
        print(f"[!] Error receiving tensor: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Laptop Head Client - receives backbone tensors and runs head inference"
    )
    parser.add_argument("head_model", help="Path to head ONNX model")
    parser.add_argument("--port", type=int, default=5000, help="TCP port (default: 5000)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (256 or 640)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument("--test-images", type=str, default=None, 
                        help="Test mode: generate backbone tensors from images locally")
    parser.add_argument("--backbone", type=str, default=None,
                        help="Path to backbone model (used in test mode)")
    
    args = parser.parse_args()
    
    # Create head client
    client = HeadClient(args.head_model, args.imgsz)
    
    # Test mode: load backbone locally and use images
    if args.test_images:
        print(f"\n[*] TEST MODE: Using local images from {args.test_images}")
        print(f"    (Not waiting for Pi connection)")
        
        # Use explicit --backbone if provided, otherwise search
        if args.backbone:
            backbone_path = Path(args.backbone)
            if not backbone_path.exists():
                print(f"[!] Backbone not found at: {backbone_path}")
                sys.exit(1)
        else:
            backbone_paths = [
                Path(args.head_model).parent / f"backbone_{args.imgsz}.onnx",
                Path("models") / f"backbone_{args.imgsz}.onnx",
                Path("models/split") / f"backbone_{args.imgsz}.onnx",
            ]
            backbone_path = None
            for p in backbone_paths:
                if p.exists():
                    backbone_path = p
                    break
            if backbone_path is None:
                print(f"[!] Could not find backbone model")
                print(f"    Searched: {[str(p) for p in backbone_paths]}")
                print(f"    Fix: pass --backbone path/to/backbone_256.onnx explicitly")
                sys.exit(1)
        
        print(f"[✓] Found backbone: {backbone_path}")
        backbone_session = ort.InferenceSession(
            str(backbone_path),
            providers=['CPUExecutionProvider']
        )
        backbone_input_name = backbone_session.get_inputs()[0].name
        backbone_output_names = [out.name for out in backbone_session.get_outputs()]
        
        # Process images
        image_folder = Path(args.test_images)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        images = []
        for ext in image_extensions:
            images.extend(image_folder.glob(f"*{ext}"))
            images.extend(image_folder.glob(f"*{ext.upper()}"))
        
        images = sorted(list(set(images)))
        
        if not images:
            print(f"[!] No images found in {image_folder}")
            sys.exit(1)
        
        print(f"[*] Found {len(images)} images")
        print(f"\nProcessing...\n")
        
        frame_num = 0
        start_time = time.time()
        
        for image_path in images:
            frame_num += 1
            
            # Load and preprocess image
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            
            img = cv2.resize(img, (args.imgsz, args.imgsz))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype(np.float32) / 255.0
            img_tensor = np.transpose(img_norm, (2, 0, 1))
            img_tensor = np.expand_dims(img_tensor, axis=0)
            
            # Run backbone -> head
            t0 = time.time()
            backbone_out = backbone_session.run(backbone_output_names, {backbone_input_name: img_tensor})
            t_backbone = time.time() - t0
            
            t0 = time.time()
            head_out = client.run_head(backbone_out[0])
            t_head = time.time() - t0
            
            # Decode
            detections = client.decode_detections(head_out, conf_threshold=args.conf)
            
            elapsed = time.time() - start_time
            fps = frame_num / elapsed if elapsed > 0 else 0
            
            print(f"[{frame_num:3d}] {image_path.name:<40} "
                  f"Backbone: {t_backbone*1000:6.2f}ms | "
                  f"Head: {t_head*1000:6.2f}ms | "
                  f"Dets: {len(detections):3d} | "
                  f"FPS: {fps:6.2f}")
            
            if detections:
                client.print_detections(detections)
        
        print(f"\n[✓] Test complete!")
        return
    
    # Normal mode: wait for Pi connection
    print(f"\n[*] Starting head server on port {args.port}...")
    print(f"    Waiting for Pi backbone to connect...")
    print(f"\n    On Pi, run:")
    print(f"    python3 pi_backbone_server.py backbone.onnx <laptop_ip> {args.port} {args.imgsz}")
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', args.port))
    server_socket.listen(1)
    
    try:
        print(f"\n[*] Listening on port {args.port}...\n")
        conn, addr = server_socket.accept()
        print(f"[✓] Pi connected from {addr}")
        
        frame_num = 0
        start_time = time.time()
        
        while True:
            # Receive tensor from Pi
            tensor = receive_tensor(conn)
            if tensor is None:
                break
            
            frame_num += 1
            
            # Run head inference
            t0 = time.time()
            head_out = client.run_head(tensor)
            t_head = time.time() - t0
            
            # Decode
            t0 = time.time()
            detections = client.decode_detections(head_out, conf_threshold=args.conf)
            t_decode = time.time() - t0
            
            elapsed = time.time() - start_time
            fps = frame_num / elapsed if elapsed > 0 else 0
            
            print(f"[{frame_num:4d}] Head: {t_head*1000:6.2f}ms | "
                  f"Decode: {t_decode*1000:6.2f}ms | "
                  f"Dets: {len(detections):3d} | "
                  f"FPS: {fps:6.2f}")
            
            if detections:
                client.print_detections(detections)
    
    except KeyboardInterrupt:
        print(f"\n[*] Stopping...")
    except Exception as e:
        print(f"[!] Error: {e}")
    finally:
        server_socket.close()
        print(f"[✓] Server closed")

if __name__ == "__main__":
    main()