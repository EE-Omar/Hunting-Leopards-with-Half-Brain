"""
Pi3 Backbone Server
Runs YOLO26 backbone inference and streams tensors over TCP to laptop.

Usage (Pi3):
    python3 pi_backbone_server.py <backbone_model> <laptop_ip> [port] [imgsz]
    
Examples:
    # Test with images folder (no camera)
    python3 pi_backbone_server.py models/split_640/backbone_640.onnx 100.x.x.x 5000 640 --images images/
    
    # Live camera (when available)
    python3 pi_backbone_server.py models/split_640/backbone_640.onnx 100.x.x.x 5000 640 --camera 0
"""

import onnxruntime as ort
import cv2
import numpy as np
import socket
import sys
import time
import struct
from pathlib import Path
import argparse

class BackboneServer:
    def __init__(self, backbone_path, laptop_ip, port=5000, imgsz=640):
        self.backbone_path = Path(backbone_path)
        self.laptop_ip = laptop_ip
        self.port = port
        self.imgsz = imgsz
        
        print(f"[*] Loading backbone model: {self.backbone_path}")
        self.session = ort.InferenceSession(
            str(self.backbone_path),
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        print(f"[✓] Backbone loaded")
        print(f"    Input: {self.input_name}")
        print(f"    Output: {self.output_names[0]}")
        
        # Get input shape to verify
        input_shape = self.session.get_inputs()[0].shape
        print(f"    Expected input shape: {input_shape}")
        
    def preprocess(self, image):
        """Preprocess image for YOLO26"""
        img = cv2.resize(image, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
    
    def run_backbone(self, image):
        """Run backbone inference"""
        img = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: img})
        return outputs[0]
    
    def send_tensor(self, sock, tensor):
        """
        Send tensor over TCP with header.
        Format: [dtype(1 byte)] [shape(4 bytes each)] [data]
        """
        # Ensure tensor is float32
        tensor = tensor.astype(np.float32)
        
        # Pack dtype
        dtype_byte = b'\x01'  # 1 = float32
        
        # Pack shape
        shape_bytes = struct.pack(f'{len(tensor.shape)}I', *tensor.shape)
        
        # Flatten and pack data
        tensor_flat = tensor.flatten()
        data_bytes = tensor_flat.tobytes()
        
        # Send header: dtype (1) + num_dims (1) + shape + data
        num_dims = len(tensor.shape)
        header = struct.pack('BB', 1, num_dims) + shape_bytes
        
        # Send all
        sock.sendall(header + data_bytes)
        
        return len(header + data_bytes)

def receive_from_camera(cap, frame_count=0):
    """Read frame from camera"""
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def stream_from_folder(image_folder, frame_rate=1.0):
    """Stream images from folder with delay"""
    image_folder = Path(image_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = []
    for ext in image_extensions:
        images.extend(image_folder.glob(f"*{ext}"))
        images.extend(image_folder.glob(f"*{ext.upper()}"))
    
    images = sorted(list(set(images)))
    
    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is not None:
            yield img
            time.sleep(1.0 / frame_rate)

def main():
    parser = argparse.ArgumentParser(
        description="Pi3 Backbone Server - streams backbone tensors to laptop"
    )
    parser.add_argument("backbone_model", help="Path to backbone ONNX model")
    parser.add_argument("laptop_ip", help="Laptop IP on tailnet (e.g., 100.x.x.x)")
    parser.add_argument("--port", type=int, default=5000, help="TCP port (default: 5000)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (256 or 640)")
    parser.add_argument("--camera", type=int, default=None, help="Camera device (0, 1, etc.)")
    parser.add_argument("--images", type=str, default=None, help="Folder with test images")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for testing")
    
    args = parser.parse_args()
    
    # Create backbone server
    server = BackboneServer(
        args.backbone_model,
        args.laptop_ip,
        args.port,
        args.imgsz
    )
    
    # Connect to laptop
    print(f"\n[*] Connecting to laptop at {args.laptop_ip}:{args.port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((args.laptop_ip, args.port))
        print(f"[✓] Connected to laptop!")
    except Exception as e:
        print(f"[!] Failed to connect: {e}")
        print(f"    Make sure:")
        print(f"    1. Laptop IP is correct (from tailnet status)")
        print(f"    2. Laptop head server is running on port {args.port}")
        sys.exit(1)
    
    # Setup frame source
    if args.images:
        print(f"[*] Using images from folder: {args.images}")
        frame_source = stream_from_folder(args.images, frame_rate=args.fps)
    elif args.camera is not None:
        print(f"[*] Opening camera device {args.camera}")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"[!] Failed to open camera {args.camera}")
            sys.exit(1)
        frame_source = iter(lambda: receive_from_camera(cap), None)
    else:
        print(f"[!] Must specify either --camera or --images")
        parser.print_help()
        sys.exit(1)
    
    # Stream loop
    print(f"\n[*] Starting backbone inference stream...")
    print(f"    Input size: {args.imgsz}x{args.imgsz}")
    print(f"    Output: (1, 128, 32, 32) approx")
    print(f"\nPress Ctrl+C to stop.\n")
    
    frame_num = 0
    start_time = time.time()
    
    try:
        for frame in frame_source:
            if frame is None:
                break
            
            frame_num += 1
            
            # Run backbone
            t0 = time.time()
            tensor = server.run_backbone(frame)
            t_backbone = time.time() - t0
            
            # Send to laptop
            t0 = time.time()
            bytes_sent = server.send_tensor(sock, tensor)
            t_send = time.time() - t0
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_num / elapsed if elapsed > 0 else 0
            
            print(f"[{frame_num:4d}] Backbone: {t_backbone*1000:6.2f}ms | "
                  f"Send: {t_send*1000:6.2f}ms | "
                  f"Bytes: {bytes_sent:8d} | "
                  f"FPS: {fps:6.2f}")
    
    except KeyboardInterrupt:
        print(f"\n[*] Stopping...")
    except Exception as e:
        print(f"[!] Error: {e}")
    finally:
        sock.close()
        print(f"[✓] Connection closed")

if __name__ == "__main__":
    main()
