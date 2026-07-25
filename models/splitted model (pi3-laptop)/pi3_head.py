#!/usr/bin/env python3
"""
pi0_head.py  --  runs on the Raspberry Pi Zero (the camera trap).
Captures/loads a frame, runs the FRONT of the network (head.onnx), compresses
the feature tensor to int8, sends it to the Pi 3, and prints the detections.

Install once on the Pi Zero:   pip install onnxruntime pillow numpy
Run:   python3 pi0_head.py --image frame.jpg --host <PI3_IP_ADDRESS>
"""
import argparse, socket, struct, json, time
import numpy as np, onnxruntime as ort
from PIL import Image

def letterbox(img, new=640, color=114):
    w0, h0 = img.size
    r = min(new / w0, new / h0); nw, nh = round(w0 * r), round(h0 * r)
    canvas = Image.new("RGB", (new, new), (color, color, color))
    canvas.paste(img.resize((nw, nh)), ((new - nw) // 2, (new - nh) // 2))
    return canvas, r, (new - nw) // 2, (new - nh) // 2

def recv_all(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c: raise ConnectionError("socket closed")
        buf += c
    return buf
def recv_msg(sock):
    n = struct.unpack(">I", recv_all(sock, 4))[0]; return recv_all(sock, n)
def send_msg(sock, data): sock.sendall(struct.pack(">I", len(data)) + data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="head.onnx")
    ap.add_argument("--image", required=True)
    ap.add_argument("--host", required=True, help="Pi 3 IP address")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--conf", type=float, default=0.5)
    a = ap.parse_args()

    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    img = Image.open(a.image).convert("RGB")
    lb, r, dx, dy = letterbox(img, 640)
    x = (np.asarray(lb).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    t0 = time.time()
    inter = sess.run(None, {iname: x})[0]                 # front layers -> feature tensor
    t_head = time.time() - t0

    # ---- compress the tensor: int8 symmetric quantization (4x smaller) ----
    scale = float(np.abs(inter).max()) / 127.0 or 1.0
    q = np.clip(np.round(inter / scale), -127, 127).astype(np.int8)

    # ---- message = scale (f64) + shape (4 int32) + int8 payload ----
    payload = struct.pack(">d4i", scale, *inter.shape) + q.tobytes()

    with socket.create_connection((a.host, a.port)) as s:
        send_msg(s, payload)
        resp = recv_msg(s)
    t_total = time.time() - t0

    dets = json.loads(resp.decode())
    print(f"head: {t_head*1000:.0f} ms | sent {len(payload)/1024:.0f} KB | round-trip {t_total*1000:.0f} ms")
    print(f"detections above {a.conf}:")
    for d in dets:
        if d[4] < a.conf: continue
        x1=(d[0]-dx)/r; y1=(d[1]-dy)/r; x2=(d[2]-dx)/r; y2=(d[3]-dy)/r
        print(f"   class {int(d[5])}  conf {d[4]:.2f}  box [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

if __name__ == "__main__":
    main()
