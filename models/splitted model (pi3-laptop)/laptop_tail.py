import os
import socket
import time
import cv2
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "tail.onnx")

HOST = "0.0.0.0"
PORT = 5000
# Output shape from head.onnx for 640x640 input resolution
HEAD_OUTPUT_SHAPE = (1, 64, 80, 80)

CLASS_NAMES = ["Arabian Leopard", "Other Animal"]  # Adjust classes as needed


def recv_all(sock, count):
    buf = bytearray()
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf.extend(newbuf)
        count -= len(newbuf)
    return buf


def run_tail_server():
    session = ort.InferenceSession(
        MODEL_PATH, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    print(
        f"[Laptop Server] Listening for RPi3 connections on port {PORT}..."
    )

    while True:
        conn, addr = server_socket.accept()
        print(f"\n[Laptop Server] Connected by {addr}")

        # Read Header Lengths
        img_len_bytes = recv_all(conn, 4)
        tensor_len_bytes = recv_all(conn, 4)

        if not img_len_bytes or not tensor_len_bytes:
            conn.close()
            continue

        img_len = int.from_bytes(img_len_bytes, "big")
        tensor_len = int.from_bytes(tensor_len_bytes, "big")

        # Read Payload Data
        img_bytes = recv_all(conn, img_len)
        tensor_bytes = recv_all(conn, tensor_len)
        conn.close()

        if img_bytes and tensor_bytes:
            # 1. Decode Image & Tensor
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            intermediate_tensor = np.frombuffer(
                tensor_bytes, dtype=np.float32
            ).reshape(HEAD_OUTPUT_SHAPE)

            # 2. Run Tail Model
            t0 = time.time()
            tail_outputs = session.run(None, {input_name: intermediate_tensor})
            print(f"[Laptop Server] Tail Execution: {(time.time()-t0)*1000:.2f} ms")

            detections = tail_outputs[0][0]  # Shape: (300, 6)

            # 3. Draw Bounding Boxes
            detections_found = 0
            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det
                if conf > 0.25:  # Confidence Threshold
                    detections_found += 1
                    x1, y1, x2, y2 = (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    )
                    cls_id = int(cls_id)
                    label_text = f"{CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else cls_id}: {conf:.2f}"

                    # Draw rectangle and text on image
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        label_text,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            print(
                f"[Laptop Server] Detections Rendered: {detections_found} bounding boxes."
            )

            # 4. Display Window
            cv2.imshow("Split Inference - Detection Result", frame)
            cv2.waitKey(1)


if __name__ == "__main__":
    run_tail_server()