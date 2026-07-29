"""
Draw detection boxes on a single image.
Works with full model or split (backbone + head).

Usage:
    # Full model
    python3 scripts/draw_detections.py models/best_256.onnx data/images/leopard.jpg

    # Split model
    python3 scripts/draw_detections.py models/split_256/head_256.onnx data/images/leopard.jpg \
        --backbone models/split_256/backbone_256.onnx

    # Save output (no display)
    python3 scripts/draw_detections.py models/best_256.onnx data/images/leopard.jpg --save

    # Run on all images in a folder
    python3 scripts/draw_detections.py models/best_256.onnx data/images/ --save --out results/
"""

import onnxruntime as ort
import cv2
import numpy as np
import sys
import argparse
from pathlib import Path

CLASSES = {
    0: ('leopard',      (0,   200, 80)),
    1: ('cheetah',      (255, 180, 0)),
    2: ('hyena',        (200, 80,  0)),
    3: ('nubian_ibex',  (0,   160, 220)),
    4: ('camel',        (180, 0,   200)),
    5: ('cat',          (0,   220, 180)),
    6: ('dog',          (220, 100, 0)),
    7: ('person',       (50,  50,  255)),
}

def load_model(path):
    sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    return sess

def preprocess(img_bgr, imgsz):
    img = cv2.resize(img_bgr, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(img, (2,0,1)), 0)

def run(sess, tensor):
    inp = sess.get_inputs()[0].name
    out = [o.name for o in sess.get_outputs()]
    return sess.run(out, {inp: tensor})[0]

def decode(raw, imgsz, conf_thresh=0.3):
    """
    raw: (1, 300, 6) → [cx, cy, w, h, conf, class]
    Returns list of (x1, y1, x2, y2, conf, class_id)
    """
    dets = []
    for row in raw[0]:
        cx, cy, w, h, conf, cls = row
        if conf < conf_thresh:
            continue
        cls_id = int(round(float(cls)))
        x1 = max(0, cx - w/2)
        y1 = max(0, cy - h/2)
        x2 = min(imgsz, cx + w/2)
        y2 = min(imgsz, cy + h/2)
        if (x2 - x1) > 2 and (y2 - y1) > 2:
            dets.append((x1, y1, x2, y2, float(conf), cls_id))
    return dets

def draw(img_bgr, dets, imgsz):
    H, W = img_bgr.shape[:2]
    sx, sy = W / imgsz, H / imgsz
    out = img_bgr.copy()
    for x1, y1, x2, y2, conf, cls_id in dets:
        px1 = int(x1 * sx); py1 = int(y1 * sy)
        px2 = int(x2 * sx); py2 = int(y2 * sy)
        name, color = CLASSES.get(cls_id, (f'cls{cls_id}', (128,128,128)))
        # box
        cv2.rectangle(out, (px1, py1), (px2, py2), color, 2)
        # label background
        label = f'{name} {conf:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (px1, py1 - th - 6), (px1 + tw + 4, py1), color, -1)
        cv2.putText(out, label, (px1 + 2, py1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
    return out

def process_image(image_path, head_sess, backbone_sess, imgsz, conf, save, out_dir):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f'[!] Cannot read {image_path}'); return

    tensor = preprocess(img, imgsz)

    if backbone_sess:
        tensor = run(backbone_sess, tensor)   # Pi step
    raw = run(head_sess, tensor)              # Laptop step (or full model)

    dets = decode(raw, imgsz, conf)
    result = draw(img, dets, imgsz)

    # Print
    print(f'{image_path.name}: {len(dets)} detection(s)')
    for x1,y1,x2,y2,conf_,cls_id in dets:
        name = CLASSES.get(cls_id, (f'cls{cls_id}',))[0]
        print(f'  {name:<14} conf={conf_:.3f}  box=({int(x1)},{int(y1)})->({int(x2)},{int(y2)})')

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'det_{image_path.name}'
        cv2.imwrite(str(out_path), result)
        print(f'  → saved: {out_path}')
    else:
        cv2.imshow(f'Detections — {image_path.name}', result)
        print('  Press any key for next image...')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Draw YOLO26 detections on images')
    parser.add_argument('model',    help='Head model (or full model if no --backbone)')
    parser.add_argument('input',    help='Image file or folder')
    parser.add_argument('--backbone', default=None, help='Backbone model (split mode)')
    parser.add_argument('--imgsz',  type=int, default=256, help='Model input size (default: 256)')
    parser.add_argument('--conf',   type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--save',   action='store_true', help='Save images instead of displaying')
    parser.add_argument('--out',    default='results/', help='Output folder when --save is used')
    args = parser.parse_args()

    head_sess = load_model(args.model)
    backbone_sess = load_model(args.backbone) if args.backbone else None
    out_dir = Path(args.out)

    mode = 'split' if backbone_sess else 'full'
    print(f'[*] Mode: {mode} | imgsz={args.imgsz} | conf≥{args.conf}')

    inp = Path(args.input)
    if inp.is_file():
        process_image(inp, head_sess, backbone_sess, args.imgsz, args.conf, args.save, out_dir)
    elif inp.is_dir():
        exts = {'.jpg','.jpeg','.png','.bmp','.tiff'}
        images = sorted([f for f in inp.iterdir() if f.suffix.lower() in exts])
        print(f'[*] Found {len(images)} images in {inp}')
        for img_path in images:
            process_image(img_path, head_sess, backbone_sess, args.imgsz, args.conf, args.save, out_dir)
    else:
        print(f'[!] Not a file or folder: {inp}'); sys.exit(1)

if __name__ == '__main__':
    main()
