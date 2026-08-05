"""
Cache layer-4 activations so the bottleneck can be trained without the frozen YOLO in
the loop. This is what makes CPU-only training practical: the ~35K-param autoencoder
then trains on precomputed tensors instead of re-running the backbone every epoch.

Preprocessing deliberately matches the deployment path in scripts/backbone_server.py
(plain cv2.resize, BGR->RGB, /255) rather than Ultralytics' letterbox — the bottleneck
should be fitted to the feature distribution it will actually see in the field.

Usage:
    python scripts/bottleneck/cache_features.py --imgsz 256 --limit 2000
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.bottleneck.modules import SPLIT_CHANNELS, backbone_stub  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def list_images(folder, limit=None, seed=42):
    """Collect image paths, shuffled deterministically so a subset stays representative."""
    folder = Path(folder)
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"[!] No images found under {folder}")
    random.Random(seed).shuffle(images)
    if limit:
        images = images[:limit]
    return images


def preprocess(path, imgsz):
    """Identical to BackboneServer.preprocess() — deployment-matched."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))


def main():
    parser = argparse.ArgumentParser(description="Cache YOLO26 layer-4 features to disk")
    parser.add_argument("--weights", default=str(ROOT / "models/full/best_v2.pt"))
    parser.add_argument("--images", default=str(ROOT / "data/v2_yolo_dataset/train/images"))
    parser.add_argument("--out", default=str(ROOT / "models/bottleneck/feats_256.npy"))
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--limit", type=int, default=2000,
                        help="Number of images to cache (2000 gives ~2M feature vectors)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from ultralytics import YOLO

    print(f"[*] Loading {args.weights}")
    net = YOLO(args.weights).model
    net.eval()
    stub = backbone_stub(net).eval()
    for p in stub.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in stub.parameters())
    print(f"[*] Backbone stub (layers 0-4): {n_params:,} params")

    images = list_images(args.images, args.limit, args.seed)
    grid = args.imgsz // 8  # layers 0-4 downsample by 8
    print(f"[*] Caching {len(images)} images -> ({SPLIT_CHANNELS}, {grid}, {grid}) fp16 each")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_mb = len(images) * SPLIT_CHANNELS * grid * grid * 2 / 1024**2
    print(f"[*] Output: {out_path}  (~{total_mb:.0f} MB)")

    cache = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float16,
        shape=(len(images), SPLIT_CHANNELS, grid, grid),
    )

    written = 0
    batch = []
    with torch.no_grad():
        for idx, path in enumerate(images):
            arr = preprocess(path, args.imgsz)
            if arr is not None:
                batch.append(arr)

            is_last = idx == len(images) - 1
            if len(batch) == args.batch or (is_last and batch):
                x = torch.from_numpy(np.stack(batch))
                feats = stub(x).to(torch.float16).numpy()
                cache[written:written + len(feats)] = feats
                written += len(feats)
                batch = []
                print(f"\r    {written}/{len(images)} cached", end="", flush=True)

    cache.flush()
    print(f"\n[+] Wrote {written} feature maps to {out_path}")

    # Distribution matters: the quantizer calibrates on percentiles because this
    # post-SiLU tensor is long-tailed. Print it so that choice is grounded in data.
    sample = np.asarray(cache[: min(200, written)], dtype=np.float32).ravel()
    print(f"[*] Layer-4 feature distribution (n={sample.size:,}):")
    print(f"    min={sample.min():.4f}  max={sample.max():.4f}  mean={sample.mean():.4f}")
    for q in (50, 90, 99, 99.9, 99.99):
        print(f"    p{q:<6} {np.percentile(sample, q):.4f}")

    if written < len(images):
        print(f"[!] {len(images) - written} images failed to load; cache tail is zeros.")
        print(f"    Pass --limit {written} when training, or re-run with a clean folder.")


if __name__ == "__main__":
    main()
