"""
Stage 1: train the layer-3 bottleneck to reconstruct its own input.

Self-supervised — no labels, no dataloader surgery, and the frozen YOLO never runs in
the training loop (features come from the cache built by cache_features.py). That is
what makes this tractable on CPU.

If the resulting mAP drop is unacceptable, Stage 2 is a detection-loss finetune; see
the plan. Start here, because this answers "how far can 16 channels go" cheaply.

Usage:
    python scripts/bottleneck/train_bottleneck.py --channels 16 --epochs 30
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.bottleneck.modules import BottleneckBlock, save_bottleneck  # noqa: E402


def batches(data, indices, batch_size, shuffle=True, rng=None):
    """Yield fp32 tensors from an fp16 memmap, without loading it all into RAM."""
    idx = indices.copy()
    if shuffle:
        (rng or np.random).shuffle(idx)
    for start in range(0, len(idx), batch_size):
        chunk = np.sort(idx[start:start + batch_size])  # sorted -> sequential memmap reads
        yield torch.from_numpy(np.asarray(data[chunk], dtype=np.float32))


@torch.no_grad()
def evaluate(block, data, indices, batch_size):
    """Reconstruction quality on held-out features."""
    block.eval()
    total_mse = total_rel = total_cos = 0.0
    n = 0
    for x in batches(data, indices, batch_size, shuffle=False):
        y = block(x)
        b = x.shape[0]
        total_mse += nn.functional.mse_loss(y, x).item() * b
        # Relative L2 is the scale-free number to quote; MSE alone is hard to read.
        total_rel += (torch.norm(y - x) / torch.norm(x)).item() * b
        total_cos += torch.nn.functional.cosine_similarity(
            y.flatten(2), x.flatten(2), dim=1
        ).mean().item() * b
        n += b
    return total_mse / n, total_rel / n, total_cos / n


def main():
    parser = argparse.ArgumentParser(description="Train the layer-3 reconstruction bottleneck")
    parser.add_argument("--feats", default=str(ROOT / "models/bottleneck/feats_320.npy"))
    parser.add_argument("--out", default=None, help="Checkpoint path (default: derived from --channels)")
    parser.add_argument("--channels", type=int, default=16, help="Bottleneck width")
    parser.add_argument("--kernel", type=int, default=1, choices=[1, 3],
                        help="Encoder kernel. 1x1 is ~2.1M MACs; 3x3 is ~18.9M (fallback if 1x1 underfits)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--no-quant", action="store_true", help="Train without the uint8 simulation")
    parser.add_argument("--quant-warmup", type=int, default=1,
                        help="Epochs to train before enabling the uint8 simulation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = np.load(args.feats, mmap_mode="r")
    n_total, ch_in, gh, gw = data.shape
    print(f"[*] Features: {data.shape} fp16 from {Path(args.feats).name}")

    perm = rng.permutation(n_total)
    n_val = max(1, int(n_total * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"[*] Split: {len(train_idx)} train / {len(val_idx)} val")

    block = BottleneckBlock(ch_in=ch_in, ch_bottleneck=args.channels,
                            kernel=args.kernel, quantize=not args.no_quant)
    n_params = sum(p.numel() for p in block.parameters())
    enc_params = sum(p.numel() for p in block.encoder.parameters())
    wire_kb = block.wire_bytes(gh) / 1024
    raw_kb = ch_in * gh * gw * 4 / 1024
    print(f"[*] Bottleneck: {ch_in} -> {args.channels} ch, {n_params:,} params "
          f"({enc_params:,} in the encoder)")
    print(f"[*] Wire: {raw_kb:.0f} KB fp32  ->  {wire_kb:.0f} KB uint8  "
          f"({raw_kb / wire_kb:.0f}x smaller)")

    optimizer = torch.optim.AdamW(block.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_path = Path(args.out) if args.out else ROOT / f"models/bottleneck/bneck_{args.channels}ch.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_rel = float("inf")
    print(f"\n{'epoch':>5} {'train mse':>11} {'val mse':>11} {'val rel L2':>11} "
          f"{'val cos':>9} {'lr':>9} {'sec':>6}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        # Let the encoder find a sane output range before quantizing it.
        block.quantize = (not args.no_quant) and epoch > args.quant_warmup

        block.train()
        t0 = time.time()
        running, seen = 0.0, 0
        for x in batches(data, train_idx, args.batch, shuffle=True, rng=rng):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(block(x), x)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.shape[0]
            seen += x.shape[0]
        scheduler.step()

        # Always report val with quantization on — that is the deployed configuration.
        was = block.quantize
        block.quantize = not args.no_quant
        val_mse, val_rel, val_cos = evaluate(block, data, val_idx, args.batch)
        block.quantize = was

        flag = ""
        if val_rel < best_rel:
            best_rel = val_rel
            save_bottleneck(block, out_path)
            flag = " *"

        print(f"{epoch:>5} {running / seen:>11.6f} {val_mse:>11.6f} {val_rel:>11.4f} "
              f"{val_cos:>9.4f} {scheduler.get_last_lr()[0]:>9.2e} {time.time() - t0:>6.1f}{flag}")

    print("-" * 70)
    print(f"[+] Best val relative L2: {best_rel:.4f}")
    print(f"[+] Saved: {out_path}")
    block_scale, block_zp = block.quant.qparams()
    print(f"[*] Wire quant params: scale={block_scale.item():.6g} zero_point={block_zp.item():.0f}")
    print(f"\n[*] Next: python scripts/bottleneck/eval_bottleneck.py --ckpt {out_path} --imgsz 256")


if __name__ == "__main__":
    main()
