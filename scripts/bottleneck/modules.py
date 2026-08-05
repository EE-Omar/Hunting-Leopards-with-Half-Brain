"""
Learned bottleneck inserted at the YOLO26 layer-4 split point.

The split tensor is (1, 128, 32, 32) at imgsz=256 — 512 KB/frame as fp32, which is
what currently makes split inference lose to running the whole model on-device. This
module squeezes it to (1, 16, 32, 32) and quantizes to uint8 on the wire: 16 KB/frame.

Layer 4 feeds BOTH layer 5 and the layer-15 P3 concat skip, so wrapping layer 4 (whose
output is the tensor Ultralytics saves) puts the bottleneck on both paths automatically.

Design constraints:
  - The encoder runs on the field device, so it must be nearly free. Layers 0-4 are only
    ~75K params but are FLOP-heavy because they run at high spatial resolution. A 1x1
    conv at the 32x32 grid costs ~2.1M MACs — a rounding error next to the backbone.
  - The decoder runs on the head device and can afford more.
  - Quantization is simulated during training (straight-through estimator) so the network
    learns to tolerate the exact uint8 round-trip the wire performs.
"""

import numpy as np
import torch
import torch.nn as nn

# Layer index of the split point in the Ultralytics DetectionModel graph.
SPLIT_LAYER = 4
SPLIT_CHANNELS = 128


def quantize_uint8(x, scale, zero_point):
    """fp32 -> uint8, matching FakeQuant exactly. This is what goes on the wire."""
    q = np.round(np.asarray(x, dtype=np.float32) / scale + zero_point)
    return np.clip(q, 0, 255).astype(np.uint8)


def dequantize_uint8(q, scale, zero_point):
    """uint8 -> fp32, the head-side inverse of `quantize_uint8`."""
    return (q.astype(np.float32) - zero_point) * scale


class FakeQuant(nn.Module):
    """Simulated uint8 affine quantization with a straight-through estimator.

    Calibrates from percentiles rather than min/max. The encoder output is unbounded and
    long-tailed; a single outlier calibrated with min/max would consume most of the 256
    available levels and leave the bulk of the distribution with almost no resolution.

    The (scale, zero_point) this produces are exactly the values the wire quantizer in
    backbone_server.py must use, so they are exported alongside the weights.
    """

    def __init__(self, num_bits=8, percentile=99.9, momentum=0.99):
        super().__init__()
        self.num_bits = num_bits
        self.qmax = 2**num_bits - 1
        self.percentile = percentile
        self.momentum = momentum
        # Buffers so the calibrated range travels with the checkpoint.
        self.register_buffer("lo", torch.tensor(0.0))
        self.register_buffer("hi", torch.tensor(0.0))
        self.register_buffer("calibrated", torch.tensor(0, dtype=torch.uint8))

    @torch.no_grad()
    def _observe(self, x):
        """EMA-update the calibrated range from the current batch."""
        flat = x.detach().flatten().float()
        if flat.numel() > 1_000_000:  # torch.quantile has a hard size ceiling
            idx = torch.randperm(flat.numel(), device=flat.device)[:1_000_000]
            flat = flat[idx]

        p = self.percentile / 100.0
        lo = torch.quantile(flat, 1.0 - p)
        hi = torch.quantile(flat, p)

        if self.calibrated.item() == 0:
            self.lo.copy_(lo)
            self.hi.copy_(hi)
            self.calibrated.fill_(1)
        else:
            m = self.momentum
            self.lo.mul_(m).add_(lo * (1.0 - m))
            self.hi.mul_(m).add_(hi * (1.0 - m))

    def qparams(self):
        """Return (scale, zero_point) as plain floats for the wire protocol."""
        scale = torch.clamp((self.hi - self.lo) / self.qmax, min=1e-8)
        zero_point = torch.round(-self.lo / scale).clamp(0, self.qmax)
        return scale, zero_point

    def forward(self, x):
        if self.training:
            self._observe(x)
        if self.calibrated.item() == 0:
            return x  # nothing observed yet — pass through
        scale, zero_point = self.qparams()
        q = torch.clamp(torch.round(x / scale + zero_point), 0, self.qmax)
        dequant = (q - zero_point) * scale
        return x + (dequant - x).detach()  # straight-through estimator


class Encoder(nn.Module):
    """Field-device half. Kept to a single conv on purpose — see module docstring."""

    def __init__(self, ch_in=SPLIT_CHANNELS, ch_bottleneck=16, kernel=1):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_bottleneck, kernel, padding=kernel // 2, bias=True)

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    """Head-device half. Expand back to 128ch, then a residual 3x3 refinement."""

    def __init__(self, ch_bottleneck=16, ch_out=SPLIT_CHANNELS):
        super().__init__()
        self.expand = nn.Conv2d(ch_bottleneck, ch_out, 1, bias=True)
        self.act = nn.SiLU()
        self.refine = nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=True)

    def forward(self, x):
        y = self.expand(x)
        return y + self.refine(self.act(y))


class BottleneckBlock(nn.Module):
    """encode -> (fake) quantize -> decode, operating on the layer-4 feature map."""

    def __init__(self, ch_in=SPLIT_CHANNELS, ch_bottleneck=16, kernel=1, quantize=True):
        super().__init__()
        self.ch_in = ch_in
        self.ch_bottleneck = ch_bottleneck
        self.kernel = kernel
        self.quantize = quantize
        self.encoder = Encoder(ch_in, ch_bottleneck, kernel)
        self.quant = FakeQuant()
        self.decoder = Decoder(ch_bottleneck, ch_in)

    def forward(self, x):
        z = self.encoder(x)
        if self.quantize:
            z = self.quant(z)
        return self.decoder(z)

    def config(self):
        """Hyperparameters needed to rebuild this block from a checkpoint."""
        return {
            "ch_in": self.ch_in,
            "ch_bottleneck": self.ch_bottleneck,
            "kernel": self.kernel,
            "quantize": self.quantize,
        }

    def wire_bytes(self, spatial=32):
        """Bytes actually placed on the wire per frame, at uint8."""
        return self.ch_bottleneck * spatial * spatial


class Layer4WithBottleneck(nn.Module):
    """Wraps the original layer-4 module so the bottleneck rides on its output.

    Ultralytics' `_predict_once` reads `.f` and `.i` off every layer, and `fuse()` walks
    `.modules()` recursively — so copying those two attributes across is all that is
    needed for `model.val()`, `model.train()` and `model.fuse()` to keep working.
    """

    def __init__(self, orig, bottleneck):
        super().__init__()
        self.orig = orig
        self.bottleneck = bottleneck
        # Attributes Ultralytics' graph walker depends on.
        self.f = orig.f
        self.i = orig.i
        self.type = getattr(orig, "type", type(orig).__name__)
        self.np = getattr(orig, "np", sum(p.numel() for p in orig.parameters()))

    def forward(self, x):
        return self.bottleneck(self.orig(x))


def wrap_layer4(net, bottleneck):
    """Insert `bottleneck` after layer 4 of a DetectionModel, in place.

    Args:
        net: the inner `DetectionModel` (i.e. `YOLO(...).model`).
        bottleneck: a `BottleneckBlock`.

    Returns:
        The same `net`, mutated.
    """
    current = net.model[SPLIT_LAYER]
    if isinstance(current, Layer4WithBottleneck):
        current.bottleneck = bottleneck  # already wrapped — just swap the block
        return net
    net.model[SPLIT_LAYER] = Layer4WithBottleneck(current, bottleneck)
    return net


def backbone_stub(net):
    """Layers 0-4 as a standalone Sequential, for feature extraction.

    Valid because every layer up to and including the split point has `f == -1`, so there
    are no skip connections to replay.
    """
    for i in range(SPLIT_LAYER + 1):
        layer = net.model[i]
        if layer.f != -1:
            raise RuntimeError(
                f"layer {i} takes input from {layer.f}, not the previous layer — "
                "backbone_stub() assumes a straight chain up to the split point"
            )
    return nn.Sequential(*[net.model[i] for i in range(SPLIT_LAYER + 1)])


def save_bottleneck(bottleneck, path):
    """Persist weights plus the config needed to rebuild the block."""
    torch.save(
        {"config": bottleneck.config(), "state_dict": bottleneck.state_dict()},
        str(path),
    )


def load_bottleneck(path, map_location="cpu"):
    """Rebuild a BottleneckBlock from a checkpoint written by `save_bottleneck`."""
    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    block = BottleneckBlock(**ckpt["config"])
    block.load_state_dict(ckpt["state_dict"])
    block.eval()
    return block
