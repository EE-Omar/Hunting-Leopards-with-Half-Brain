"""
Export the trained bottleneck to deployable ONNX.

Rather than rebuilding the neck/head graph, this exports only the tiny encoder and
decoder and composes them onto the already-verified split artifacts produced by
scripts/create_split.py:

    backbone_enc_320.onnx = backbone_320.onnx  ->  encoder.onnx
    dec_head_320.onnx     = decoder.onnx       ->  head_320.onnx

Note what is NOT here: int8 export. The encoder and decoder stay FP32 ONNX and the
uint8 conversion happens in numpy on the wire (see backbone_server.send_tensor). That
sidesteps the broken Ultralytics INT8 export path entirely — the quantization the model
was trained against is a tensor format, not a model format.

Usage:
    python scripts/bottleneck/export_bottleneck.py --ckpt models/bottleneck/bneck_16ch.pt
"""

import argparse
import json
import sys
from pathlib import Path

import onnx
import torch
from onnx import compose

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# SPLIT_TENSOR is the tensor name create_split.py cut on; both split halves use it as
# their boundary.
from scripts.bottleneck.modules import SPLIT_TENSOR, load_bottleneck  # noqa: E402


def export_part(module, dummy, path, input_name, output_name, opset):
    """Export one nn.Module to ONNX with fixed, explicitly named IO."""
    torch.onnx.export(
        module,
        dummy,
        str(path),
        input_names=[input_name],
        output_names=[output_name],
        opset_version=opset,
        dynamo=False,
    )
    return onnx.load(str(path))


def default_opset(model):
    """Opset of the default ONNX domain. merge_models rejects mismatched opsets, so the
    encoder/decoder must be exported at whatever create_split.py produced."""
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return entry.version
    raise RuntimeError("no default-domain opset found")


def rename_tensors(model, mapping):
    """Rename tensors everywhere they appear in the graph."""
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.name in mapping:
            vi.name = mapping[vi.name]
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name in mapping:
                node.input[i] = mapping[name]
        for i, name in enumerate(node.output):
            if name in mapping:
                node.output[i] = mapping[name]
    for init in model.graph.initializer:
        if init.name in mapping:
            init.name = mapping[init.name]
    return model


def compose_models(first, second, io_map, path, input_name, output_name):
    """Merge two ONNX graphs.

    Prefixing avoids name collisions during the merge, but it also renames the public
    inputs/outputs (images -> a_images). The rest of the repo refers to `images` and
    `output0`, so strip the prefixes back off the graph boundary afterwards.
    """
    first = compose.add_prefix(first, prefix="a_")
    second = compose.add_prefix(second, prefix="b_")
    mapped = [(f"a_{o}", f"b_{i}") for o, i in io_map]
    merged = compose.merge_models(first, second, io_map=mapped)

    rename_tensors(merged, {
        merged.graph.input[0].name: input_name,
        merged.graph.output[0].name: output_name,
    })

    onnx.checker.check_model(merged)
    onnx.save(merged, str(path))
    return merged


def main():
    parser = argparse.ArgumentParser(description="Export + compose the bottleneck ONNX models")
    parser.add_argument("--ckpt", default=str(ROOT / "models/bottleneck/bneck_16ch.pt"))
    parser.add_argument("--backbone", default=str(ROOT / "models/split/backbone_320.onnx"))
    parser.add_argument("--head", default=str(ROOT / "models/split/head_320.onnx"))
    parser.add_argument("--outdir", default=str(ROOT / "models/split"))
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--opset", type=int, default=None,
                        help="Default: match the opset of the split models being composed with")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tmpdir = outdir / "_bottleneck_parts"
    tmpdir.mkdir(parents=True, exist_ok=True)

    backbone = onnx.load(args.backbone)
    head = onnx.load(args.head)
    opset = args.opset or default_opset(backbone)
    if default_opset(head) != opset:
        raise SystemExit(
            f"[!] Opset mismatch: backbone={opset}, head={default_opset(head)}. "
            "Re-run scripts/create_split.py so both halves match."
        )

    grid = args.imgsz // 8
    block = load_bottleneck(args.ckpt)
    block.eval()
    scale, zero_point = block.quant.qparams()

    print(f"[*] Bottleneck: {block.ch_in} -> {block.ch_bottleneck} ch, kernel {block.kernel}")
    print(f"[*] Wire quant: scale={scale.item():.6g}  zero_point={zero_point.item():.0f}")
    print(f"[*] Opset: {opset} (matched to the split models)")

    # --- encoder ---------------------------------------------------------------
    enc_path = tmpdir / "encoder.onnx"
    enc = export_part(
        block.encoder,
        torch.zeros(1, block.ch_in, grid, grid),
        enc_path,
        input_name=SPLIT_TENSOR,
        output_name="bottleneck",
        opset=opset,
    )
    print(f"[+] encoder.onnx  ({block.ch_in},{grid},{grid}) -> ({block.ch_bottleneck},{grid},{grid})")

    # --- decoder ---------------------------------------------------------------
    dec_path = tmpdir / "decoder.onnx"
    dec = export_part(
        block.decoder,
        torch.zeros(1, block.ch_bottleneck, grid, grid),
        dec_path,
        input_name="bottleneck",
        output_name=SPLIT_TENSOR,
        opset=opset,
    )
    print(f"[+] decoder.onnx  ({block.ch_bottleneck},{grid},{grid}) -> ({block.ch_in},{grid},{grid})")

    # --- compose onto the existing verified split halves ------------------------
    be_path = outdir / f"backbone_enc_{args.imgsz}.onnx"
    compose_models(backbone, enc, [(SPLIT_TENSOR, SPLIT_TENSOR)], be_path,
                   input_name="images", output_name="bottleneck")
    print(f"[+] {be_path.name}  images -> bottleneck")

    dh_path = outdir / f"dec_head_{args.imgsz}.onnx"
    compose_models(dec, head, [(SPLIT_TENSOR, SPLIT_TENSOR)], dh_path,
                   input_name="bottleneck", output_name="output0")
    print(f"[+] {dh_path.name}  bottleneck -> output0")

    # --- sidecar with everything the wire needs ---------------------------------
    meta = {
        "imgsz": args.imgsz,
        "channels": block.ch_bottleneck,
        "grid": grid,
        "scale": float(scale.item()),
        "zero_point": int(zero_point.item()),
        "wire_bytes": int(block.wire_bytes(grid)),
        "backbone_enc": be_path.name,
        "dec_head": dh_path.name,
    }
    meta_path = outdir / f"bottleneck_{args.imgsz}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[+] {meta_path.name}  (scale/zero_point for the uint8 wire)")

    print(f"\n[*] Wire payload: {meta['wire_bytes'] / 1024:.1f} KB/frame")
    print(f"[*] Verify: python scripts/bottleneck/verify_bottleneck.py data/images/ --imgsz {args.imgsz}")


if __name__ == "__main__":
    main()
