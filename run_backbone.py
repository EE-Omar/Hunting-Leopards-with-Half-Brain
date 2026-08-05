"""
Pi entry point — run backbone inference and stream tensors to laptop.

Usage:
    python3 run_backbone.py

Edit config_backbone.yaml to change model, source, or network settings.
"""

import yaml
import sys
from pathlib import Path

def load_config():
    config_path = Path("config_backbone.yaml")
    if not config_path.exists():
        print("[!] config_backbone.yaml not found. Are you in the project root?")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()

    backbone_path = cfg["model"]["backbone"]
    imgsz        = cfg["model"]["imgsz"]
    client_ip    = cfg["network"]["head_ip"]
    port         = cfg["network"]["port"]
    source_type  = cfg["source"]["type"]
    images_path  = cfg["source"].get("images_path", None)
    camera_device= cfg["source"].get("camera_device", 0)
    fps          = cfg["source"].get("fps", 2)

    print(f"[*] Config loaded")
    print(f"    Backbone : {backbone_path}")
    print(f"    Image sz : {imgsz}x{imgsz}")
    print(f"    Head IP  : {client_ip}:{port}")
    print(f"    Source   : {source_type}")

    if not Path(backbone_path).exists():
        print(f"[!] Backbone model not found: {backbone_path}")
        print(f"    Run: python3 scripts/create_split.py to generate split models")
        sys.exit(1)

    sys.argv = ["backbone_server.py", backbone_path, client_ip,
                "--port", str(port),
                "--imgsz", str(imgsz)]

    bottleneck = cfg.get("bottleneck", {})
    if bottleneck.get("enabled"):
        quant_meta = bottleneck.get("quant_meta")
        if not quant_meta or not Path(quant_meta).exists():
            print(f"[!] bottleneck.enabled is true but quant_meta not found: {quant_meta}")
            print(f"    Run: python scripts/bottleneck/export_bottleneck.py")
            sys.exit(1)
        print(f"    Bottleneck: {quant_meta} (uint8 wire)")
        sys.argv += ["--quant-meta", quant_meta]

    if source_type == "images":
        if not images_path:
            print("[!] source.images_path not set in config_backbone.yaml")
            sys.exit(1)
        sys.argv += ["--images", images_path, "--fps", str(fps)]
    elif source_type == "camera":
        sys.argv += ["--camera", str(camera_device)]
    else:
        print(f"[!] Unknown source type: {source_type}. Use 'images' or 'camera'")
        sys.exit(1)

    import scripts.backbone_server as backbone
    backbone.main()

if __name__ == "__main__":
    main()
