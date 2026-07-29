"""
Laptop entry point — receive tensors from Pi, run head inference, decode detections.

Usage:
    python run_head.py

Edit config_head.yaml to change model, port, or output options.
"""

import yaml
import sys
from pathlib import Path

def load_config():
    config_path = Path("config_head.yaml")
    if not config_path.exists():
        print("[!] config_head.yaml not found. Are you in the project root?")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()

    head_path = cfg["model"]["head"]
    imgsz     = cfg["model"]["imgsz"]
    port      = cfg["network"]["port"]
    conf      = cfg["detection"]["conf_threshold"]

    print(f"[*] Config loaded")
    print(f"    Head     : {head_path}")
    print(f"    Image sz : {imgsz}x{imgsz}")
    print(f"    Port     : {port}")
    print(f"    Conf thr : {conf}")

    if not Path(head_path).exists():
        print(f"[!] Head model not found: {head_path}")
        print(f"    Run: python scripts/create_split.py to generate split models")
        sys.exit(1)

    sys.argv = ["head_client.py", head_path,
                "--port", str(port),
                "--imgsz", str(imgsz),
                "--conf", str(conf)]

    import scripts.head_client as head
    head.main()

if __name__ == "__main__":
    main()
