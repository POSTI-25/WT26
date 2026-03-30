import argparse
import json
from pathlib import Path

from receiver_service import get_gpu_report, get_local_ip, print_gpu_report, write_gpu_store

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_GPU_STORE_FILE = PROJECT_ROOT / "data" / "gpu" / "receiver_gpu_store.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print local contributor GPU details.")
    parser.add_argument(
        "--gpu-store-file",
        default=str(DEFAULT_GPU_STORE_FILE),
        help=f"Path to write receiver GPU summary JSON (default: {DEFAULT_GPU_STORE_FILE}).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP hint used for storing local ip_address in GPU summary (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--no-write-store",
        action="store_true",
        help="Print GPU info without writing the GPU summary JSON file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of human-readable output.",
    )
    args = parser.parse_args()

    report = get_gpu_report()
    if not args.no_write_store:
        local_ip = get_local_ip(args.host)
        gpu_store_path = Path(args.gpu_store_file).expanduser()
        if not gpu_store_path.is_absolute():
            gpu_store_path = PROJECT_ROOT / gpu_store_path
        write_gpu_store(report, gpu_store_path, local_ip)
        print(f"[receiver] Stored GPU summary at {gpu_store_path}")

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print_gpu_report(report)


if __name__ == "__main__":
    main()
