import argparse
import subprocess
import sys
import time
from pathlib import Path


def load_gpu_count(gpu_json_path: Path) -> int:
    import json

    data = json.loads(gpu_json_path.read_text(encoding="utf-8"))
    gpu_cards = data.get("gpu_cards", [])
    return len(gpu_cards)


def make_worker_names(worker_count: int) -> list[str]:
    return [f"worker{i}" for i in range(worker_count)]


def build_worker_command(
    python_exe: str,
    rank: int,
    world_size: int,
    worker_name: str,
    master_addr: str,
    master_port: int,
) -> list[str]:
    return [
        python_exe,
        "-u",
        "cnnSplit/model_parallel_rpc.py",
        "--role",
        "worker",
        "--name",
        worker_name,
        "--rank",
        str(rank),
        "--world-size",
        str(world_size),
        "--master-addr",
        master_addr,
        "--master-port",
        str(master_port),
    ]


def build_coordinator_command(
    python_exe: str,
    world_size: int,
    worker_names: list[str],
    master_addr: str,
    master_port: int,
    gpu_json_path: Path,
    batch_size: int,
) -> list[str]:
    return [
        python_exe,
        "-u",
        "cnnSplit/model_parallel_rpc.py",
        "--role",
        "coordinator",
        "--rank",
        "0",
        "--world-size",
        str(world_size),
        "--master-addr",
        master_addr,
        "--master-port",
        str(master_port),
        "--gpu-json",
        str(gpu_json_path),
        "--worker-names",
        ",".join(worker_names),
        "--batch-size",
        str(batch_size),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local multi-process RPC demo for model-parallel CNN chunks."
    )
    parser.add_argument(
        "--gpu-json",
        default="data/gpu/receiver_gpu_store.json",
        help="Path to GPU summary JSON used by the coordinator.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker count override. If omitted or <= 0, inferred from gpu_json.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to run child processes.",
    )
    parser.add_argument("--master-addr", default="127.0.0.1", help="RPC master address.")
    parser.add_argument("--master-port", type=int, default=29501, help="RPC master port.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Synthetic input batch size for the coordinator forward pass.",
    )
    parser.add_argument(
        "--startup-wait-seconds",
        type=float,
        default=2.0,
        help="Delay before starting coordinator after worker launch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    gpu_json_path = Path(args.gpu_json)
    if not gpu_json_path.is_absolute():
        gpu_json_path = repo_root / gpu_json_path
    if not gpu_json_path.exists():
        raise FileNotFoundError(f"GPU JSON file not found: {gpu_json_path}")

    worker_count = args.workers if args.workers > 0 else load_gpu_count(gpu_json_path)
    if worker_count <= 0:
        raise ValueError("Worker count is 0. Provide --workers or a gpu_json with gpu_cards.")

    worker_names = make_worker_names(worker_count)
    world_size = worker_count + 1

    worker_processes: list[subprocess.Popen] = []
    try:
        for rank, worker_name in enumerate(worker_names, start=1):
            worker_cmd = build_worker_command(
                python_exe=args.python,
                rank=rank,
                world_size=world_size,
                worker_name=worker_name,
                master_addr=args.master_addr,
                master_port=args.master_port,
            )
            print(f"[launcher] Starting {worker_name}: {' '.join(worker_cmd)}")
            proc = subprocess.Popen(worker_cmd, cwd=str(repo_root))
            worker_processes.append(proc)

        time.sleep(max(0.0, args.startup_wait_seconds))

        coordinator_cmd = build_coordinator_command(
            python_exe=args.python,
            world_size=world_size,
            worker_names=worker_names,
            master_addr=args.master_addr,
            master_port=args.master_port,
            gpu_json_path=gpu_json_path,
            batch_size=args.batch_size,
        )
        print(f"[launcher] Starting coordinator: {' '.join(coordinator_cmd)}")
        coordinator_result = subprocess.run(coordinator_cmd, cwd=str(repo_root), check=False)
        return coordinator_result.returncode
    finally:
        for proc in worker_processes:
            if proc.poll() is None:
                proc.terminate()

        for proc in worker_processes:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
