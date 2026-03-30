from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn


RPC_REQUIRED_SYMBOLS = ("init_rpc", "shutdown", "remote")


def rpc_is_available() -> bool:
    return all(hasattr(rpc, symbol) for symbol in RPC_REQUIRED_SYMBOLS)


def ensure_rpc_available() -> None:
    if rpc_is_available():
        return

    raise RuntimeError(
        "PyTorch RPC APIs are not available in this environment. "
        f"Installed torch version: {torch.__version__}. "
        "Use split-only test via `python -m cnnSplit.test_train_split --gpu-json <path>` "
        "or install a PyTorch build that includes torch.distributed.rpc support."
    )


def load_gpu_json(gpu_json: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(gpu_json, dict):
        return gpu_json

    path = Path(gpu_json)
    return json.loads(path.read_text(encoding="utf-8"))


def compute_gpu_scores(
    gpu_json: dict[str, Any] | str | Path,
    worker_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    data = load_gpu_json(gpu_json)
    cards = data.get("gpu_cards", [])
    if not cards:
        raise ValueError("No GPU cards found in gpu_json['gpu_cards'].")

    if worker_names is None:
        worker_names = [f"worker{i}" for i in range(len(cards))]
    if len(worker_names) != len(cards):
        raise ValueError(
            f"worker_names length ({len(worker_names)}) must match gpu_cards length ({len(cards)})."
        )

    workers: list[dict[str, Any]] = []
    for i, card in enumerate(cards):
        cores = float(card.get("cuda_cores") or 0.0)
        usage = float(card.get("gpu_usage_percent") or 0.0)
        usage = max(0.0, min(100.0, usage))
        score = cores * (1.0 - usage / 100.0)

        workers.append(
            {
                "worker": worker_names[i],
                "gpu_index": card.get("index", i),
                "gpu_card": card.get("gpu_card", "unknown"),
                "cuda_cores": cores,
                "gpu_usage_percent": usage,
                "compute_score": score,
            }
        )

    total_score = sum(w["compute_score"] for w in workers)
    if total_score <= 0:
        uniform_share = 1.0 / len(workers)
        for w in workers:
            w["normalized_share"] = uniform_share
    else:
        for w in workers:
            w["normalized_share"] = w["compute_score"] / total_score

    return workers


def extract_sequential_layers(model: nn.Module) -> list[nn.Module]:
    if isinstance(model, nn.Sequential):
        return list(model.children())

    if hasattr(model, "model") and isinstance(model.model, nn.Sequential):
        return list(model.model.children())

    children = list(model.children())
    if len(children) == 1 and isinstance(children[0], nn.Sequential):
        return list(children[0].children())

    raise ValueError(
        "Model must be nn.Sequential or expose a top-level nn.Sequential via model.model."
    )


def greedy_contiguous_layer_counts(num_layers: int, shares: list[float]) -> list[int]:
    if num_layers < 0:
        raise ValueError("num_layers must be non-negative")
    if not shares:
        raise ValueError("shares cannot be empty")

    worker_count = len(shares)
    counts = [0 for _ in range(worker_count)]

    if num_layers == 0:
        return counts

    target = [s * num_layers for s in shares]

    seed_workers = min(num_layers, worker_count)
    for idx in range(seed_workers):
        counts[idx] = 1

    assigned = seed_workers
    while assigned < num_layers:
        deficits = [target[i] - counts[i] for i in range(worker_count)]
        best_idx = max(range(worker_count), key=lambda i: (deficits[i], shares[i]))
        counts[best_idx] += 1
        assigned += 1

    return counts


def build_contiguous_layer_assignment(
    model: nn.Module,
    workers: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    layers = extract_sequential_layers(model)
    shares = [float(w["normalized_share"]) for w in workers]
    counts = greedy_contiguous_layer_counts(len(layers), shares)

    assignment: dict[str, dict[str, Any]] = {}
    start = 0
    for idx, worker in enumerate(workers):
        count = counts[idx]
        end = start + count
        assignment[worker["worker"]] = {
            "layer_indices": list(range(start, end)),
            "layers": layers[start:end],
        }
        start = end

    return assignment


class RemoteChunkExecutor:
    def __init__(self, layers: list[nn.Module], preferred_device: str | None = None):
        if preferred_device is None:
            preferred_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if preferred_device.startswith("cuda") and not torch.cuda.is_available():
            preferred_device = "cpu"

        self.device = torch.device(preferred_device)
        self.chunk = nn.Sequential(*layers).to(self.device)
        self.chunk.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = self.chunk(x)
        return y.cpu()


def _create_remote_chunk(
    layers: list[nn.Module], preferred_device: str | None = None
) -> RemoteChunkExecutor:
    return RemoteChunkExecutor(layers, preferred_device)


def deploy_model_chunks_rpc(
    assignment: dict[str, dict[str, Any]],
    worker_devices: dict[str, str] | None = None,
) -> dict[str, rpc.RRef]:
    ensure_rpc_available()

    worker_devices = worker_devices or {}
    remote_chunks: dict[str, rpc.RRef] = {}

    for worker_name, payload in assignment.items():
        layers = payload["layers"]
        if not layers:
            continue
        preferred_device = worker_devices.get(worker_name)
        remote_chunks[worker_name] = rpc.remote(
            worker_name,
            _create_remote_chunk,
            args=(layers, preferred_device),
        )

    return remote_chunks


def distributed_forward(
    x: torch.Tensor,
    assignment_order: list[str],
    remote_chunks: dict[str, rpc.RRef],
) -> torch.Tensor:
    out = x
    for worker_name in assignment_order:
        chunk_ref = remote_chunks.get(worker_name)
        if chunk_ref is None:
            continue
        out = chunk_ref.rpc_sync().forward(out)
    return out


def init_rpc_framework(
    name: str,
    rank: int,
    world_size: int,
    master_addr: str = "127.0.0.1",
    master_port: int = 29501,
) -> None:
    ensure_rpc_available()

    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))

    if hasattr(rpc, "TensorPipeRpcBackendOptions"):
        options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=16,
            rpc_timeout=300,
        )
        rpc.init_rpc(
            name=name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options,
        )
        return

    rpc.init_rpc(
        name=name,
        rank=rank,
        world_size=world_size,
    )


def split_model_and_deploy(
    model: nn.Module,
    gpu_json: dict[str, Any] | str | Path,
    worker_names: list[str] | None = None,
    deploy_rpc: bool = False,
    worker_devices: dict[str, str] | None = None,
) -> dict[str, Any]:
    workers = compute_gpu_scores(gpu_json, worker_names=worker_names)
    assignment = build_contiguous_layer_assignment(model, workers)

    print("\nGPU compute scores:")
    for w in workers:
        print(
            f"  {w['worker']}: score={w['compute_score']:.2f} "
            f"(cores={int(w['cuda_cores'])}, usage={w['gpu_usage_percent']:.1f}%)"
        )

    print("\nNormalized shares:")
    for w in workers:
        print(f"  {w['worker']}: share={w['normalized_share']:.4f}")

    print("\nLayer assignment (contiguous):")
    for worker_name, payload in assignment.items():
        print(f"  {worker_name}: layers={payload['layer_indices']}")

    remote_chunks: dict[str, rpc.RRef] = {}
    if deploy_rpc:
        remote_chunks = deploy_model_chunks_rpc(assignment, worker_devices=worker_devices)

    layer_mapping = {
        worker_name: payload["layers"]
        for worker_name, payload in assignment.items()
    }
    layer_index_mapping = {
        worker_name: payload["layer_indices"]
        for worker_name, payload in assignment.items()
    }
    assignment_order = [w["worker"] for w in workers]

    return {
        "workers": workers,
        "layer_mapping": layer_mapping,
        "layer_index_mapping": layer_index_mapping,
        "assignment_order": assignment_order,
        "remote_chunks": remote_chunks,
    }


def run_worker(
    name: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> None:
    init_rpc_framework(
        name=name,
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
    )
    print(f"[{name}] RPC worker initialized. Waiting for coordinator...")
    rpc.shutdown()
    print(f"[{name}] RPC shutdown complete.")


def run_coordinator(
    gpu_json_path: str,
    worker_names: list[str],
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    batch_size: int = 4,
) -> None:
    init_rpc_framework(
        name="coordinator",
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
    )

    try:
        try:
            from train import CNN
        except ImportError:
            from cnnSplit.train import CNN

        model = CNN().eval()
        plan = split_model_and_deploy(
            model=model,
            gpu_json=gpu_json_path,
            worker_names=worker_names,
            deploy_rpc=True,
        )

        x = torch.randn(batch_size, 1, 28, 28)
        y = distributed_forward(
            x,
            assignment_order=plan["assignment_order"],
            remote_chunks=plan["remote_chunks"],
        )
        print(f"\nDistributed forward output shape: {tuple(y.shape)}")
    finally:
        rpc.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a CNN model into contiguous chunks and execute with PyTorch RPC."
    )
    parser.add_argument(
        "--role",
        choices=["coordinator", "worker"],
        required=True,
        help="Process role in the RPC world.",
    )
    parser.add_argument("--name", default="worker0", help="RPC name for worker role.")
    parser.add_argument("--rank", type=int, required=True, help="Global RPC rank.")
    parser.add_argument("--world-size", type=int, required=True, help="RPC world size.")
    parser.add_argument("--master-addr", default="127.0.0.1", help="RPC master address.")
    parser.add_argument("--master-port", type=int, default=29501, help="RPC master port.")
    parser.add_argument(
        "--gpu-json",
        default="gpu_store.json",
        help="Path to GPU summary JSON for coordinator role.",
    )
    parser.add_argument(
        "--worker-names",
        default="",
        help="Comma-separated worker names for coordinator role (e.g., worker0,worker1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Synthetic input batch size for coordinator demo forward pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.role == "worker":
        run_worker(
            name=args.name,
            rank=args.rank,
            world_size=args.world_size,
            master_addr=args.master_addr,
            master_port=args.master_port,
        )
        return

    worker_names = [w.strip() for w in args.worker_names.split(",") if w.strip()]
    if not worker_names:
        data = load_gpu_json(args.gpu_json)
        worker_names = [f"worker{i}" for i, _ in enumerate(data.get("gpu_cards", []))]

    run_coordinator(
        gpu_json_path=args.gpu_json,
        worker_names=worker_names,
        rank=args.rank,
        world_size=args.world_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
