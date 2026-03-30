import argparse
import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ops.network_cli import (
    DEFAULT_CONFIG_PATH,
    load_cli_config,
    probe_receiver,
    scan_receivers,
)


DEFAULT_SCAN_STORE = Path("data/gpu/network_gpu_store.json")
DEFAULT_SPLIT_STORE = Path("data/gpu/distributed_receiver_gpu_store.json")
DEFAULT_PLAN_OUT = Path("data/gpu/distributed_split_plan.json")
DEFAULT_ASSETS = [
    "cnnSplit/train.py",
    "dataset/sign_language_mnist_train.csv",
    "dataset/sign_mnist_test.csv",
]


def receive_line(reader) -> str:
    line = reader.readline()
    if not line:
        raise ConnectionError("Connection closed while reading line.")
    return line.decode("utf-8").rstrip("\n")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def to_abs(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return repo_root() / p


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def discover_receivers_to_store(
    config: dict[str, Any],
    subnet: str | None,
    port: int | None,
    timeout: float | None,
    workers: int | None,
    host: str | None,
    direct: bool,
    scan_store_path: Path,
) -> dict[str, Any]:
    effective_subnet = subnet or config["discovery_subnet"]
    effective_port = int(port if port is not None else config["target_port"])
    effective_timeout = float(timeout if timeout is not None else config["scan_timeout_seconds"])
    effective_workers = int(workers if workers is not None else config["scan_max_workers"])

    target_host = (host or str(config.get("target_host", "")).strip()).strip()

    if direct:
        if not target_host:
            raise RuntimeError("Direct mode requires --host or target_host in config.")
        if target_host in ("127.0.0.1", "localhost"):
            raise RuntimeError(
                "Direct mode target_host is localhost. Set --host to your friend's machine "
                "hostname or IP, or update config/cli_config.json target_host."
            )

        candidate = probe_receiver(
            host=target_host,
            port=effective_port,
            timeout=max(effective_timeout, 1.0),
        )
        receivers = [candidate] if candidate is not None else []
        if not receivers:
            raise RuntimeError(
                f"Direct connect failed to {target_host}:{effective_port}. "
                "Ensure receiver_service is running and firewall allows the port."
            )
    else:
        receivers = scan_receivers(
            subnet=effective_subnet,
            port=effective_port,
            timeout=effective_timeout,
            max_workers=effective_workers,
            max_hosts=int(config["max_scan_hosts"]),
        )

    if not direct and not receivers:
        fallback_hosts: list[str] = []
        if host:
            fallback_hosts.append(host)

        cfg_target_host = str(config.get("target_host", "")).strip()
        if cfg_target_host and cfg_target_host not in ("127.0.0.1", "localhost"):
            fallback_hosts.append(cfg_target_host)

        tried: set[str] = set()
        for fallback_host in fallback_hosts:
            if fallback_host in tried:
                continue
            tried.add(fallback_host)
            print(f"[distributed] Scan found 0 receivers. Trying direct host fallback: {fallback_host}")
            candidate = probe_receiver(
                host=fallback_host,
                port=effective_port,
                timeout=max(effective_timeout, 1.0),
            )
            if candidate is not None:
                receivers.append(candidate)
                break

    store = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "subnet": effective_subnet,
        "port": effective_port,
        "receiver_count": len(receivers),
        "receivers": receivers,
    }
    write_json(scan_store_path, store)

    print(f"[distributed] Receivers found: {len(receivers)}")
    for item in receivers:
        print(
            "[distributed] {host}:{port} | gpus={count} | ok={ok}".format(
                host=item.get("host"),
                port=item.get("port"),
                count=item.get("gpu_count"),
                ok=item.get("ok"),
            )
        )
    print(f"[distributed] Stored scan results at: {scan_store_path}")
    return store


def worker_name_for(host: str, gpu_index: int) -> str:
    safe = host.replace(".", "_").replace("-", "_")
    return f"{safe}_gpu{gpu_index}"


def build_distributed_gpu_store(
    scan_store: dict[str, Any],
    split_store_path: Path,
) -> dict[str, Any]:
    receivers = scan_store.get("receivers", [])
    if not receivers:
        raise RuntimeError("No receivers available in scan store.")

    gpu_cards: list[dict[str, Any]] = []
    worker_names: list[str] = []
    worker_placement: list[dict[str, Any]] = []

    for receiver in receivers:
        host = str(receiver.get("host", "unknown"))
        gpu_report = receiver.get("gpu_report", {})
        gpus = gpu_report.get("gpus", []) if isinstance(gpu_report, dict) else []

        for gpu in gpus:
            gpu_index = int(gpu.get("index", 0))
            worker = worker_name_for(host, gpu_index)
            worker_names.append(worker)
            worker_placement.append(
                {
                    "worker": worker,
                    "host": host,
                    "port": int(receiver.get("port", 50051)),
                    "gpu_index": gpu_index,
                    "gpu_card": gpu.get("name"),
                }
            )
            gpu_cards.append(
                {
                    "index": gpu_index,
                    "gpu_card": gpu.get("name"),
                    "gpu_usage_percent": gpu.get("utilization_gpu_percent"),
                    "cuda_cores": gpu.get("cuda_cores"),
                }
            )

    if not gpu_cards:
        raise RuntimeError("Receivers were discovered, but no GPU entries were reported.")

    split_store = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "network_scan",
        "gpu_cards": gpu_cards,
        "worker_names": worker_names,
        "worker_placement": worker_placement,
    }
    write_json(split_store_path, split_store)
    print(f"[distributed] Stored split GPU store at: {split_store_path}")
    return split_store


def build_split_plan(
    split_store: dict[str, Any],
    plan_out_path: Path,
) -> dict[str, Any]:
    from cnnSplit.model_parallel_rpc import split_model_and_deploy
    from cnnSplit.train import CNN

    model = CNN().eval()
    worker_names = split_store.get("worker_names")
    if not isinstance(worker_names, list) or not worker_names:
        raise RuntimeError("split store missing worker_names.")

    plan = split_model_and_deploy(
        model=model,
        gpu_json=split_store,
        worker_names=worker_names,
        deploy_rpc=False,
    )

    placement_by_worker = {
        str(item["worker"]): item
        for item in split_store.get("worker_placement", [])
    }
    layer_index_mapping = plan["layer_index_mapping"]

    routes = []
    for worker, layers in layer_index_mapping.items():
        routes.append(
            {
                "worker": worker,
                "host": placement_by_worker.get(worker, {}).get("host"),
                "port": placement_by_worker.get(worker, {}).get("port"),
                "gpu_index": placement_by_worker.get(worker, {}).get("gpu_index"),
                "layer_indices": layers,
            }
        )

    serializable_plan = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workers": plan["workers"],
        "layer_index_mapping": layer_index_mapping,
        "routes": routes,
    }
    write_json(plan_out_path, serializable_plan)
    print(f"[distributed] Stored split plan at: {plan_out_path}")
    return serializable_plan


def send_file_to_receiver(host: str, port: int, file_path: Path, chunk_size: int) -> None:
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = file_path.stat().st_size
    metadata = {
        "filename": file_path.name,
        "size": file_size,
        "chunk_size": chunk_size,
    }

    with socket.create_connection((host, port), timeout=30) as sock:
        reader = sock.makefile("rb")
        sock.sendall((json.dumps(metadata) + "\n").encode("utf-8"))
        ack = receive_line(reader)
        if ack != "OK":
            raise RuntimeError(f"Receiver rejected metadata for {file_path.name}: {ack}")

        with file_path.open("rb") as in_file:
            while True:
                chunk = in_file.read(chunk_size)
                if not chunk:
                    break
                sock.sendall(chunk)

        reply_line = receive_line(reader)
        reply = json.loads(reply_line)
        if not reply.get("ok", False):
            raise RuntimeError(
                f"Transfer failed for {file_path.name} -> {host}:{port}: "
                f"{reply.get('message', 'unknown error')}"
            )


def send_assets_to_receivers(
    scan_store: dict[str, Any],
    file_paths: list[str],
    port_override: int | None,
    chunk_size: int,
) -> None:
    receivers = scan_store.get("receivers", [])
    if not receivers:
        raise RuntimeError("No receivers available in scan store.")

    abs_files = [to_abs(path_str) for path_str in file_paths]

    targets = []
    seen_hosts: set[str] = set()
    for receiver in receivers:
        host = str(receiver.get("host", "")).strip()
        if not host or host in seen_hosts:
            continue
        seen_hosts.add(host)
        port = int(port_override if port_override is not None else receiver.get("port", 50051))
        targets.append((host, port))

    for host, port in targets:
        print(f"[distributed] Sending assets to {host}:{port}")
        for file_path in abs_files:
            print(f"[distributed]   -> {file_path}")
            send_file_to_receiver(host, port, file_path, chunk_size)
    print("[distributed] Asset distribution complete.")


def parse_file_list(raw_files: str) -> list[str]:
    items = [item.strip() for item in raw_files.split(",") if item.strip()]
    if not items:
        raise ValueError("files list cannot be empty")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "High-level CLI for multi-computer CNN split planning and asset distribution."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to config file (default: {DEFAULT_CONFIG_PATH})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser(
        "discover", help="Discover receiver nodes and store scan results."
    )
    discover_parser.add_argument("--subnet", default=None, help="CIDR subnet override.")
    discover_parser.add_argument(
        "--host",
        default=None,
        help="Direct receiver host fallback if subnet scan finds 0 receivers.",
    )
    discover_parser.add_argument(
        "--direct",
        action="store_true",
        help="Skip subnet scan and connect directly to --host (or config target_host).",
    )
    discover_parser.add_argument("--port", type=int, default=None, help="Receiver port override.")
    discover_parser.add_argument("--timeout", type=float, default=None, help="Probe timeout override.")
    discover_parser.add_argument("--workers", type=int, default=None, help="Scan thread count override.")
    discover_parser.add_argument(
        "--scan-store",
        default=str(DEFAULT_SCAN_STORE),
        help=f"Output scan store path (default: {DEFAULT_SCAN_STORE})",
    )

    split_parser = subparsers.add_parser(
        "split", help="Build split store from receivers and compute CNN layer plan."
    )
    split_parser.add_argument(
        "--scan-store",
        default=str(DEFAULT_SCAN_STORE),
        help=f"Input scan store path (default: {DEFAULT_SCAN_STORE})",
    )
    split_parser.add_argument(
        "--split-store",
        default=str(DEFAULT_SPLIT_STORE),
        help=f"Output split GPU store path (default: {DEFAULT_SPLIT_STORE})",
    )
    split_parser.add_argument(
        "--plan-out",
        default=str(DEFAULT_PLAN_OUT),
        help=f"Output split plan path (default: {DEFAULT_PLAN_OUT})",
    )

    send_parser = subparsers.add_parser(
        "send", help="Send CNN/dataset assets to all discovered receiver hosts."
    )
    send_parser.add_argument(
        "--scan-store",
        default=str(DEFAULT_SCAN_STORE),
        help=f"Input scan store path (default: {DEFAULT_SCAN_STORE})",
    )
    send_parser.add_argument(
        "--files",
        default=",".join(DEFAULT_ASSETS),
        help=(
            "Comma-separated file paths to send to each receiver "
            f"(default: {','.join(DEFAULT_ASSETS)})"
        ),
    )
    send_parser.add_argument("--port", type=int, default=None, help="Receiver port override.")
    send_parser.add_argument("--chunk-size", type=int, default=1024 * 1024, help="Chunk size bytes.")

    full_parser = subparsers.add_parser(
        "full",
        help="Run discover + split + send in one command.",
    )
    full_parser.add_argument("--subnet", default=None, help="CIDR subnet override.")
    full_parser.add_argument(
        "--host",
        default=None,
        help="Direct receiver host fallback if subnet scan finds 0 receivers.",
    )
    full_parser.add_argument(
        "--direct",
        action="store_true",
        help="Skip subnet scan and connect directly to --host (or config target_host).",
    )
    full_parser.add_argument("--port", type=int, default=None, help="Receiver port override.")
    full_parser.add_argument("--timeout", type=float, default=None, help="Probe timeout override.")
    full_parser.add_argument("--workers", type=int, default=None, help="Scan thread count override.")
    full_parser.add_argument(
        "--scan-store",
        default=str(DEFAULT_SCAN_STORE),
        help=f"Output scan store path (default: {DEFAULT_SCAN_STORE})",
    )
    full_parser.add_argument(
        "--split-store",
        default=str(DEFAULT_SPLIT_STORE),
        help=f"Output split GPU store path (default: {DEFAULT_SPLIT_STORE})",
    )
    full_parser.add_argument(
        "--plan-out",
        default=str(DEFAULT_PLAN_OUT),
        help=f"Output split plan path (default: {DEFAULT_PLAN_OUT})",
    )
    full_parser.add_argument(
        "--files",
        default=",".join(DEFAULT_ASSETS),
        help=(
            "Comma-separated file paths to send to each receiver "
            f"(default: {','.join(DEFAULT_ASSETS)})"
        ),
    )
    full_parser.add_argument("--chunk-size", type=int, default=1024 * 1024, help="Chunk size bytes.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = to_abs(str(args.config))
    config = load_cli_config(config_path)

    if args.command == "discover":
        discover_receivers_to_store(
            config=config,
            subnet=args.subnet,
            port=args.port,
            timeout=args.timeout,
            workers=args.workers,
            host=args.host,
            direct=bool(args.direct),
            scan_store_path=to_abs(args.scan_store),
        )
        return

    if args.command == "split":
        scan_store = read_json(to_abs(args.scan_store))
        split_store = build_distributed_gpu_store(
            scan_store=scan_store,
            split_store_path=to_abs(args.split_store),
        )
        build_split_plan(
            split_store=split_store,
            plan_out_path=to_abs(args.plan_out),
        )
        return

    if args.command == "send":
        scan_store = read_json(to_abs(args.scan_store))
        send_assets_to_receivers(
            scan_store=scan_store,
            file_paths=parse_file_list(args.files),
            port_override=args.port,
            chunk_size=args.chunk_size,
        )
        return

    if args.command == "full":
        scan_store = discover_receivers_to_store(
            config=config,
            subnet=args.subnet,
            port=args.port,
            timeout=args.timeout,
            workers=args.workers,
            host=args.host,
            direct=bool(args.direct),
            scan_store_path=to_abs(args.scan_store),
        )
        split_store = build_distributed_gpu_store(
            scan_store=scan_store,
            split_store_path=to_abs(args.split_store),
        )
        build_split_plan(
            split_store=split_store,
            plan_out_path=to_abs(args.plan_out),
        )
        send_assets_to_receivers(
            scan_store=scan_store,
            file_paths=parse_file_list(args.files),
            port_override=args.port,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
