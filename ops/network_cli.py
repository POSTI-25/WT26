import argparse
import concurrent.futures
import ipaddress
import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR.parent / "config" / "cli_config.json"


class ConfigError(Exception):
    pass


def receive_line(reader) -> str:
    line = reader.readline()
    if not line:
        raise ConnectionError("Connection closed while reading line.")
    return line.decode("utf-8").rstrip("\n")


def load_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in config file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Config root must be a JSON object.")
    return data


def load_cli_config(config_path: Path) -> dict[str, Any]:
    raw = load_json_file(config_path)

    try:
        target_host = str(raw.get("target_host", "127.0.0.1"))
        target_port = int(raw.get("target_port", 50051))
        discovery_subnet = str(raw.get("discovery_subnet", "192.168.1.0/24"))
        connect_timeout_seconds = float(raw.get("connect_timeout_seconds", 3.0))
        scan_timeout_seconds = float(raw.get("scan_timeout_seconds", 0.35))
        scan_max_workers = int(raw.get("scan_max_workers", 64))
        max_scan_hosts = int(raw.get("max_scan_hosts", 512))
        scan_store_file = str(raw.get("scan_store_file", "../data/gpu/network_gpu_store.json"))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Config contains invalid value types: {exc}") from exc

    if target_port <= 0 or target_port > 65535:
        raise ConfigError("target_port must be between 1 and 65535.")
    if connect_timeout_seconds <= 0:
        raise ConfigError("connect_timeout_seconds must be > 0.")
    if scan_timeout_seconds <= 0:
        raise ConfigError("scan_timeout_seconds must be > 0.")
    if scan_max_workers <= 0:
        raise ConfigError("scan_max_workers must be > 0.")
    if max_scan_hosts <= 0:
        raise ConfigError("max_scan_hosts must be > 0.")

    return {
        "target_host": target_host,
        "target_port": target_port,
        "discovery_subnet": discovery_subnet,
        "connect_timeout_seconds": connect_timeout_seconds,
        "scan_timeout_seconds": scan_timeout_seconds,
        "scan_max_workers": scan_max_workers,
        "max_scan_hosts": max_scan_hosts,
        "scan_store_file": scan_store_file,
    }


def resolve_output_path(raw_path: str, config_path: Path) -> Path:
    output = Path(raw_path)
    if output.is_absolute():
        return output
    return config_path.parent / output


def send_json_request(host: str, port: int, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        reader = sock.makefile("rb")
        sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        reply_line = receive_line(reader)

    data = json.loads(reply_line)
    if not isinstance(data, dict):
        raise ValueError("Receiver response is not a JSON object.")
    return data


def probe_receiver(host: str, port: int, timeout: float) -> dict[str, Any] | None:
    try:
        ping_reply = send_json_request(host, port, {"request": "ping"}, timeout)
    except Exception:
        return None

    try:
        gpu_reply = send_json_request(host, port, {"request": "gpu_info"}, timeout)
    except Exception as exc:
        gpu_reply = {
            "ok": False,
            "gpu_count": 0,
            "gpus": [],
            "message": f"gpu_info request failed: {exc}",
        }

    gpu_cards = []
    for gpu in gpu_reply.get("gpus", []):
        gpu_cards.append(
            {
                "index": gpu.get("index"),
                "name": gpu.get("name"),
                "usage_percent": gpu.get("utilization_gpu_percent"),
                "memory_used_mb": gpu.get("memory_used_mb"),
                "memory_total_mb": gpu.get("memory_total_mb"),
            }
        )

    return {
        "host": host,
        "port": port,
        "ok": bool(gpu_reply.get("ok", False)),
        "source": gpu_reply.get("source", "unknown"),
        "gpu_count": int(gpu_reply.get("gpu_count", 0)),
        "gpu_cards": gpu_cards,
        "ping": ping_reply,
        "gpu_report": gpu_reply,
    }


def list_hosts_in_subnet(subnet: str) -> list[str]:
    network = ipaddress.ip_network(subnet, strict=False)
    return [str(host) for host in network.hosts()]


def scan_receivers(
    subnet: str,
    port: int,
    timeout: float,
    max_workers: int,
    max_hosts: int,
) -> list[dict[str, Any]]:
    hosts = list_hosts_in_subnet(subnet)
    if len(hosts) > max_hosts:
        raise ConfigError(
            f"Subnet {subnet} contains {len(hosts)} hosts, which exceeds max_scan_hosts={max_hosts}."
        )

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(probe_receiver, host, port, timeout) for host in hosts]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    results.sort(key=lambda item: ipaddress.ip_address(item["host"]))
    return results


def run_connect(args, config):
    host = args.host or config["target_host"]
    port = args.port if args.port is not None else config["target_port"]
    timeout = args.timeout if args.timeout is not None else config["connect_timeout_seconds"]

    with socket.create_connection((host, port), timeout=timeout):
        pass

    print(f"[cli] TCP connection to {host}:{port} succeeded.")
    try:
        ping = send_json_request(host, port, {"request": "ping"}, timeout)
        print(
            "[cli] Receiver reply: service={service} gpu_count={count} source={source}".format(
                service=ping.get("service", "unknown"),
                count=ping.get("gpu_count", "?"),
                source=ping.get("source", "unknown"),
            )
        )
    except Exception as exc:
        print(f"[cli] Ping request failed after TCP connect: {exc}")


def run_scan(args, config, config_path: Path):
    subnet = args.subnet or config["discovery_subnet"]
    port = args.port if args.port is not None else config["target_port"]
    timeout = args.timeout if args.timeout is not None else config["scan_timeout_seconds"]
    workers = args.workers if args.workers is not None else config["scan_max_workers"]

    raw_store_file = args.store_file or config["scan_store_file"]
    store_path = resolve_output_path(raw_store_file, config_path)

    receivers = scan_receivers(
        subnet=subnet,
        port=port,
        timeout=timeout,
        max_workers=workers,
        max_hosts=config["max_scan_hosts"],
    )

    store = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "subnet": subnet,
        "port": port,
        "receiver_count": len(receivers),
        "receivers": receivers,
    }
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(store, indent=2), encoding="utf-8")

    print(f"[cli] Receivers found: {len(receivers)}")
    for item in receivers:
        print(
            "[cli] {host}:{port} | gpus={count} | source={source} | ok={ok}".format(
                host=item.get("host"),
                port=item.get("port"),
                count=item.get("gpu_count"),
                source=item.get("source"),
                ok=item.get("ok"),
            )
        )

    print(f"[cli] Stored network GPU details to: {store_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CLI for receiver connectivity checks and subnet scanning for GPU details."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to CLI JSON config file (default: {DEFAULT_CONFIG_PATH})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    connect_parser = subparsers.add_parser(
        "connect",
        help="Connect to one receiver host/port from config and send ping.",
    )
    connect_parser.add_argument("--host", default=None, help="Override target_host from config.")
    connect_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override target_port from config.",
    )
    connect_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override connect timeout (seconds).",
    )

    scan_parser = subparsers.add_parser(
        "scan",
        help="Ping all hosts in subnet and fetch GPU details from receiver nodes.",
    )
    scan_parser.add_argument(
        "--subnet",
        default=None,
        help="CIDR subnet to scan (example: 192.168.1.0/24).",
    )
    scan_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override receiver port from config.",
    )
    scan_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-host scan timeout in seconds.",
    )
    scan_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker thread count for parallel subnet probing.",
    )
    scan_parser.add_argument(
        "--store-file",
        default=None,
        help="Override output JSON file path for scan results.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    config = load_cli_config(config_path)

    if args.command == "connect":
        run_connect(args, config)
    elif args.command == "scan":
        run_scan(args, config, config_path)


if __name__ == "__main__":
    main()
