from __future__ import annotations

import concurrent.futures
import ipaddress
import json
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Transfer Dashboard API", version="1.0.0")

# Keep a lightweight in-memory history for the React dashboard.
TRANSFER_HISTORY: list[dict[str, Any]] = []
MAX_HISTORY_ITEMS = 200
DEFAULT_SCAN_MAX_WORKERS = 64
DEFAULT_SCAN_MAX_HOSTS = 512

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def receive_line(reader) -> str:
    line = reader.readline()
    if not line:
        raise ConnectionError("Connection closed while reading line.")
    return line.decode("utf-8").rstrip("\n")


def parse_target(raw_host: str, fallback_port: int) -> tuple[str, int]:
    host_input = (raw_host or "").strip()
    if not host_input:
        return "", fallback_port

    host = host_input
    port = fallback_port

    # Accept values like http://10.0.0.5:50051 or https://node.local
    if "://" in host_input:
        parsed = urlsplit(host_input)
        if parsed.hostname:
            host = parsed.hostname
        if parsed.port:
            port = parsed.port
        return host, port

    # Accept values like 10.0.0.5:50051 (but avoid breaking IPv6 literals)
    if host_input.count(":") == 1 and not host_input.startswith("["):
        candidate_host, candidate_port = host_input.rsplit(":", 1)
        if candidate_port.isdigit():
            return candidate_host.strip(), int(candidate_port)

    return host, port


def split_subnet_values(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    return [part.strip() for part in str(raw_value).split(",") if part.strip()]


def normalize_subnet(raw_subnet: str) -> str:
    network = ipaddress.ip_network(raw_subnet.strip(), strict=False)
    if not isinstance(network, ipaddress.IPv4Network):
        raise ValueError(f"Only IPv4 subnet scanning is supported: {raw_subnet}")
    return str(network)


def list_hosts_in_subnet(subnet: str) -> list[str]:
    network = ipaddress.ip_network(subnet, strict=False)
    if not isinstance(network, ipaddress.IPv4Network):
        raise ValueError(f"Only IPv4 subnet scanning is supported: {subnet}")
    return [str(host) for host in network.hosts()]


def discover_fallback_private_subnets() -> list[str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            local_ip = str(probe.getsockname()[0])
    except OSError:
        return []

    try:
        address = ipaddress.IPv4Address(local_ip)
    except ValueError:
        return []

    if not address.is_private or address.is_loopback or address.is_link_local:
        return []

    return [str(ipaddress.IPv4Network(f"{local_ip}/24", strict=False))]


def load_default_discovery_subnets() -> list[str]:
    config_path = Path(__file__).resolve().parents[1] / "config" / "cli_config.json"
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                configured = split_subnet_values(str(data.get("discovery_subnet", "")))
                if configured:
                    return configured
        except Exception:
            pass

    discovered = discover_fallback_private_subnets()
    if discovered:
        return discovered

    return ["192.168.1.0/24"]


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

    return {
        "ip_address": gpu_reply.get("socket_receiver_ip") or ping_reply.get("receiver_ip") or host,
        "requested_host": host,
        "port": port,
        "ok": bool(gpu_reply.get("ok", False)),
        "service": ping_reply.get("service", "unknown"),
        "hostname": gpu_reply.get("hostname") or ping_reply.get("hostname"),
        "source": gpu_reply.get("source", "unknown"),
        "gpu_count": int(gpu_reply.get("gpu_count", 0)),
        "gpus": gpu_reply.get("gpus", []),
        "ping": ping_reply,
    }


def scan_receivers(
    subnets: list[str],
    port: int,
    timeout: float,
    max_workers: int,
    max_hosts: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    normalized_subnets: list[str] = []
    hosts: list[str] = []
    seen_hosts: set[str] = set()

    for raw_subnet in subnets:
        subnet = normalize_subnet(raw_subnet)
        if subnet not in normalized_subnets:
            normalized_subnets.append(subnet)
        for host in list_hosts_in_subnet(subnet):
            if host in seen_hosts:
                continue
            seen_hosts.add(host)
            hosts.append(host)

    if len(hosts) > max_hosts:
        raise ValueError(
            "Requested scan covers "
            f"{len(hosts)} hosts across {', '.join(normalized_subnets)}, "
            f"which exceeds max_hosts={max_hosts}."
        )

    results: list[dict[str, Any]] = []
    worker_count = min(max_workers, max(1, len(hosts)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(probe_receiver, host, port, timeout) for host in hosts]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    results.sort(key=lambda item: ipaddress.ip_address(item["ip_address"]))
    return normalized_subnets, results


def probe_remote_gpu(host: str, port: int, timeout_seconds: int) -> dict[str, Any]:
    with socket.create_connection((host, port), timeout=timeout_seconds) as sock:
        sock.settimeout(timeout_seconds)
        reader = sock.makefile("rb")
        sock.sendall((json.dumps({"request": "gpu_info"}) + "\n").encode("utf-8"))
        return json.loads(receive_line(reader))


def send_file_bytes(
    host: str,
    port: int,
    timeout_seconds: int,
    chunk_size: int,
    file_name: str,
    data: bytes,
) -> dict[str, Any]:
    file_size = len(data)
    metadata = {
        "filename": file_name,
        "size": file_size,
        "chunk_size": chunk_size,
    }

    with socket.create_connection((host, port), timeout=timeout_seconds) as sock:
        sock.settimeout(timeout_seconds)
        reader = sock.makefile("rb")
        sock.sendall((json.dumps(metadata) + "\n").encode("utf-8"))

        ack = receive_line(reader)
        if ack != "OK":
            raise RuntimeError(f"Receiver rejected metadata: {ack}")

        sent = 0
        start = time.perf_counter()
        payload = memoryview(data)

        while sent < file_size:
            end = min(sent + chunk_size, file_size)
            sock.sendall(payload[sent:end])
            sent = end

        reply = json.loads(receive_line(reader))
        elapsed_seconds = max(time.perf_counter() - start, 1e-9)
        throughput_mbps = (file_size / (1024 * 1024)) / elapsed_seconds

        return {
            "ok": bool(reply.get("ok", False)),
            "reply": reply,
            "elapsed_seconds": elapsed_seconds,
            "throughput_mbps": throughput_mbps,
            "bytes_sent": sent,
            "bytes_total": file_size,
        }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "transfer-dashboard-api"}


@app.post("/api/gpu-probe")
def gpu_probe(payload: dict[str, Any]) -> dict[str, Any]:
    raw_host = str(payload.get("host", "")).strip()
    fallback_port = int(payload.get("port", 50051))
    host, port = parse_target(raw_host, fallback_port)
    timeout_seconds = int(payload.get("timeout_seconds", 30))

    if not host:
        raise HTTPException(status_code=400, detail="host is required")

    try:
        report = probe_remote_gpu(host, port, timeout_seconds)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"ok": True, "report": report}


@app.post("/api/receivers/scan")
def receivers_scan(payload: dict[str, Any]) -> dict[str, Any]:
    raw_subnets = payload.get("subnets")
    if isinstance(raw_subnets, str):
        subnets = split_subnet_values(raw_subnets)
    elif isinstance(raw_subnets, list):
        subnets = [str(item).strip() for item in raw_subnets if str(item).strip()]
    else:
        subnets = []

    if not subnets:
        subnets = split_subnet_values(str(payload.get("subnet", "")))

    if not subnets:
        subnets = load_default_discovery_subnets()

    port = int(payload.get("port", 50051))
    timeout = float(payload.get("timeout_seconds", 0.35))
    max_workers = int(payload.get("max_workers", DEFAULT_SCAN_MAX_WORKERS))
    max_hosts = int(payload.get("max_hosts", DEFAULT_SCAN_MAX_HOSTS))

    if port <= 0 or port > 65535:
        raise HTTPException(status_code=400, detail="port must be between 1 and 65535")
    if timeout <= 0:
        raise HTTPException(status_code=400, detail="timeout_seconds must be > 0")
    if max_workers <= 0:
        raise HTTPException(status_code=400, detail="max_workers must be > 0")
    if max_hosts <= 0:
        raise HTTPException(status_code=400, detail="max_hosts must be > 0")

    try:
        scanned_subnets, receivers = scan_receivers(
            subnets=subnets,
            port=port,
            timeout=timeout,
            max_workers=max_workers,
            max_hosts=max_hosts,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "ok": True,
        "scanned_subnets": scanned_subnets,
        "receiver_count": len(receivers),
        "receivers": receivers,
    }


@app.post("/api/send-file")
async def send_file(
    host: str = Form(...),
    port: int = Form(50051),
    timeout_seconds: int = Form(30),
    chunk_size: int = Form(1024 * 1024),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    parsed_host, parsed_port = parse_target(host, port)
    if not parsed_host:
        raise HTTPException(status_code=400, detail="host is required")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="file is empty")

    try:
        result = send_file_bytes(
            host=parsed_host,
            port=parsed_port,
            timeout_seconds=timeout_seconds,
            chunk_size=chunk_size,
            file_name=file.filename or "upload.bin",
            data=data,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    TRANSFER_HISTORY.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": file.filename or "upload.bin",
            "size_mb": round(len(data) / (1024 * 1024), 3),
            "elapsed_s": round(float(result["elapsed_seconds"]), 3),
            "throughput_mb_s": round(float(result["throughput_mbps"]), 3),
            "ok": bool(result["ok"]),
        }
    )
    if len(TRANSFER_HISTORY) > MAX_HISTORY_ITEMS:
        del TRANSFER_HISTORY[0 : len(TRANSFER_HISTORY) - MAX_HISTORY_ITEMS]

    return {"ok": True, "result": result}


@app.get("/api/history")
def history(limit: int = 50) -> dict[str, Any]:
    if limit < 1:
        limit = 1
    if limit > MAX_HISTORY_ITEMS:
        limit = MAX_HISTORY_ITEMS
    return {"ok": True, "history": TRANSFER_HISTORY[-limit:]}
