import argparse
import json
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


CUDA_CORES_PER_SM = {
    (2, 0): 32,
    (2, 1): 48,
    (3, 0): 192,
    (3, 2): 192,
    (3, 5): 192,
    (3, 7): 192,
    (5, 0): 128,
    (5, 2): 128,
    (5, 3): 128,
    (6, 0): 64,
    (6, 1): 128,
    (6, 2): 128,
    (7, 0): 64,
    (7, 2): 64,
    (7, 5): 64,
    (8, 0): 64,
    (8, 6): 128,
    (8, 7): 128,
    (8, 9): 128,
    (9, 0): 128,
}


def receive_line(reader) -> str:
    line = reader.readline()
    if not line:
        raise ConnectionError("Connection closed while reading line.")
    return line.decode("utf-8").rstrip("\n")


def get_socket_ip(sock: socket.socket, peer: bool) -> str | None:
    try:
        endpoint = sock.getpeername() if peer else sock.getsockname()
    except OSError:
        return None

    if isinstance(endpoint, tuple) and endpoint:
        return str(endpoint[0])
    return None


def get_local_ip(preferred_host: str) -> str:
    if preferred_host and preferred_host not in {"0.0.0.0", "::"}:
        return preferred_host

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            return str(probe.getsockname()[0])
    except OSError:
        return "127.0.0.1"


def enrich_report_with_socket_ips(report: dict, conn: socket.socket) -> dict:
    report_with_socket = dict(report)
    report_with_socket["hostname"] = socket.gethostname()
    report_with_socket["socket_receiver_ip"] = get_socket_ip(conn, peer=False)
    report_with_socket["socket_peer_ip"] = get_socket_ip(conn, peer=True)
    return report_with_socket


def get_cuda_core_fields_with_nvml(pynvml, handle):
    cuda_cores = None
    sm_count = None
    compute_capability = None

    if hasattr(pynvml, "nvmlDeviceGetNumGpuCores"):
        try:
            cuda_cores = int(pynvml.nvmlDeviceGetNumGpuCores(handle))
        except Exception:
            cuda_cores = None

    try:
        sm_count = int(pynvml.nvmlDeviceGetMultiprocessorCount(handle))
    except Exception:
        sm_count = None

    try:
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        major = int(major)
        minor = int(minor)
        compute_capability = f"{major}.{minor}"
        if cuda_cores is None and sm_count is not None:
            cores_per_sm = CUDA_CORES_PER_SM.get((major, minor))
            if cores_per_sm is not None:
                cuda_cores = sm_count * cores_per_sm
    except Exception:
        compute_capability = None

    return cuda_cores, sm_count, compute_capability


def query_gpus_with_nvml():
    try:
        import pynvml  # type: ignore
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
    except Exception:
        return None

    gpus = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8", errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)

            temperature_c = None
            try:
                temperature_c = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temperature_c = None

            cuda_cores, sm_count, compute_capability = get_cuda_core_fields_with_nvml(
                pynvml, handle
            )

            gpus.append(
                {
                    "index": index,
                    "name": name,
                    "uuid": uuid,
                    "memory_total_mb": int(mem.total // (1024 * 1024)),
                    "memory_used_mb": int(mem.used // (1024 * 1024)),
                    "memory_free_mb": int(mem.free // (1024 * 1024)),
                    "utilization_gpu_percent": int(util.gpu),
                    "temperature_c": temperature_c,
                    "cuda_cores": cuda_cores,
                    "sm_count": sm_count,
                    "compute_capability": compute_capability,
                    "source": "nvml",
                }
            )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return gpus


def query_gpus_with_nvidia_smi():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None

    gpus = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "memory_total_mb": int(parts[3]),
                "memory_used_mb": int(parts[4]),
                "memory_free_mb": int(parts[5]),
                "utilization_gpu_percent": int(parts[6]),
                "temperature_c": int(parts[7]),
                "cuda_cores": None,
                "sm_count": None,
                "compute_capability": None,
                "source": "nvidia-smi",
            }
        )
    return gpus


def get_gpu_report():
    gpus = query_gpus_with_nvml()
    if gpus is not None:
        return {
            "ok": True,
            "source": "nvml",
            "gpu_count": len(gpus),
            "gpus": gpus,
        }

    gpus = query_gpus_with_nvidia_smi()
    if gpus is not None:
        return {
            "ok": True,
            "source": "nvidia-smi",
            "gpu_count": len(gpus),
            "gpus": gpus,
        }

    return {
        "ok": False,
        "source": "none",
        "gpu_count": 0,
        "gpus": [],
        "message": (
            "GPU query failed. Install NVIDIA driver and optionally 'nvidia-ml-py' "
            "(module: pynvml) for NVML support."
        ),
    }


def build_ping_reply(report):
    reply = {
        "ok": bool(report.get("ok", False)),
        "service": "gpu_receiver",
        "source": report.get("source", "none"),
        "gpu_count": int(report.get("gpu_count", 0)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if report.get("hostname"):
        reply["hostname"] = report.get("hostname")
    if report.get("socket_receiver_ip"):
        reply["receiver_ip"] = report.get("socket_receiver_ip")
    if report.get("socket_peer_ip"):
        reply["peer_ip"] = report.get("socket_peer_ip")
    return reply


def print_gpu_report(report) -> None:
    receiver_ip = report.get("socket_receiver_ip")
    peer_ip = report.get("socket_peer_ip")
    if receiver_ip or peer_ip:
        print(
            "[receiver] Socket IPs | receiver_ip={receiver_ip} | peer_ip={peer_ip}".format(
                receiver_ip=receiver_ip or "unknown",
                peer_ip=peer_ip or "unknown",
            )
        )

    if not report.get("ok", False):
        print(f"[receiver] GPU probe unavailable: {report.get('message', 'unknown reason')}")
        return

    print(
        f"[receiver] GPU probe source: {report.get('source')} | count: {report.get('gpu_count')}"
    )
    for gpu in report.get("gpus", []):
        print(
            "[receiver] GPU {index}: {name} | util={util}% | mem={used}/{total} MB | "
            "cuda_cores={cuda_cores} | sm_count={sm_count} | cc={cc} | temp={temp}".format(
                index=gpu.get("index"),
                name=gpu.get("name"),
                util=gpu.get("utilization_gpu_percent"),
                used=gpu.get("memory_used_mb"),
                total=gpu.get("memory_total_mb"),
                cuda_cores=(
                    gpu.get("cuda_cores") if gpu.get("cuda_cores") is not None else "unknown"
                ),
                sm_count=(gpu.get("sm_count") if gpu.get("sm_count") is not None else "unknown"),
                cc=(
                    gpu.get("compute_capability")
                    if gpu.get("compute_capability") is not None
                    else "unknown"
                ),
                temp=gpu.get("temperature_c"),
            )
        )


def build_gpu_store(report, ip_address: str):
    gpu_cards = []
    for gpu in report.get("gpus", []):
        gpu_cards.append(
            {
                "gpu_card": gpu.get("name"),
                "gpu_usage_percent": gpu.get("utilization_gpu_percent"),
                "memory_available_mb": gpu.get("memory_free_mb"),
                "cuda_cores": gpu.get("cuda_cores"),
            }
        )

    return {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ip_address": ip_address,
        "gpu_count": len(gpu_cards),
        "gpu_cards": gpu_cards,
    }


def write_gpu_store(report, gpu_store_path: Path, ip_address: str) -> None:
    store = build_gpu_store(report, ip_address)
    gpu_store_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_store_path.write_text(json.dumps(store, indent=2), encoding="utf-8")


def write_gpu_status(report, gpu_status_path: Path) -> None:
    gpu_status_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_status_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def refresh_gpu_files(gpu_store_path: Path, gpu_status_path: Path | None, local_ip: str):
    latest_gpu_report = get_gpu_report()
    write_gpu_store(latest_gpu_report, gpu_store_path, local_ip)
    if gpu_status_path is not None:
        write_gpu_status(latest_gpu_report, gpu_status_path)
    return latest_gpu_report


def handle_client(
    conn: socket.socket,
    output_dir: Path,
    gpu_store_path: Path,
    gpu_status_path: Path | None,
    local_ip: str,
) -> None:
    with conn:
        reader = conn.makefile("rb")
        metadata_line = receive_line(reader)
        metadata = json.loads(metadata_line)

        latest_gpu_report = refresh_gpu_files(gpu_store_path, gpu_status_path, local_ip)
        latest_gpu_report = enrich_report_with_socket_ips(latest_gpu_report, conn)

        if metadata.get("request") == "ping":
            print_gpu_report(latest_gpu_report)
            conn.sendall((json.dumps(build_ping_reply(latest_gpu_report)) + "\n").encode("utf-8"))
            return

        if metadata.get("request") == "gpu_info":
            print_gpu_report(latest_gpu_report)
            conn.sendall((json.dumps(latest_gpu_report) + "\n").encode("utf-8"))
            return

        filename = Path(str(metadata["filename"])).name
        expected_size = int(metadata["size"])
        chunk_size = int(metadata.get("chunk_size", 1024 * 1024))

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"received_{filename}"

        conn.sendall(b"OK\n")

        written = 0
        with out_path.open("wb") as out_file:
            while written < expected_size:
                to_read = min(chunk_size, expected_size - written)
                data = reader.read(to_read)
                if not data:
                    raise ConnectionError("Connection closed during file transfer.")
                out_file.write(data)
                written += len(data)
                print(f"\r[receiver] Progress: {written}/{expected_size} bytes", end="")

        print(f"\n[receiver] Saved file to {out_path}")

        reply = {
            "ok": written == expected_size,
            "bytes_received": written,
            "path": str(out_path),
            "message": "Transfer complete." if written == expected_size else "Size mismatch.",
        }
        conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple TCP file receiver.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=50051, help="Bind port (default: 50051)")
    parser.add_argument(
        "--output-dir",
        default="data/incoming",
        help="Directory where files are written (default: data/incoming).",
    )
    parser.add_argument(
        "--gpu-status-file",
        default="",
        help="Optional path to write GPU report JSON on startup.",
    )
    parser.add_argument(
        "--gpu-store-file",
        default="data/gpu/receiver_gpu_store.json",
        help="Path to store receiver GPU card + usage summary (default: data/gpu/receiver_gpu_store.json).",
    )
    parser.add_argument(
        "--gpu-refresh-seconds",
        type=float,
        default=2.0,
        help="How often to refresh stored GPU usage while server is running (default: 2.0).",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Print contributor GPU details and exit.",
    )
    args = parser.parse_args()

    gpu_report = get_gpu_report()
    print_gpu_report(gpu_report)
    local_ip = get_local_ip(args.host)
    gpu_store_path = Path(args.gpu_store_file)
    write_gpu_store(gpu_report, gpu_store_path, local_ip)
    print(f"[receiver] Stored GPU summary at {gpu_store_path}")

    gpu_status_path = Path(args.gpu_status_file) if args.gpu_status_file else None
    if args.gpu_status_file:
        write_gpu_status(gpu_report, gpu_status_path)
        print(f"[receiver] Wrote GPU report to {gpu_status_path}")

    if args.gpu_only:
        return

    output_dir = Path(args.output_dir)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(5)
    server.settimeout(1.0)

    print(f"[receiver] Listening on {args.host}:{args.port}")
    refresh_interval = max(0.5, float(args.gpu_refresh_seconds))
    next_refresh = time.monotonic() + refresh_interval
    try:
        while True:
            now = time.monotonic()
            if now >= next_refresh:
                refresh_gpu_files(gpu_store_path, gpu_status_path, local_ip)
                next_refresh = now + refresh_interval

            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue

            print(f"[receiver] Connection from {addr[0]}:{addr[1]}")
            try:
                handle_client(conn, output_dir, gpu_store_path, gpu_status_path, local_ip)
            except Exception as exc:
                print(f"\n[receiver] Error: {exc}")
                try:
                    conn.sendall(
                        (json.dumps({"ok": False, "message": str(exc)}) + "\n").encode("utf-8")
                    )
                except OSError:
                    pass
    except KeyboardInterrupt:
        print("\n[receiver] Shutting down.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
