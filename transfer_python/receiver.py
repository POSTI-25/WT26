import argparse
import json
import socket
import subprocess
from pathlib import Path


def receive_line(reader) -> str:
    line = reader.readline()
    if not line:
        raise ConnectionError("Connection closed while reading line.")
    return line.decode("utf-8").rstrip("\n")


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


def print_gpu_report(report) -> None:
    if not report.get("ok", False):
        print(f"[receiver] GPU probe unavailable: {report.get('message', 'unknown reason')}")
        return

    print(
        f"[receiver] GPU probe source: {report.get('source')} | count: {report.get('gpu_count')}"
    )
    for gpu in report.get("gpus", []):
        print(
            "[receiver] GPU {index}: {name} | util={util}% | mem={used}/{total} MB | temp={temp}".format(
                index=gpu.get("index"),
                name=gpu.get("name"),
                util=gpu.get("utilization_gpu_percent"),
                used=gpu.get("memory_used_mb"),
                total=gpu.get("memory_total_mb"),
                temp=gpu.get("temperature_c"),
            )
        )


def handle_client(conn: socket.socket, output_dir: Path) -> None:
    with conn:
        reader = conn.makefile("rb")
        metadata_line = receive_line(reader)
        metadata = json.loads(metadata_line)

        if metadata.get("request") == "gpu_info":
            conn.sendall((json.dumps(get_gpu_report()) + "\n").encode("utf-8"))
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
        default="incoming",
        help="Directory where files are written (default: incoming)",
    )
    parser.add_argument(
        "--gpu-status-file",
        default="",
        help="Optional path to write GPU report JSON on startup.",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Print contributor GPU details and exit.",
    )
    args = parser.parse_args()

    gpu_report = get_gpu_report()
    print_gpu_report(gpu_report)
    if args.gpu_status_file:
        status_path = Path(args.gpu_status_file)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(gpu_report, indent=2), encoding="utf-8")
        print(f"[receiver] Wrote GPU report to {status_path}")

    if args.gpu_only:
        return

    output_dir = Path(args.output_dir)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(5)

    print(f"[receiver] Listening on {args.host}:{args.port}")
    try:
        while True:
            conn, addr = server.accept()
            print(f"[receiver] Connection from {addr[0]}:{addr[1]}")
            try:
                handle_client(conn, output_dir)
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
