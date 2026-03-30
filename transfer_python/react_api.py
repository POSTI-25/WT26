from __future__ import annotations

import json
import socket
import time
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Transfer Dashboard API", version="1.0.0")

# Keep a lightweight in-memory history for the React dashboard.
TRANSFER_HISTORY: list[dict[str, Any]] = []
MAX_HISTORY_ITEMS = 200

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
