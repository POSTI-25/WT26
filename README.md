# Simple Hotspot File Transfer + Contributor GPU Probe (Python)

This is the easiest alternative to the previous C++/gRPC setup.

It uses Python standard library for transfer. For better NVIDIA GPU metrics, you can optionally install NVML Python bindings.

## Project Structure

- `transfer_python/receiver.py`
- `transfer_python/sender.py`

## Prerequisites

- Python 3.8+ (already available on most systems)
- Optional for NVML mode: `pip install nvidia-ml-py`

## Run

On receiver node:

```powershell
python .\transfer_python\receiver.py --host 0.0.0.0 --port 50051 --output-dir .\incoming
```

On sender node:

```powershell
python .\transfer_python\sender.py <RECEIVER_IP> .\path\to\file.bin --port 50051 --chunk-size 1048576
```

GPU info only (from dashboard/client machine):

```powershell
python .\transfer_python\sender.py <RECEIVER_IP> --port 50051 --gpu-info
```

## Protocol

1. Sender sends metadata JSON line (`filename`, `size`, `chunk_size`).
2. Receiver replies `OK`.
3. Sender streams file bytes in chunks.
4. Receiver replies with final JSON status.

GPU query mode:

1. Sender sends `{"request": "gpu_info"}`.
2. Receiver returns a JSON report with GPU count and per-GPU usage/memory info.

## Notes

- Both nodes must be on the same hotspot/LAN and reachable by IP.
- Open firewall for the chosen port if needed.
- Receiver prefers NVML (`pynvml`) and falls back to `nvidia-smi` automatically.
- This is a simple v1 transport (no TLS/auth/retry-resume yet).
