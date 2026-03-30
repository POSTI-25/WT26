# Simple Hotspot File Transfer + Contributor GPU Probe (Python)

Python-based local network file transfer and GPU discovery.

The project includes:

- A receiver service that accepts file uploads and serves GPU data.
- A sender client for file transfer or single-host GPU query.
- A config-driven CLI for receiver connectivity checks and subnet GPU discovery.

## Project Structure

- `transfer_python/receiver.py`
- `transfer_python/sender.py`
- `transfer_python/cli.py`
- `transfer_python/cli_config.json`

## Prerequisites

- Python 3.8+
- Optional for richer NVIDIA metrics: `pip install nvidia-ml-py`

## Required Setup for CLI Commands

`connect` and `scan` work only against machines running `receiver.py`.

Run this on each receiver machine:

```powershell
python .\transfer_python\receiver.py --host 0.0.0.0 --port 50051 --output-dir .\incoming --gpu-store-file .\transfer_python\gpu_store.json
```

Optional receiver commands:

```powershell
# Write full GPU status JSON each refresh
python .\transfer_python\receiver.py --gpu-status-file .\transfer_python\gpu_status.json

# Change GPU refresh interval for stored files
python .\transfer_python\receiver.py --gpu-refresh-seconds 2

# Print GPU details once and exit
python .\transfer_python\receiver.py --gpu-only
```

## CLI Config File

Edit before running CLI commands:

- `transfer_python/cli_config.json`

Current fields:

- `target_host`: host used by `connect`.
- `target_port`: port used by `connect` and `scan`.
- `discovery_subnet`: CIDR subnet scanned by `scan`.
- `connect_timeout_seconds`: timeout for `connect`.
- `scan_timeout_seconds`: per-host timeout for `scan`.
- `scan_max_workers`: thread count for parallel scan.
- `max_scan_hosts`: safety limit for subnet size.
- `scan_store_file`: local JSON output file for `scan` results.

## New CLI Commands

All commands below are run from the workspace root.

### `connect`

Connect to a single receiver (from config by default), then send a `ping` request.

```powershell
python .\transfer_python\cli.py connect
```

Options:

- `--config <path>`: JSON config file path (default: `transfer_python/cli_config.json`).
- `--host <ip-or-hostname>`: override `target_host`.
- `--port <port>`: override `target_port`.
- `--timeout <seconds>`: override `connect_timeout_seconds`.

Examples:

```powershell
python .\transfer_python\cli.py connect --host 192.168.1.20 --port 50051
python .\transfer_python\cli.py connect --timeout 5
python .\transfer_python\cli.py --config .\transfer_python\cli_config.json connect
```

### `scan`

Scans the configured subnet, pings receiver nodes, fetches GPU details (`gpu_info`), and stores results locally.

```powershell
python .\transfer_python\cli.py scan
```

Options:

- `--config <path>`: JSON config file path (default: `transfer_python/cli_config.json`).
- `--subnet <cidr>`: override `discovery_subnet`.
- `--port <port>`: override `target_port`.
- `--timeout <seconds>`: override per-host timeout.
- `--workers <count>`: override parallel worker count.
- `--store-file <path>`: override output JSON path for scan results.

Examples:

```powershell
python .\transfer_python\cli.py scan --subnet 192.168.1.0/24 --port 50051
python .\transfer_python\cli.py scan --timeout 0.4 --workers 128
python .\transfer_python\cli.py scan --store-file .\transfer_python\network_gpu_store.json
```

## Existing Sender Commands

File transfer:

```powershell
python .\transfer_python\sender.py <RECEIVER_IP> .\path\to\file.bin --port 50051 --chunk-size 1048576
```

Single receiver GPU report:

```powershell
python .\transfer_python\sender.py <RECEIVER_IP> --port 50051 --gpu-info
```

## Run Received Files (Generalized)

Auto-run latest received file from `incoming`:

```powershell
python .\transfer_python\run_received.py
```

Run a specific received file:

```powershell
python .\transfer_python\run_received.py --file .\incoming\received_test.cpp
```

Run latest with a custom pattern:

```powershell
python .\transfer_python\run_received.py --pattern received_*.py
```

Open unsupported extensions with default Windows app:

```powershell
python .\transfer_python\run_received.py --open-unknown
```

## Protocol Summary

File transfer:

1. Sender sends metadata JSON line (`filename`, `size`, `chunk_size`).
2. Receiver replies `OK`.
3. Sender streams file bytes.
4. Receiver replies with final JSON status.

CLI discovery:

1. CLI `connect`/`scan` sends `{"request": "ping"}` to check receiver availability.
2. CLI `scan` then sends `{"request": "gpu_info"}` to fetch detailed GPU report.
3. Results are stored in `scan_store_file`.

## Notes

- All nodes must be reachable on the same LAN/hotspot.
- Open firewall for the configured receiver port if needed.
- Receiver uses NVML (`pynvml`) when available and falls back to `nvidia-smi`.
- Transport is intentionally simple (no TLS/auth/resume yet).
