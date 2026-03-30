# Simple Hotspot File Transfer (Python, No Extra Packages)

This is the easiest alternative to the previous C++/gRPC setup.

It uses only Python standard library (`socket`, `json`, `argparse`) so there are no external dependencies to install.

## Project Structure

- `transfer_python/receiver.py`
- `transfer_python/sender.py`
- `legacy_grpc_proto/transfer.proto` (legacy from previous C++/gRPC iteration)

## Prerequisites

- Python 3.8+ (already available on most systems)

## Run

On receiver node:

```powershell
python .\transfer_python\receiver.py --host 0.0.0.0 --port 50051 --output-dir .\incoming
```

On sender node:

```powershell
python .\transfer_python\sender.py <RECEIVER_IP> .\path\to\file.bin --port 50051 --chunk-size 1048576
```

## Protocol

1. Sender sends metadata JSON line (`filename`, `size`, `chunk_size`).
2. Receiver replies `OK`.
3. Sender streams file bytes in chunks.
4. Receiver replies with final JSON status.

## Notes

- Both nodes must be on the same hotspot/LAN and reachable by IP.
- Open firewall for the chosen port if needed.
- This is a simple v1 transport (no TLS/auth/retry-resume yet).
