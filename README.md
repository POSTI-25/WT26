# gRPC Hotspot File Transfer (C++)

This repository uses gRPC streaming for direct node-to-node transfer over hotspot/LAN:

- `receiver`: runs a gRPC server and writes incoming file chunks to disk.
- `sender`: runs a gRPC client and streams metadata + file chunks.

## Project Structure

- `CMakeLists.txt`
- `proto/transfer.proto`
- `src/sender.cpp`
- `src/receiver.cpp`

## Prerequisites

- C++20 compiler
- Protobuf (`protoc` + C++ libs)
- gRPC C++ libs and CMake package files
- CMake 3.16+

## Build

```powershell
cmake -S . -B build
cmake --build build --config Release
```

## Run

On receiver node:

```powershell
.\build\Release\receiver.exe 50051 .\incoming
```

On sender node:

```powershell
.\build\Release\sender.exe <RECEIVER_IP> .\path\to\file.bin 50051 1048576
```

Parameters:
- `port` default: `50051`
- `chunk_size_bytes` default: `1048576` (1 MB)

## Notes

- Both nodes must be reachable on the hotspot network.
- Open firewall for the gRPC port if needed.
- This is v1 transport only (insecure creds, no auth/retry/checksum yet).
