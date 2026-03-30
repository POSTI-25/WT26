# ZeroMQ Hotspot File Transfer (C++)

This repository now includes a simple structured setup for direct node-to-node transfer over a hotspot/LAN using ZeroMQ:

- `sender`: connects to receiver and sends file chunks.
- `receiver`: binds ports and writes received file to disk.

## Project Structure

- `CMakeLists.txt`
- `src/sender.cpp`
- `src/receiver.cpp`

## Prerequisites

- C++20 compiler
- ZeroMQ runtime/dev package
- `cppzmq` header (`zmq.hpp`)
- CMake 3.16+

## Build

```powershell
cmake -S . -B build
cmake --build build --config Release
```

## Run

On receiver node:

```powershell
.\build\Release\receiver.exe 5555 6000 .\incoming
```

On sender node:

```powershell
.\build\Release\sender.exe <RECEIVER_IP> .\path\to\file.bin 5555 6000 1048576
```

Parameters:
- `control_port` default: `5555`
- `data_port` default: `6000`
- `chunk_size_bytes` default: `1048576` (1 MB)

## Notes

- Both nodes must be reachable on the hotspot network.
- Open firewall for chosen ports if needed.
- This is v1 transport only (no encryption/checksum-retry yet).
