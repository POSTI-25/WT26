import argparse
import json
import socket
from pathlib import Path


def receive_line(reader) -> str:
    line = reader.readline()
    if not line:
        raise ConnectionError("Connection closed while reading line.")
    return line.decode("utf-8").rstrip("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple TCP file sender.")
    parser.add_argument("receiver_host", help="Receiver IP/hostname")
    parser.add_argument("file_path", help="Path to file to send")
    parser.add_argument("--port", type=int, default=50051, help="Receiver port (default: 50051)")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024,
        help="Chunk size in bytes (default: 1048576)",
    )
    args = parser.parse_args()

    file_path = Path(args.file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = file_path.stat().st_size
    metadata = {
        "filename": file_path.name,
        "size": file_size,
        "chunk_size": args.chunk_size,
    }

    with socket.create_connection((args.receiver_host, args.port), timeout=30) as sock:
        reader = sock.makefile("rb")
        sock.sendall((json.dumps(metadata) + "\n").encode("utf-8"))

        ack = receive_line(reader)
        if ack != "OK":
            raise RuntimeError(f"Receiver rejected metadata: {ack}")

        sent = 0
        with file_path.open("rb") as in_file:
            while True:
                chunk = in_file.read(args.chunk_size)
                if not chunk:
                    break
                sock.sendall(chunk)
                sent += len(chunk)
                print(f"\r[sender] Progress: {sent}/{file_size} bytes", end="")

        reply_line = receive_line(reader)
        reply = json.loads(reply_line)
        print()
        if not reply.get("ok", False):
            raise RuntimeError(f"Transfer failed: {reply.get('message', 'unknown error')}")

        print(
            f"[sender] Transfer complete. Receiver stored {reply.get('bytes_received', 0)} bytes."
        )


if __name__ == "__main__":
    main()
