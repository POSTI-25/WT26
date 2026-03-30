import argparse
import json
import socket
from pathlib import Path


def receive_line(reader) -> str:
    line = reader.readline()
    if not line:
        raise ConnectionError("Connection closed while reading line.")
    return line.decode("utf-8").rstrip("\n")


def handle_client(conn: socket.socket, output_dir: Path) -> None:
    with conn:
        reader = conn.makefile("rb")
        metadata_line = receive_line(reader)
        metadata = json.loads(metadata_line)

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
    args = parser.parse_args()

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
