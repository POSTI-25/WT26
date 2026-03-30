import argparse
import os
import shutil
import subprocess
from pathlib import Path


def require_tool(tool_name: str) -> None:
    if shutil.which(tool_name) is None:
        raise RuntimeError(f"Required tool not found in PATH: {tool_name}")


def run_command(command: list[str], cwd: Path | None = None) -> None:
    print(f"[runner] {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def find_latest_file(incoming_dir: Path, pattern: str) -> Path:
    matches = [p for p in incoming_dir.glob(pattern) if p.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"No files matched pattern '{pattern}' in '{incoming_dir}'."
        )
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def compile_and_run_c(file_path: Path) -> None:
    compiler = shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        raise RuntimeError("No C compiler found (gcc/clang).")
    output_exe = file_path.with_suffix(".exe")
    run_command([compiler, str(file_path), "-O2", "-o", str(output_exe)])
    run_command([str(output_exe)], cwd=file_path.parent)


def compile_and_run_cpp(file_path: Path) -> None:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise RuntimeError("No C++ compiler found (g++/clang++).")
    output_exe = file_path.with_suffix(".exe")
    run_command([compiler, str(file_path), "-std=c++20", "-O2", "-o", str(output_exe)])
    run_command([str(output_exe)], cwd=file_path.parent)


def compile_and_run_java(file_path: Path) -> None:
    require_tool("javac")
    require_tool("java")
    run_command(["javac", str(file_path)], cwd=file_path.parent)
    run_command(["java", "-cp", str(file_path.parent), file_path.stem], cwd=file_path.parent)


def run_file(file_path: Path, open_unknown: bool) -> None:
    ext = file_path.suffix.lower()

    if ext == ".py":
        require_tool("python")
        run_command(["python", str(file_path)], cwd=file_path.parent)
        return
    if ext == ".js":
        require_tool("node")
        run_command(["node", str(file_path)], cwd=file_path.parent)
        return
    if ext in (".ps1",):
        require_tool("powershell")
        run_command(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(file_path)],
            cwd=file_path.parent,
        )
        return
    if ext in (".bat", ".cmd"):
        require_tool("cmd")
        run_command(["cmd", "/c", str(file_path)], cwd=file_path.parent)
        return
    if ext == ".exe":
        run_command([str(file_path)], cwd=file_path.parent)
        return
    if ext == ".c":
        compile_and_run_c(file_path)
        return
    if ext in (".cpp", ".cc", ".cxx"):
        compile_and_run_cpp(file_path)
        return
    if ext == ".java":
        compile_and_run_java(file_path)
        return
    if ext == ".go":
        require_tool("go")
        run_command(["go", "run", str(file_path)], cwd=file_path.parent)
        return
    if ext == ".rb":
        require_tool("ruby")
        run_command(["ruby", str(file_path)], cwd=file_path.parent)
        return

    if open_unknown and os.name == "nt":
        print(f"[runner] Opening with default app: {file_path}")
        os.startfile(str(file_path))
        return

    raise RuntimeError(
        f"Unsupported file extension: '{ext or '(none)'}'. "
        "Use --open-unknown to open with default app on Windows."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a received file by extension, or auto-run latest received_* file."
    )
    parser.add_argument(
        "--incoming-dir",
        default="incoming",
        help="Directory containing received files (default: incoming).",
    )
    parser.add_argument(
        "--pattern",
        default="received_*",
        help="Glob pattern used when --file is not provided (default: received_*).",
    )
    parser.add_argument(
        "--file",
        default="",
        help="Explicit file path to run. If omitted, latest match from --incoming-dir is used.",
    )
    parser.add_argument(
        "--open-unknown",
        action="store_true",
        help="Open unsupported file types with the default app (Windows only).",
    )
    args = parser.parse_args()

    if args.file:
        target = Path(args.file).expanduser()
        if not target.is_absolute():
            target = Path.cwd() / target
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found: {target}")
    else:
        incoming_dir = Path(args.incoming_dir).expanduser()
        if not incoming_dir.is_absolute():
            incoming_dir = Path.cwd() / incoming_dir
        target = find_latest_file(incoming_dir, args.pattern)

    print(f"[runner] Selected file: {target}")
    run_file(target, args.open_unknown)


if __name__ == "__main__":
    main()
