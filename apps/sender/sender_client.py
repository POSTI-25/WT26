import asyncio
import argparse
import concurrent.futures
import ipaddress
import json
import re
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR.parent.parent / "config" / "cli_config.json"


class ProgramState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class SchedulerJob:
    job_id: str
    required_vram_mb: int
    checkpoints: list[dict[str, Any]]
    state: ProgramState = ProgramState.QUEUED
    checkpoint_index: int = -1
    assigned_worker: str | None = None
    last_error: str | None = None

    def next_checkpoint(self) -> dict[str, Any] | None:
        next_index = self.checkpoint_index + 1
        if next_index >= len(self.checkpoints):
            return None
        return self.checkpoints[next_index]

    def checkpoint_vram_requirement(self) -> int:
        next_checkpoint = self.next_checkpoint()
        if next_checkpoint is None:
            return 0
        return int(next_checkpoint.get("required_vram_mb", self.required_vram_mb))


@dataclass
class SchedulerWorker:
    host: str
    port: int
    free_vram_mb: int = 0
    total_vram_mb: int = 0
    state: ProgramState = ProgramState.FAILED
    active_job_ids: list[str] = field(default_factory=list)

    @property
    def worker_id(self) -> str:
        return f"{self.host}:{self.port}"


class FaultTolerantGpuScheduler:
    def __init__(
        self,
        jobs: list[SchedulerJob],
        workers: list[SchedulerWorker],
        state_file: Path,
        worker_timeout_seconds: float,
        scheduler_interval_seconds: float,
    ) -> None:
        self.jobs = {job.job_id: job for job in jobs}
        self.known_workers = {worker.worker_id: worker for worker in workers}
        self.active_gpus: list[SchedulerWorker] = []
        self.state_file = state_file
        self.worker_timeout_seconds = worker_timeout_seconds
        self.scheduler_interval_seconds = scheduler_interval_seconds
        self.program_state = ProgramState.QUEUED
        self.no_worker_cycles = 0

    async def workers(self) -> None:
        probes = [
            asyncio.to_thread(
                probe_gpu_node,
                worker.host,
                worker.port,
                self.worker_timeout_seconds,
            )
            for worker in self.known_workers.values()
        ]

        probe_results = await asyncio.gather(*probes, return_exceptions=True)
        reachable_workers: list[SchedulerWorker] = []

        for worker, probe_result in zip(self.known_workers.values(), probe_results):
            was_failed = worker.state == ProgramState.FAILED

            if isinstance(probe_result, Exception) or probe_result is None:
                worker.state = ProgramState.FAILED
                worker.free_vram_mb = 0
                worker.total_vram_mb = 0
                if not was_failed:
                    self.rollback_jobs_for_failed_worker(worker.worker_id)
                continue

            total_vram_mb = 0
            free_vram_mb = 0
            for gpu in probe_result.get("gpu_cards", []):
                total = int(gpu.get("memory_total_mb") or 0)
                used = int(gpu.get("memory_used_mb") or 0)
                total_vram_mb += total
                free_vram_mb += max(0, total - used)

            worker.total_vram_mb = total_vram_mb
            worker.free_vram_mb = free_vram_mb
            worker.state = ProgramState.RUNNING if worker.active_job_ids else ProgramState.COMPLETED
            reachable_workers.append(worker)

        self.active_gpus = [worker for worker in reachable_workers if worker.state != ProgramState.FAILED]
        self.persist_state()

    async def eval_queue(self) -> None:
        queued_jobs = [
            job
            for job in self.jobs.values()
            if job.state in {ProgramState.QUEUED, ProgramState.FAILED}
        ]

        for job in queued_jobs:
            required_vram_mb = job.checkpoint_vram_requirement()
            if required_vram_mb <= 0 and job.next_checkpoint() is None:
                job.state = ProgramState.COMPLETED
                continue

            selected_worker = self.select_worker_for_job(required_vram_mb)
            if selected_worker is None:
                continue

            await self.run_job_from_last_checkpoint(job, selected_worker)

        self.recompute_program_state()
        self.persist_state()

    def select_worker_for_job(self, required_vram_mb: int) -> SchedulerWorker | None:
        candidates = [
            worker
            for worker in self.active_gpus
            if worker.state in {ProgramState.RUNNING, ProgramState.COMPLETED}
            and worker.free_vram_mb >= required_vram_mb
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda worker: worker.free_vram_mb, reverse=True)
        return candidates[0]

    async def run_job_from_last_checkpoint(
        self,
        job: SchedulerJob,
        worker: SchedulerWorker,
    ) -> None:
        reserve_vram_mb = max(job.required_vram_mb, job.checkpoint_vram_requirement())
        worker.free_vram_mb = max(0, worker.free_vram_mb - reserve_vram_mb)
        worker.active_job_ids.append(job.job_id)
        worker.state = ProgramState.RUNNING
        job.state = ProgramState.RUNNING
        job.assigned_worker = worker.worker_id
        job.last_error = None

        try:
            while True:
                checkpoint = job.next_checkpoint()
                if checkpoint is None:
                    job.state = ProgramState.COMPLETED
                    break

                latest_worker = self.known_workers.get(worker.worker_id)
                if latest_worker is None or latest_worker.state == ProgramState.FAILED:
                    raise RuntimeError(f"Assigned worker {worker.worker_id} is offline.")

                await asyncio.sleep(0.05)
                job.checkpoint_index += 1

        except Exception as exc:
            self.rollback_job(job, str(exc))
        finally:
            worker.free_vram_mb += reserve_vram_mb
            worker.active_job_ids = [item for item in worker.active_job_ids if item != job.job_id]
            if worker.state != ProgramState.FAILED:
                worker.state = ProgramState.COMPLETED if not worker.active_job_ids else ProgramState.RUNNING

    def rollback_job(self, job: SchedulerJob, reason: str) -> None:
        job.state = ProgramState.QUEUED
        job.assigned_worker = None
        job.last_error = reason

    def rollback_jobs_for_failed_worker(self, worker_id: str) -> None:
        for job in self.jobs.values():
            if job.assigned_worker == worker_id and job.state == ProgramState.RUNNING:
                self.rollback_job(job, f"Worker {worker_id} failed. Rolled back to checkpoint index {job.checkpoint_index}.")

    def recompute_program_state(self) -> None:
        if any(job.state == ProgramState.RUNNING for job in self.jobs.values()):
            self.program_state = ProgramState.RUNNING
            self.no_worker_cycles = 0
            return

        if all(job.state == ProgramState.COMPLETED for job in self.jobs.values()):
            self.program_state = ProgramState.COMPLETED
            self.no_worker_cycles = 0
            return

        if self.active_gpus:
            self.program_state = ProgramState.QUEUED
            self.no_worker_cycles = 0
            return

        self.no_worker_cycles += 1
        if self.no_worker_cycles >= 3:
            self.program_state = ProgramState.FAILED
        else:
            self.program_state = ProgramState.QUEUED

    def persist_state(self) -> None:
        payload = {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "program_state": self.program_state.value,
            "active_gpu_count": len(self.active_gpus),
            "workers": [
                {
                    "worker_id": worker.worker_id,
                    "host": worker.host,
                    "port": worker.port,
                    "state": worker.state.value,
                    "free_vram_mb": worker.free_vram_mb,
                    "total_vram_mb": worker.total_vram_mb,
                    "active_job_ids": worker.active_job_ids,
                }
                for worker in self.known_workers.values()
            ],
            "jobs": [
                {
                    "job_id": job.job_id,
                    "state": job.state.value,
                    "required_vram_mb": job.required_vram_mb,
                    "checkpoint_index": job.checkpoint_index,
                    "checkpoint_count": len(job.checkpoints),
                    "last_completed_checkpoint": (
                        job.checkpoints[job.checkpoint_index].get("name")
                        if job.checkpoint_index >= 0 and job.checkpoint_index < len(job.checkpoints)
                        else None
                    ),
                    "next_checkpoint": (
                        job.next_checkpoint().get("name") if job.next_checkpoint() else None
                    ),
                    "assigned_worker": job.assigned_worker,
                    "last_error": job.last_error,
                }
                for job in self.jobs.values()
            ],
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    async def run(self, max_cycles: int = 0) -> ProgramState:
        cycle = 0
        while True:
            cycle += 1
            await self.workers()
            await self.eval_queue()

            if self.program_state in {ProgramState.COMPLETED, ProgramState.FAILED}:
                return self.program_state

            if max_cycles > 0 and cycle >= max_cycles:
                return self.program_state

            await asyncio.sleep(self.scheduler_interval_seconds)


def parse_scheduler_jobs(raw_jobs: Any) -> list[SchedulerJob]:
    if isinstance(raw_jobs, dict):
        raw_jobs = raw_jobs.get("jobs", [])

    if not isinstance(raw_jobs, list):
        raise ValueError("Scheduler jobs file must be a JSON list or an object with a 'jobs' list.")

    jobs: list[SchedulerJob] = []
    for index, item in enumerate(raw_jobs):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid job entry at index {index}: expected object.")

        job_id = str(item.get("job_id") or f"job-{index + 1}")
        required_vram_mb = int(item.get("required_vram_mb", 0))
        checkpoints = item.get("checkpoints") or []

        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError(f"Job '{job_id}' requires a non-empty checkpoints list.")

        normalized_checkpoints: list[dict[str, Any]] = []
        for cp_index, checkpoint in enumerate(checkpoints):
            if isinstance(checkpoint, str):
                normalized_checkpoints.append(
                    {
                        "name": checkpoint,
                        "required_vram_mb": required_vram_mb,
                    }
                )
                continue
            if not isinstance(checkpoint, dict):
                raise ValueError(
                    f"Job '{job_id}' checkpoint at index {cp_index} must be a string or object."
                )
            normalized_checkpoints.append(
                {
                    "name": str(checkpoint.get("name") or f"checkpoint-{cp_index + 1}"),
                    "required_vram_mb": int(
                        checkpoint.get("required_vram_mb", required_vram_mb)
                    ),
                }
            )

        if required_vram_mb <= 0:
            required_vram_mb = max(
                int(cp.get("required_vram_mb", 0)) for cp in normalized_checkpoints
            )

        jobs.append(
            SchedulerJob(
                job_id=job_id,
                required_vram_mb=required_vram_mb,
                checkpoints=normalized_checkpoints,
            )
        )

    return jobs


def load_scheduler_jobs(jobs_file: Path) -> list[SchedulerJob]:
    try:
        raw = json.loads(jobs_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Scheduler jobs file not found: {jobs_file}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in scheduler jobs file {jobs_file}: {exc}") from exc

    jobs = parse_scheduler_jobs(raw)
    if not jobs:
        raise ValueError("Scheduler jobs file contains no jobs.")
    return jobs


def build_scheduler_workers(receivers: list[dict[str, Any]]) -> list[SchedulerWorker]:
    workers: list[SchedulerWorker] = []
    for receiver in receivers:
        host = str(receiver.get("ip_address") or receiver.get("requested_host") or "").strip()
        if not host:
            continue
        workers.append(
            SchedulerWorker(
                host=host,
                port=int(receiver.get("port") or 50051),
            )
        )
    return workers


def run_scheduler(args) -> None:
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    config = load_sender_config(config_path)
    requested_subnets, _, local_ips = resolve_discovery_context(args, config)

    port = args.port if args.port is not None else config["target_port"]
    timeout = args.timeout if args.timeout is not None else config["scan_timeout_seconds"]
    workers = args.workers if args.workers is not None else config["scan_max_workers"]
    max_hosts = args.max_hosts if args.max_hosts is not None else config["max_scan_hosts"]

    _, reachable_receivers, _ = scan_gpu_nodes(
        subnets=requested_subnets,
        port=port,
        timeout=timeout,
        max_workers=workers,
        max_hosts=max_hosts,
        local_ips=local_ips,
        include_self=args.include_self,
    )

    scheduler_workers = build_scheduler_workers(reachable_receivers)
    if not scheduler_workers:
        raise RuntimeError("No known workers discovered. Cannot start scheduler.")

    jobs_file = Path(args.jobs_file).expanduser()
    if not jobs_file.is_absolute():
        jobs_file = Path.cwd() / jobs_file
    jobs = load_scheduler_jobs(jobs_file)

    state_file = Path(args.scheduler_state_file).expanduser()
    if not state_file.is_absolute():
        state_file = Path.cwd() / state_file

    scheduler = FaultTolerantGpuScheduler(
        jobs=jobs,
        workers=scheduler_workers,
        state_file=state_file,
        worker_timeout_seconds=timeout,
        scheduler_interval_seconds=max(0.2, float(args.scheduler_interval)),
    )
    final_state = asyncio.run(scheduler.run(max_cycles=max(0, int(args.scheduler_cycles))))

    print(
        "[sender] Scheduler finished | state={state} | jobs={count} | state_file={path}".format(
            state=final_state.value,
            count=len(jobs),
            path=state_file,
        )
    )


def receive_line(reader) -> str:
    line = reader.readline()
    if not line:
        raise ConnectionError("Connection closed while reading line.")
    return line.decode("utf-8").rstrip("\n")


def load_json_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object.")
    return data


def load_sender_config(config_path: Path) -> dict[str, Any]:
    raw = load_json_file(config_path)

    target_port = int(raw.get("target_port", 50051))
    discovery_subnet = str(raw.get("discovery_subnet", "192.168.1.0/24"))
    scan_timeout_seconds = float(raw.get("scan_timeout_seconds", 0.35))
    scan_max_workers = int(raw.get("scan_max_workers", 64))
    max_scan_hosts = int(raw.get("max_scan_hosts", 512))
    sender_store_file = str(
        raw.get(
            "sender_discovery_store_file",
            "../data/gpu/sender_gpu_nodes_snapshot.json",
        )
    )

    if target_port <= 0 or target_port > 65535:
        raise ValueError("target_port must be between 1 and 65535.")
    if scan_timeout_seconds <= 0:
        raise ValueError("scan_timeout_seconds must be > 0.")
    if scan_max_workers <= 0:
        raise ValueError("scan_max_workers must be > 0.")
    if max_scan_hosts <= 0:
        raise ValueError("max_scan_hosts must be > 0.")

    return {
        "target_port": target_port,
        "discovery_subnet": discovery_subnet,
        "scan_timeout_seconds": scan_timeout_seconds,
        "scan_max_workers": scan_max_workers,
        "max_scan_hosts": max_scan_hosts,
        "sender_store_file": sender_store_file,
    }


def resolve_output_path(raw_path: str, config_path: Path) -> Path:
    output = Path(raw_path)
    if output.is_absolute():
        return output
    return config_path.parent / output


def split_subnet_values(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    return [part.strip() for part in str(raw_value).split(",") if part.strip()]


def normalize_subnet(raw_subnet: str) -> str:
    network = ipaddress.ip_network(raw_subnet.strip(), strict=False)
    if not isinstance(network, ipaddress.IPv4Network):
        raise ValueError(f"Only IPv4 subnet scanning is supported: {raw_subnet}")
    return str(network)


def is_private_candidate(address: ipaddress.IPv4Address) -> bool:
    return address.is_private and not address.is_loopback and not address.is_link_local


def narrow_auto_scan_network(
    address: ipaddress.IPv4Address,
    network: ipaddress.IPv4Network,
) -> ipaddress.IPv4Network:
    if network.prefixlen < 24:
        return ipaddress.IPv4Network(f"{address}/24", strict=False)
    return network


def discover_windows_private_subnets_and_ips() -> tuple[list[str], list[str]]:
    try:
        proc = subprocess.run(
            ["ipconfig"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return [], []

    subnets: list[str] = []
    local_ips: list[str] = []
    current_ip = None
    current_mask = None

    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()

        ip_match = re.search(r"IPv4[^:]*:\s*([0-9.]+)", line)
        if ip_match:
            current_ip = ip_match.group(1)

        mask_match = re.search(r"Subnet Mask[^:]*:\s*([0-9.]+)", line)
        if mask_match:
            current_mask = mask_match.group(1)

        if not current_ip or not current_mask:
            continue

        try:
            address = ipaddress.IPv4Address(current_ip)
            interface = ipaddress.IPv4Interface(f"{current_ip}/{current_mask}")
        except ValueError:
            current_ip = None
            current_mask = None
            continue

        if is_private_candidate(address):
            local_ips.append(str(address))
            subnets.append(str(narrow_auto_scan_network(address, interface.network)))

        current_ip = None
        current_mask = None

    return subnets, local_ips


def discover_posix_private_subnets_and_ips() -> tuple[list[str], list[str]]:
    try:
        proc = subprocess.run(
            ["ip", "-o", "-f", "inet", "addr", "show"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return [], []

    subnets: list[str] = []
    local_ips: list[str] = []
    for raw_line in proc.stdout.splitlines():
        match = re.search(r"\binet\s+([0-9.]+)/(\d+)\b", raw_line)
        if not match:
            continue

        ip_text, prefix_length = match.groups()
        try:
            address = ipaddress.IPv4Address(ip_text)
            interface = ipaddress.IPv4Interface(f"{ip_text}/{prefix_length}")
        except ValueError:
            continue

        if not is_private_candidate(address):
            continue

        local_ips.append(str(address))
        subnets.append(str(narrow_auto_scan_network(address, interface.network)))

    return subnets, local_ips


def discover_fallback_private_subnet_and_ip() -> tuple[list[str], list[str]]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            ip_text = str(probe.getsockname()[0])
    except OSError:
        return [], []

    try:
        address = ipaddress.IPv4Address(ip_text)
    except ValueError:
        return [], []

    if not is_private_candidate(address):
        return [], []

    subnet = narrow_auto_scan_network(
        address,
        ipaddress.IPv4Network(f"{ip_text}/24", strict=False),
    )
    return [str(subnet)], [str(address)]


def discover_local_private_topology() -> tuple[list[str], set[str]]:
    all_subnets: list[str] = []
    all_ips: set[str] = {"127.0.0.1"}
    seen_subnets: set[str] = set()

    discovered = [
        discover_windows_private_subnets_and_ips(),
        discover_posix_private_subnets_and_ips(),
        discover_fallback_private_subnet_and_ip(),
    ]

    for raw_subnets, raw_ips in discovered:
        for raw_subnet in raw_subnets:
            normalized = normalize_subnet(raw_subnet)
            if normalized in seen_subnets:
                continue
            seen_subnets.add(normalized)
            all_subnets.append(normalized)
        for ip_text in raw_ips:
            all_ips.add(ip_text)

    return all_subnets, all_ips


def send_json_request(host: str, port: int, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        reader = sock.makefile("rb")
        sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        reply_line = receive_line(reader)

    data = json.loads(reply_line)
    if not isinstance(data, dict):
        raise ValueError("Receiver response is not a JSON object.")
    return data


def probe_gpu_node(host: str, port: int, timeout: float) -> dict[str, Any] | None:
    try:
        ping_reply = send_json_request(host, port, {"request": "ping"}, timeout)
    except Exception:
        return None

    try:
        gpu_reply = send_json_request(host, port, {"request": "gpu_info"}, timeout)
    except Exception as exc:
        gpu_reply = {
            "ok": False,
            "gpu_count": 0,
            "gpus": [],
            "source": ping_reply.get("source", "unknown"),
            "message": f"gpu_info request failed: {exc}",
        }

    gpu_cards = []
    for gpu in gpu_reply.get("gpus", []):
        gpu_cards.append(
            {
                "index": gpu.get("index"),
                "name": gpu.get("name"),
                "usage_percent": gpu.get("utilization_gpu_percent"),
                "memory_used_mb": gpu.get("memory_used_mb"),
                "memory_total_mb": gpu.get("memory_total_mb"),
                "cuda_cores": gpu.get("cuda_cores"),
                "compute_capability": gpu.get("compute_capability"),
            }
        )

    return {
        "ip_address": (
            gpu_reply.get("socket_receiver_ip")
            or ping_reply.get("receiver_ip")
            or host
        ),
        "requested_host": host,
        "port": port,
        "ok": bool(gpu_reply.get("ok", False)),
        "service": ping_reply.get("service", "unknown"),
        "hostname": gpu_reply.get("hostname") or ping_reply.get("hostname"),
        "source": gpu_reply.get("source") or ping_reply.get("source", "unknown"),
        "gpu_count": int(gpu_reply.get("gpu_count", 0)),
        "gpu_cards": gpu_cards,
        "ping": ping_reply,
        "gpu_report": gpu_reply,
    }


def list_hosts_in_subnet(subnet: str) -> list[str]:
    network = ipaddress.ip_network(subnet, strict=False)
    if not isinstance(network, ipaddress.IPv4Network):
        raise ValueError(f"Only IPv4 subnet scanning is supported: {subnet}")
    return [str(host) for host in network.hosts()]


def build_scan_hosts(subnets: list[str], max_hosts: int) -> tuple[list[str], list[str]]:
    normalized_subnets: list[str] = []
    hosts: list[str] = []
    seen_hosts: set[str] = set()

    for raw_subnet in subnets:
        subnet = normalize_subnet(raw_subnet)
        if subnet not in normalized_subnets:
            normalized_subnets.append(subnet)
        for host in list_hosts_in_subnet(subnet):
            if host in seen_hosts:
                continue
            seen_hosts.add(host)
            hosts.append(host)

    if len(hosts) > max_hosts:
        raise ValueError(
            "Requested scan covers "
            f"{len(hosts)} hosts across {', '.join(normalized_subnets)}, "
            f"which exceeds max_scan_hosts={max_hosts}."
        )

    return normalized_subnets, hosts


def scan_gpu_nodes(
    subnets: list[str],
    port: int,
    timeout: float,
    max_workers: int,
    max_hosts: int,
    local_ips: set[str],
    include_self: bool,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    scanned_subnets, hosts = build_scan_hosts(subnets, max_hosts)
    if not include_self:
        hosts = [host for host in hosts if host not in local_ips]

    reachable_receivers: list[dict[str, Any]] = []
    worker_count = min(max_workers, max(1, len(hosts)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(probe_gpu_node, host, port, timeout) for host in hosts]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                reachable_receivers.append(result)

    reachable_receivers.sort(key=lambda item: ipaddress.ip_address(item["ip_address"]))
    gpu_nodes = [node for node in reachable_receivers if int(node.get("gpu_count", 0)) > 0]
    return scanned_subnets, reachable_receivers, gpu_nodes


def write_snapshot_file(snapshot: dict[str, Any], store_path: Path) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def resolve_discovery_context(args, config: dict[str, Any]) -> tuple[list[str], str, set[str]]:
    discovered_subnets, local_ips = discover_local_private_topology()

    requested_subnets = split_subnet_values(args.subnet)
    if requested_subnets:
        subnet_source = "argument"
    elif discovered_subnets:
        requested_subnets = discovered_subnets
        subnet_source = "auto-discovered local adapters"
    else:
        requested_subnets = split_subnet_values(config["discovery_subnet"])
        subnet_source = "config fallback"

    if not requested_subnets:
        raise RuntimeError("No subnet available for discovery.")

    return requested_subnets, subnet_source, local_ips


def run_discover_gpus(args) -> None:
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    config = load_sender_config(config_path)
    requested_subnets, subnet_source, local_ips = resolve_discovery_context(args, config)

    port = args.port if args.port is not None else config["target_port"]
    timeout = args.timeout if args.timeout is not None else config["scan_timeout_seconds"]
    workers = args.workers if args.workers is not None else config["scan_max_workers"]
    max_hosts = args.max_hosts if args.max_hosts is not None else config["max_scan_hosts"]

    raw_store_file = args.store_file or config["sender_store_file"]
    store_path = resolve_output_path(raw_store_file, config_path)

    scanned_subnets, reachable_receivers, gpu_nodes = scan_gpu_nodes(
        subnets=requested_subnets,
        port=port,
        timeout=timeout,
        max_workers=workers,
        max_hosts=max_hosts,
        local_ips=local_ips,
        include_self=args.include_self,
    )

    snapshot = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "sender_discover_gpus",
        "subnet_source": subnet_source,
        "scanned_subnets": scanned_subnets,
        "port": port,
        "timeout_seconds": timeout,
        "self_ipv4": sorted(local_ips),
        "reachable_receiver_count": len(reachable_receivers),
        "gpu_node_count": len(gpu_nodes),
        "gpu_nodes": gpu_nodes,
    }
    write_snapshot_file(snapshot, store_path)

    print(
        "[sender] Discovery source: {source} | subnets: {subnets}".format(
            source=subnet_source,
            subnets=", ".join(scanned_subnets),
        )
    )
    print(
        "[sender] Reachable receivers: {reachable} | GPU nodes: {gpu_nodes}".format(
            reachable=len(reachable_receivers),
            gpu_nodes=len(gpu_nodes),
        )
    )
    for node in gpu_nodes:
        print(
            "[sender] {hostname} | {ip}:{port} | gpus={count} | source={source}".format(
                hostname=node.get("hostname") or "unknown-host",
                ip=node.get("ip_address"),
                port=node.get("port"),
                count=node.get("gpu_count"),
                source=node.get("source"),
            )
        )

    print(f"[sender] Stored GPU node snapshot to: {store_path}")


def send_file_to_receiver(
    receiver_host: str,
    port: int,
    file_path: Path,
    chunk_size: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    file_size = file_path.stat().st_size
    metadata = {
        "filename": file_path.name,
        "size": file_size,
        "chunk_size": chunk_size,
    }

    try:
        with socket.create_connection((receiver_host, port), timeout=timeout_seconds) as sock:
            sock.settimeout(timeout_seconds)
            reader = sock.makefile("rb")
            sock.sendall((json.dumps(metadata) + "\n").encode("utf-8"))

            ack = receive_line(reader)
            if ack != "OK":
                return {
                    "ip_address": receiver_host,
                    "port": port,
                    "ok": False,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                    "message": f"Receiver rejected metadata: {ack}",
                }

            sent = 0
            with file_path.open("rb") as in_file:
                while True:
                    chunk = in_file.read(chunk_size)
                    if not chunk:
                        break
                    sock.sendall(chunk)
                    sent += len(chunk)

            reply = json.loads(receive_line(reader))
            ok = bool(reply.get("ok", False)) and sent == file_size
            return {
                "ip_address": receiver_host,
                "port": port,
                "ok": ok,
                "bytes_sent": sent,
                "bytes_received": int(reply.get("bytes_received", 0)),
                "message": reply.get("message", ""),
                "path": reply.get("path"),
            }
    except Exception as exc:
        return {
            "ip_address": receiver_host,
            "port": port,
            "ok": False,
            "bytes_sent": 0,
            "bytes_received": 0,
            "message": str(exc),
        }


def resolve_send_all_file_path(args) -> Path:
    file_path_text = args.file or args.file_path or args.receiver_host
    if not file_path_text:
        raise RuntimeError("File path is required for --send-all. Use --file <path>.")

    file_path = Path(file_path_text).expanduser()
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path


def run_send_all(args) -> None:
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    file_path = resolve_send_all_file_path(args)
    config = load_sender_config(config_path)
    requested_subnets, subnet_source, local_ips = resolve_discovery_context(args, config)

    port = args.port if args.port is not None else config["target_port"]
    timeout = args.timeout if args.timeout is not None else config["scan_timeout_seconds"]
    workers = args.workers if args.workers is not None else config["scan_max_workers"]
    max_hosts = args.max_hosts if args.max_hosts is not None else config["max_scan_hosts"]
    send_timeout = float(args.send_timeout)
    if send_timeout <= 0:
        raise ValueError("send_timeout must be > 0.")

    raw_store_file = args.store_file or config["sender_store_file"]
    store_path = resolve_output_path(raw_store_file, config_path)

    scanned_subnets, reachable_receivers, gpu_nodes = scan_gpu_nodes(
        subnets=requested_subnets,
        port=port,
        timeout=timeout,
        max_workers=workers,
        max_hosts=max_hosts,
        local_ips=local_ips,
        include_self=args.include_self,
    )
    candidate_nodes = gpu_nodes if args.require_gpu else reachable_receivers

    targets: list[tuple[str, int]] = []
    seen_targets: set[tuple[str, int]] = set()
    for node in candidate_nodes:
        target_host = str(node.get("ip_address") or node.get("requested_host") or "").strip()
        target_port = int(node.get("port") or port)
        if not target_host:
            continue
        key = (target_host, target_port)
        if key in seen_targets:
            continue
        seen_targets.add(key)
        targets.append(key)

    if not targets:
        no_target_label = "GPU receiver" if args.require_gpu else "receiver"
        snapshot = {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "sender_send_all",
            "subnet_source": subnet_source,
            "scanned_subnets": scanned_subnets,
            "file_path": str(file_path),
            "file_size_bytes": file_path.stat().st_size,
            "target_filter": "gpu_only" if args.require_gpu else "all_receivers",
            "discovered_receiver_count": len(reachable_receivers),
            "discovered_gpu_node_count": len(gpu_nodes),
            "target_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "results": [],
        }
        write_snapshot_file(snapshot, store_path)
        print(
            "[sender] No reachable {label} nodes found in: {subnets}".format(
                label=no_target_label,
                subnets=", ".join(scanned_subnets),
            )
        )
        print(f"[sender] Stored send-all results to: {store_path}")
        return

    worker_count = min(workers, max(1, len(targets)))
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                send_file_to_receiver,
                receiver_host=target_host,
                port=target_port,
                file_path=file_path,
                chunk_size=args.chunk_size,
                timeout_seconds=send_timeout,
            )
            for target_host, target_port in targets
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: ipaddress.ip_address(item["ip_address"]))
    success_count = sum(1 for item in results if item.get("ok"))
    fail_count = len(results) - success_count

    snapshot = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "sender_send_all",
        "subnet_source": subnet_source,
        "scanned_subnets": scanned_subnets,
        "file_path": str(file_path),
        "file_size_bytes": file_path.stat().st_size,
        "target_filter": "gpu_only" if args.require_gpu else "all_receivers",
        "discovered_receiver_count": len(reachable_receivers),
        "discovered_gpu_node_count": len(gpu_nodes),
        "target_count": len(results),
        "success_count": success_count,
        "failure_count": fail_count,
        "results": results,
    }
    write_snapshot_file(snapshot, store_path)

    print(
        "[sender] Send-all source: {source} | subnets: {subnets}".format(
            source=subnet_source,
            subnets=", ".join(scanned_subnets),
        )
    )
    print(
        "[sender] File: {file_name} | discovered receivers: {receivers} | discovered gpu nodes: {gpu_nodes} | targets: {targets} | success: {ok} | failed: {fail}".format(
            file_name=file_path.name,
            receivers=len(reachable_receivers),
            gpu_nodes=len(gpu_nodes),
            targets=len(results),
            ok=success_count,
            fail=fail_count,
        )
    )
    for result in results:
        print(
            "[sender] {ip}:{port} | ok={ok} | sent={sent} | received={received} | msg={msg}".format(
                ip=result.get("ip_address"),
                port=result.get("port"),
                ok=result.get("ok"),
                sent=result.get("bytes_sent"),
                received=result.get("bytes_received"),
                msg=result.get("message") or "",
            )
        )
    print(f"[sender] Stored send-all results to: {store_path}")


def run_single_gpu_info(receiver_host: str, port: int) -> None:
    with socket.create_connection((receiver_host, port), timeout=30) as sock:
        reader = sock.makefile("rb")
        sock.sendall((json.dumps({"request": "gpu_info"}) + "\n").encode("utf-8"))
        reply = json.loads(receive_line(reader))
        print(json.dumps(reply, indent=2))


def run_file_send(receiver_host: str, file_path_text: str, port: int, chunk_size: int) -> None:
    file_path = Path(file_path_text)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = file_path.stat().st_size
    metadata = {
        "filename": file_path.name,
        "size": file_size,
        "chunk_size": chunk_size,
    }

    with socket.create_connection((receiver_host, port), timeout=30) as sock:
        reader = sock.makefile("rb")
        sock.sendall((json.dumps(metadata) + "\n").encode("utf-8"))

        ack = receive_line(reader)
        if ack != "OK":
            raise RuntimeError(f"Receiver rejected metadata: {ack}")

        sent = 0
        with file_path.open("rb") as in_file:
            while True:
                chunk = in_file.read(chunk_size)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple TCP file sender + network GPU discovery.")
    parser.add_argument("receiver_host", nargs="?", help="Receiver IP/hostname")
    parser.add_argument("file_path", nargs="?", help="Path to file to send (single-receiver mode).")
    parser.add_argument("--port", type=int, default=50051, help="Receiver port (default: 50051)")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024,
        help="Chunk size in bytes (default: 1048576)",
    )
    parser.add_argument(
        "--gpu-info",
        action="store_true",
        help="Request GPU details from one receiver_host.",
    )
    parser.add_argument(
        "--discover-gpus",
        action="store_true",
        help="Discover other GPU nodes in subnet(s) using one thread per host and store JSON snapshot.",
    )
    parser.add_argument(
        "--send-all",
        action="store_true",
        help=(
            "Discover GPU receiver nodes and send one file to all targets concurrently "
            "(one thread per receiver)."
        ),
    )
    parser.add_argument(
        "--schedule-jobs",
        action="store_true",
        help=(
            "Run fault-tolerant GPU scheduler: monitor workers, rollback failed worker jobs "
            "to last checkpoint, and resume queued work on free VRAM."
        ),
    )
    parser.add_argument(
        "--file",
        default="",
        help="File path used by --send-all mode.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to sender discovery config JSON (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--subnet",
        default=None,
        help=(
            "Optional CIDR subnet, or comma-separated subnets. "
            "If omitted, local private subnets are auto-discovered."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-host discovery timeout for --discover-gpus/--send-all. Defaults to config.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Thread count for --discover-gpus/--send-all. Defaults to config scan_max_workers.",
    )
    parser.add_argument(
        "--max-hosts",
        type=int,
        default=None,
        help="Host scan limit for --discover-gpus/--send-all. Defaults to config max_scan_hosts.",
    )
    parser.add_argument(
        "--store-file",
        default=None,
        help=(
            "Output JSON path for --discover-gpus/--send-all snapshots. "
            "Defaults to config sender_discovery_store_file."
        ),
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include local machine IP(s) in --discover-gpus/--send-all target set.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="For --send-all, target only receivers that report gpu_count > 0.",
    )
    parser.add_argument(
        "--send-timeout",
        type=float,
        default=30.0,
        help="Per-target file send timeout in seconds for --send-all (default: 30.0).",
    )
    parser.add_argument(
        "--jobs-file",
        default="data/gpu/jobs_queue.json",
        help="JSON file containing scheduler jobs and checkpoints.",
    )
    parser.add_argument(
        "--scheduler-state-file",
        default="data/gpu/scheduler_state.json",
        help="Output JSON file where scheduler state is persisted.",
    )
    parser.add_argument(
        "--scheduler-interval",
        type=float,
        default=1.0,
        help="Scheduler loop interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--scheduler-cycles",
        type=int,
        default=0,
        help="Optional max scheduler cycles (0 runs until COMPLETED/FAILED).",
    )
    args = parser.parse_args()

    if args.send_all and args.discover_gpus:
        parser.error("--send-all cannot be combined with --discover-gpus.")
    if args.send_all and args.gpu_info:
        parser.error("--send-all cannot be combined with --gpu-info.")
    if args.schedule_jobs and args.discover_gpus:
        parser.error("--schedule-jobs cannot be combined with --discover-gpus.")
    if args.schedule_jobs and args.send_all:
        parser.error("--schedule-jobs cannot be combined with --send-all.")
    if args.schedule_jobs and args.gpu_info:
        parser.error("--schedule-jobs cannot be combined with --gpu-info.")

    if args.discover_gpus:
        run_discover_gpus(args)
        return

    if args.send_all:
        run_send_all(args)
        return

    if args.schedule_jobs:
        run_scheduler(args)
        return

    if not args.receiver_host:
        parser.error("receiver_host is required unless --discover-gpus or --send-all is used.")

    if args.gpu_info:
        run_single_gpu_info(args.receiver_host, args.port)
        return

    if not args.file_path:
        parser.error("file_path is required unless --gpu-info, --discover-gpus, or --send-all is used.")

    run_file_send(
        receiver_host=args.receiver_host,
        file_path_text=args.file_path,
        port=args.port,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
