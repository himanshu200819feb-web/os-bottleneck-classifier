"""Collect independent, grouped OS telemetry runs for bottleneck classification.

Displays live per-run and whole-dataset progress, row targets, elapsed time,
ETA, collection phase, and selected resource measurements.
"""

import csv
import hashlib
import math
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from multiprocessing import Event, Process, cpu_count, freeze_support
from multiprocessing.synchronize import Event as EventType

import psutil
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


CONFIG = load_config()
SEED = CONFIG["project"]["random_seed"]
PATHS = CONFIG["paths"]
TELEMETRY = CONFIG["telemetry"]
WORKLOADS = CONFIG["workloads"]
OUTPUT_CSV = PATHS["raw_csv"]
SAMPLE_INTERVAL = float(TELEMETRY["sampling_interval_sec"])
DURATION = int(TELEMETRY["duration_per_run_sec"])
WARMUP = int(TELEMETRY["warmup_drop_sec"])
COOLDOWN = int(TELEMETRY["cooldown_drop_sec"])
SETTLE = float(TELEMETRY["settle_time_sec"])
FLUSH_ROWS = int(TELEMETRY["csv_flush_interval_rows"])
IO_TEMP_FILE = os.path.join(tempfile.gettempdir(), "os_telemetry_io_buffer.bin")

CSV_HEADERS = [
    "timestamp", "capture_session_id", "run_id", "machine_id",
    "workload_intensity", "cpu_user_pct", "cpu_system_pct",
    "cpu_iowait_pct", "cpu_idle_pct", "load_avg_1m_per_cpu",
    "context_switches_per_sec", "interrupts_per_sec", "mem_used_pct",
    "mem_available_mb", "swap_used_pct", "swap_in_kb_per_sec",
    "swap_out_kb_per_sec", "disk_read_kb_per_sec",
    "disk_write_kb_per_sec", "disk_busy_pct", "net_rx_kb_per_sec",
    "net_tx_kb_per_sec", "label",
]


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def progress_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    ratio = min(1.0, max(0.0, current / total))
    filled = int(round(ratio * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def clear_progress_line() -> None:
    print("\r" + " " * 180 + "\r", end="", flush=True)


def machine_id() -> str:
    identity = f"{platform.node()}|{platform.system()}|{platform.machine()}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]


def safe_delta(current: float, previous: float) -> float:
    return max(0.0, float(current) - float(previous))


def cpu_worker(stop_flag: EventType) -> None:
    value = 1
    while not stop_flag.is_set():
        for number in range(1, 15000):
            value = (value * number + 17) % 10000019


def memory_worker(
    stop_flag: EventType,
    target_fraction: float,
    max_alloc_gb: float,
    reserve_gb: float,
    page_size: int,
    recheck_seconds: float,
) -> None:
    available = psutil.virtual_memory().available
    reserve = int(reserve_gb * 1024**3)
    ceiling = int(max_alloc_gb * 1024**3)
    target = min(int(available * target_fraction), ceiling, max(0, available - reserve))
    if target < 256 * 1024**2:
        return

    try:
        data = bytearray(target)
        next_check = time.perf_counter() + recheck_seconds
        while not stop_flag.is_set():
            for offset in range(0, target, page_size):
                data[offset] = (data[offset] + 1) % 256
                if offset % (page_size * 8192) == 0 and stop_flag.is_set():
                    break
            if time.perf_counter() >= next_check:
                if psutil.virtual_memory().available < reserve // 2:
                    break
                next_check = time.perf_counter() + recheck_seconds
    except (MemoryError, OSError):
        while not stop_flag.is_set():
            time.sleep(0.2)


def io_worker(
    stop_flag: EventType,
    path: str,
    chunk_size_mb: int,
    writes_per_cycle: int,
    buffer_size: int,
    cycle_sleep: float,
) -> None:
    chunk = os.urandom(chunk_size_mb * 1024**2)
    try:
        while not stop_flag.is_set():
            with open(path, "wb", buffering=buffer_size) as file:
                for _ in range(writes_per_cycle):
                    if stop_flag.is_set():
                        break
                    file.write(chunk)
                file.flush()
                try:
                    os.fsync(file.fileno())
                except OSError:
                    pass
            time.sleep(cycle_sleep)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def start_workers(workload: str | None, intensity: str, stop_flag: EventType) -> list[Process]:
    if workload == "cpu":
        fraction = float(WORKLOADS["cpu"]["worker_fraction"][intensity])
        count = max(1, math.ceil(cpu_count() * fraction))
        print(f"    Workers : {count} CPU saturation process(es)")
        return [Process(target=cpu_worker, args=(stop_flag,), daemon=True) for _ in range(count)]

    if workload == "memory":
        cfg = WORKLOADS["memory"]
        fraction = float(cfg["available_fraction"][intensity])
        print(
            f"    Workers : 1 memory-pressure process "
            f"(target={fraction:.0%} available RAM, max={cfg['max_alloc_gb']} GiB)"
        )
        args = (
            stop_flag, fraction, float(cfg["max_alloc_gb"]),
            float(cfg["reserve_available_gb"]), int(cfg["page_size_bytes"]),
            float(cfg["recheck_interval_sec"]),
        )
        return [Process(target=memory_worker, args=args, daemon=True)]

    if workload == "io":
        cfg = WORKLOADS["io"]
        writes = int(cfg["writes_per_cycle"][intensity])
        cycle_mb = int(cfg["chunk_size_mb"]) * writes
        print(f"    Workers : 1 disk-I/O process ({cycle_mb} MiB written per cycle)")
        args = (
            stop_flag, IO_TEMP_FILE, int(cfg["chunk_size_mb"]), writes,
            int(cfg["buffer_size_bytes"]), float(cfg["cycle_sleep_sec"][intensity]),
        )
        return [Process(target=io_worker, args=args, daemon=True)]

    print("    Workers : none (idle/normal observation)")
    return []


def stop_workers(workers: list[Process], stop_flag: EventType) -> None:
    stop_flag.set()
    for worker in workers:
        worker.join(timeout=3.0)
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1.0)
    for worker in workers:
        if worker.is_alive() and hasattr(worker, "kill"):
            worker.kill()
    try:
        os.remove(IO_TEMP_FILE)
    except OSError:
        pass


def display_live_progress(
    run_number: int,
    total_runs: int,
    label: str,
    intensity: str,
    phase: str,
    elapsed: float,
    run_rows: int,
    rows_per_run: int,
    completed_rows_before_run: int,
    total_target_rows: int,
    collection_start: float,
    estimated_total_seconds: float,
    cpu_user: float,
    memory_used: float,
    disk_write: float,
) -> None:
    overall_rows = completed_rows_before_run + run_rows
    row_percent = 100.0 * overall_rows / max(1, total_target_rows)
    run_percent = 100.0 * run_rows / max(1, rows_per_run)
    wall_elapsed = time.perf_counter() - collection_start
    eta = max(0.0, estimated_total_seconds - wall_elapsed)

    line_one = (
        f"\r    {phase:<8} Run {run_number:02d}/{total_runs:02d} "
        f"{progress_bar(run_rows, rows_per_run)} "
        f"{run_rows:4d}/{rows_per_run:<4d} rows ({run_percent:6.2f}%) | "
        f"Total {overall_rows:5d}/{total_target_rows:<5d} ({row_percent:6.2f}%) | "
        f"Elapsed {format_duration(wall_elapsed)} | ETA {format_duration(eta)} | "
        f"CPU {cpu_user:5.1f}% MEM {memory_used:5.1f}% WRITE {disk_write:9.0f} KB/s"
    )
    print(line_one[:220].ljust(220), end="", flush=True)


def capture_run(
    writer: csv.writer,
    csv_file,
    session_id: str,
    run_id: str,
    host_id: str,
    label: str,
    intensity: str,
    stop_flag: EventType,
    run_number: int,
    total_runs: int,
    completed_rows_before_run: int,
    total_target_rows: int,
    collection_start: float,
    estimated_total_seconds: float,
) -> int:
    rows_per_run = max(1, int((DURATION - WARMUP - COOLDOWN) / SAMPLE_INTERVAL))
    previous_disk = psutil.disk_io_counters()
    previous_net = psutil.net_io_counters()
    previous_cpu_stats = psutil.cpu_stats()
    previous_swap = psutil.swap_memory()
    psutil.cpu_times_percent(interval=None)

    start = time.perf_counter()
    previous_time = start
    next_sample = start
    written = 0
    since_flush = 0

    display_live_progress(
        run_number, total_runs, label, intensity, "STARTING", 0.0, 0,
        rows_per_run, completed_rows_before_run, total_target_rows,
        collection_start, estimated_total_seconds, 0.0, 0.0, 0.0,
    )

    while written < rows_per_run:
        next_sample += SAMPLE_INTERVAL
        delay = next_sample - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

        now = time.perf_counter()
        elapsed = now - start
        dt = now - previous_time
        if elapsed >= DURATION or dt <= 0:
            break

        cpu = psutil.cpu_times_percent(interval=None)
        current_cpu_stats = psutil.cpu_stats()
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        current_disk = psutil.disk_io_counters()
        current_net = psutil.net_io_counters()

        context_switches = safe_delta(current_cpu_stats.ctx_switches, previous_cpu_stats.ctx_switches) / dt
        interrupts = safe_delta(current_cpu_stats.interrupts, previous_cpu_stats.interrupts) / dt
        swap_in = safe_delta(getattr(swap, "sin", 0), getattr(previous_swap, "sin", 0)) / 1024 / dt
        swap_out = safe_delta(getattr(swap, "sout", 0), getattr(previous_swap, "sout", 0)) / 1024 / dt

        if current_disk and previous_disk:
            disk_read = safe_delta(current_disk.read_bytes, previous_disk.read_bytes) / 1024 / dt
            disk_write = safe_delta(current_disk.write_bytes, previous_disk.write_bytes) / 1024 / dt
            busy_now = getattr(current_disk, "busy_time", None)
            busy_before = getattr(previous_disk, "busy_time", None)
            disk_busy = (
                min(100.0, safe_delta(busy_now, busy_before) / (dt * 1000) * 100)
                if busy_now is not None and busy_before is not None else 0.0
            )
        else:
            disk_read = disk_write = disk_busy = 0.0

        if current_net and previous_net:
            net_rx = safe_delta(current_net.bytes_recv, previous_net.bytes_recv) / 1024 / dt
            net_tx = safe_delta(current_net.bytes_sent, previous_net.bytes_sent) / 1024 / dt
        else:
            net_rx = net_tx = 0.0

        previous_cpu_stats = current_cpu_stats
        previous_swap = swap
        previous_disk = current_disk
        previous_net = current_net
        previous_time = now

        phase = "WARMUP" if elapsed < WARMUP else "CAPTURE"
        if WARMUP <= elapsed <= DURATION - COOLDOWN:
            try:
                load_per_cpu = psutil.getloadavg()[0] / max(1, cpu_count())
            except (AttributeError, OSError):
                load_per_cpu = psutil.cpu_percent(interval=None) / 100.0

            writer.writerow([
                datetime.now(timezone.utc).isoformat(), session_id, run_id, host_id,
                intensity, round(cpu.user, 2), round(cpu.system, 2),
                round(getattr(cpu, "iowait", 0.0), 2), round(cpu.idle, 2),
                round(load_per_cpu, 4), round(context_switches, 1), round(interrupts, 1),
                round(vm.percent, 2), round(vm.available / 1024**2, 2), round(swap.percent, 2),
                round(swap_in, 2), round(swap_out, 2), round(disk_read, 2),
                round(disk_write, 2), round(disk_busy, 2), round(net_rx, 2),
                round(net_tx, 2), label,
            ])
            written += 1
            since_flush += 1
            if since_flush >= FLUSH_ROWS:
                csv_file.flush()
                since_flush = 0

        display_live_progress(
            run_number, total_runs, label, intensity, phase, elapsed, written,
            rows_per_run, completed_rows_before_run, total_target_rows,
            collection_start, estimated_total_seconds, cpu.user, vm.percent, disk_write,
        )

    csv_file.flush()
    clear_progress_line()
    return written


def build_schedule() -> list[tuple[str, str | None, str, int]]:
    class_to_workload = {
        "Normal": None,
        "CPU_Bound": "cpu",
        "Memory_Pressure": "memory",
        "IO_Contention": "io",
    }
    configured = TELEMETRY["classes"]
    if set(configured) != set(class_to_workload):
        raise ValueError(f"telemetry.classes must be exactly {sorted(class_to_workload)}")

    profiles = list(TELEMETRY["intensity_profiles"])
    schedule = []
    for repetition in range(1, int(TELEMETRY["runs_per_class"]) + 1):
        block = []
        for class_index, label in enumerate(configured):
            intensity = (
                "idle" if label == "Normal"
                else profiles[(repetition + class_index - 1) % len(profiles)]
            )
            block.append((label, class_to_workload[label], intensity, repetition))
        if TELEMETRY.get("randomize_run_order", True):
            random.shuffle(block)
        schedule.extend(block)
    return schedule


def main() -> None:
    random.seed(SEED)
    if DURATION <= WARMUP + COOLDOWN:
        raise ValueError("duration_per_run_sec must exceed warmup + cooldown")
    if SAMPLE_INTERVAL <= 0:
        raise ValueError("sampling_interval_sec must be positive")

    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        answer = input(f"Overwrite existing dataset '{OUTPUT_CSV}'? Type YES: ").strip()
        if answer != "YES":
            print("Collection cancelled; existing dataset preserved.")
            return

    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    host_id = machine_id()
    schedule = build_schedule()
    total_runs = len(schedule)
    rows_per_run = max(1, int((DURATION - WARMUP - COOLDOWN) / SAMPLE_INTERVAL))
    total_target_rows = rows_per_run * total_runs
    active_seconds_per_run = WARMUP + rows_per_run * SAMPLE_INTERVAL
    estimated_total_seconds = (
        total_runs * active_seconds_per_run + max(0, total_runs - 1) * SETTLE
    )

    print("=" * 100)
    print("                 OS TELEMETRY GROUPED DATASET GENERATION PIPELINE")
    print("=" * 100)
    print(f" Output CSV          : {OUTPUT_CSV}")
    print(f" Session ID          : {session_id}")
    print(f" Machine ID          : {host_id}")
    print(f" Classes             : {', '.join(TELEMETRY['classes'])}")
    print(f" Runs per class      : {TELEMETRY['runs_per_class']}")
    print(f" Total independent runs: {total_runs}")
    print(f" Sampling interval   : {SAMPLE_INTERVAL:.2f} second(s)")
    print(f" Run timing          : {DURATION}s configured; {WARMUP}s warmup; {COOLDOWN}s excluded")
    print(f" Rows per run        : {rows_per_run}")
    print(f" TARGET TOTAL ROWS   : {total_target_rows}")
    print(f" Inter-run settle    : {SETTLE:.1f} second(s)")
    print(f" Estimated duration  : {format_duration(estimated_total_seconds)}")
    print("=" * 100)

    collection_start = time.perf_counter()
    completed_rows = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADERS)

        for index, (label, workload, intensity, repetition) in enumerate(schedule, start=1):
            run_id = f"{session_id}-{label.lower()}-{repetition:02d}-{uuid.uuid4().hex[:6]}"
            print("\n" + "-" * 100)
            print(
                f"[*] RUN {index}/{total_runs} | Class={label} | "
                f"Intensity={intensity} | Repetition={repetition}"
            )
            print(f"    Run ID  : {run_id}")
            print(f"    Row goal: {rows_per_run} | Dataset goal: {total_target_rows}")

            stop_flag = Event()
            workers = start_workers(workload, intensity, stop_flag)
            run_started = time.perf_counter()
            try:
                for worker in workers:
                    worker.start()
                rows = capture_run(
                    writer, csv_file, session_id, run_id, host_id,
                    label, intensity, stop_flag, index, total_runs,
                    completed_rows, total_target_rows, collection_start,
                    estimated_total_seconds,
                )
            finally:
                stop_workers(workers, stop_flag)

            completed_rows += rows
            run_elapsed = time.perf_counter() - run_started
            overall_percent = completed_rows / max(1, total_target_rows) * 100
            print(
                f"[OK] Run {index}/{total_runs} completed: {rows}/{rows_per_run} rows | "
                f"run time={format_duration(run_elapsed)} | cumulative="
                f"{completed_rows}/{total_target_rows} ({overall_percent:.2f}%)"
            )

            if rows != rows_per_run:
                print(
                    f"[WARNING] Run produced {rows_per_run - rows} fewer rows than expected. "
                    "Inspect system load or sampling configuration."
                )

            if index < total_runs:
                print(f"[~] Settling system for {SETTLE:.1f}s before the next run...", flush=True)
                settle_start = time.perf_counter()
                while time.perf_counter() - settle_start < SETTLE:
                    remaining = SETTLE - (time.perf_counter() - settle_start)
                    print(
                        f"\r    Next run in {max(0, math.ceil(remaining)):3d}s | "
                        f"Dataset rows {completed_rows}/{total_target_rows}",
                        end="", flush=True,
                    )
                    time.sleep(min(1.0, max(0.05, remaining)))
                clear_progress_line()

    total_elapsed = time.perf_counter() - collection_start
    completion_percent = completed_rows / max(1, total_target_rows) * 100
    print("\n" + "=" * 100)
    print("[SUCCESS] DATASET GENERATION COMPLETE")
    print(f" Output              : {OUTPUT_CSV}")
    print(f" Completed runs      : {total_runs}/{total_runs}")
    print(f" Rows written        : {completed_rows}/{total_target_rows} ({completion_percent:.2f}%)")
    print(f" Total elapsed time  : {format_duration(total_elapsed)}")
    print(f" Average rows/run    : {completed_rows / max(1, total_runs):.1f}")
    print(" Next step           : python preprocess.py")
    print("=" * 100)


if __name__ == "__main__":
    freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        clear_progress_line()
        print("\n[!] Collection interrupted. Partial CSV retained for inspection.")
        sys.exit(1)