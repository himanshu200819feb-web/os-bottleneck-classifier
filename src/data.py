"""Download, verify and load the Server Machine Dataset (SMD).

SMD (Su et al., KDD 2019, OmniAnomaly) has 28 production servers, 38
anonymised metrics each, sampled once a minute. For every machine there is
an unlabelled `train` half (mostly normal), a `test` half, per-point
`test_label` (0/1) and `interpretation_label` (which metrics caused each
anomaly, 1-indexed).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

N_METRICS = 38
SUBDIRS = ("train", "test", "test_label", "interpretation_label")
ALL_MACHINES = (
    [f"machine-1-{i}" for i in range(1, 9)]
    + [f"machine-2-{i}" for i in range(1, 10)]
    + [f"machine-3-{i}" for i in range(1, 12)]
)
METRIC_NAMES = [f"m{i:02d}" for i in range(1, N_METRICS + 1)]  # SMD does not publish real names


@dataclass
class Machine:
    name: str
    train: np.ndarray            # (T_train, 38) float32
    test: np.ndarray             # (T_test, 38) float32
    labels: np.ndarray           # (T_test,) int8
    interpretation: list = field(default_factory=list)  # [(start, end, [metric_idx0, ...])]


def machine_list(cfg: dict) -> list[str]:
    machines = cfg["data"].get("machines", "all")
    if machines == "all":
        return list(ALL_MACHINES)
    unknown = set(machines) - set(ALL_MACHINES)
    if unknown:
        raise ValueError(f"Unknown machines in config: {sorted(unknown)}")
    return list(machines)


# ---------------------------------------------------------------- download
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    tmp.replace(dest)


def download(cfg: dict, raw_dir: Path, workers: int = 8) -> dict:
    """Download the configured machines (skips files already present) and
    write MANIFEST.json with a SHA-256 per file plus one combined digest."""
    base = cfg["data"]["source_url"].rstrip("/")
    jobs = [(f"{base}/{sub}/{m}.txt", raw_dir / sub / f"{m}.txt")
            for m in machine_list(cfg) for sub in SUBDIRS]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda job: _fetch(*job), jobs))
    return write_manifest(raw_dir, machine_list(cfg))


def write_manifest(raw_dir: Path, machines: list[str]) -> dict:
    files = {f"{sub}/{m}.txt": _sha256(raw_dir / sub / f"{m}.txt")
             for m in machines for sub in SUBDIRS}
    combined = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()
    manifest = {"dataset": "SMD", "machines": machines, "combined_sha256": combined, "files": files}
    (raw_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def dataset_fingerprint(raw_dir: Path, machines: list[str]) -> str:
    """Combined checksum of exactly the files this run uses."""
    return write_manifest(raw_dir, machines)["combined_sha256"]


# ---------------------------------------------------------------- loading
def _read_matrix(path: Path) -> np.ndarray:
    arr = pd.read_csv(path, header=None, dtype=np.float32).to_numpy()
    if arr.ndim != 2 or arr.shape[1] != N_METRICS:
        raise ValueError(f"{path}: expected {N_METRICS} columns, got shape {arr.shape}")
    return arr


def _read_interpretation(path: Path) -> list:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        span, dims = line.split(":", 1)
        start, end = (int(x) for x in span.split("-"))
        idx = [int(d) - 1 for d in dims.split(",") if d.strip()]
        out.append((start, end, idx))
    return out


def load_machine(raw_dir: Path, name: str) -> Machine:
    train = _read_matrix(raw_dir / "train" / f"{name}.txt")
    test = _read_matrix(raw_dir / "test" / f"{name}.txt")
    labels = pd.read_csv(raw_dir / "test_label" / f"{name}.txt", header=None).to_numpy().ravel().astype(np.int8)
    if len(labels) != len(test):
        raise ValueError(f"{name}: {len(labels)} labels for {len(test)} test rows")
    interp = _read_interpretation(raw_dir / "interpretation_label" / f"{name}.txt")
    return Machine(name, train, test, labels, interp)


def load_all(raw_dir: Path, machines: list[str]) -> dict[str, Machine]:
    missing = [m for m in machines if not (raw_dir / "test" / f"{m}.txt").exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} machine(s) not downloaded (e.g. {missing[0]}). "
            "Run: python -m src.pipeline download")
    return {m: load_machine(raw_dir, m) for m in machines}


def segments(mask: np.ndarray, gap: int = 0) -> list[tuple[int, int]]:
    """Contiguous runs of True as inclusive (start, end) pairs; runs separated
    by <= `gap` False points are merged."""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))
