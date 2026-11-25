"""One entry point for the whole project.

    python -m src.pipeline download   [--config config.yaml]
    python -m src.pipeline train      [--config config.yaml]
    python -m src.pipeline all        [--config config.yaml]   # download + train

Each `train` writes runs/<run_id>/ containing a copy of the config, a
manifest (seed, dataset checksum, library versions, timings), the leaderboard,
per-fold metrics and out-of-fold scores for every machine.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
import warnings
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import yaml
from sklearn.exceptions import ConvergenceWarning

from . import data
from .config import load_config, resolve, PROJECT_ROOT
from .evaluate import add_paper_f1, cross_validate, paper_f1
from .models import enabled_models
from .rca import localisation_eval

PACKAGES = ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "flask", "google-genai", "pydantic", "pyyaml"]


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# Neural nets stop at max_iter by design (early stopping usually ends sooner); the warning is noise.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="max_samples .* is greater than the total number of samples")


def cmd_download(cfg: dict) -> dict:
    raw = resolve(cfg["data"]["raw_dir"])
    machines = data.machine_list(cfg)
    print(f"Downloading {len(machines)} SMD machines to {raw.relative_to(PROJECT_ROOT)} ...")
    t0 = time.time()
    manifest = data.download(cfg, raw)
    print(f"Done in {time.time() - t0:.0f}s. Dataset SHA-256: {manifest['combined_sha256']}")
    return manifest


def cmd_train(cfg: dict, config_arg: str) -> Path:
    seed_everything(cfg["seed"])
    raw = resolve(cfg["data"]["raw_dir"])
    names = data.machine_list(cfg)
    machines = data.load_all(raw, names)
    checksum = data.dataset_fingerprint(raw, names)
    model_keys = enabled_models(cfg)
    print(f"Loaded {len(machines)} machines; dataset SHA-256 {checksum[:16]}...; models: {', '.join(model_keys)}")

    t0 = time.time()
    result = cross_validate(cfg, machines, model_keys)
    elapsed = time.time() - t0

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = resolve(cfg["runs"]["dir"]) / run_id
    (out / "scores").mkdir(parents=True, exist_ok=True)

    clean_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    (out / "config.yaml").write_text(yaml.safe_dump(clean_cfg, sort_keys=False), encoding="utf-8")

    total_points = int(sum(len(m.labels) for m in machines.values()))
    total_anom = int(sum(int(m.labels.sum()) for m in machines.values()))
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": f"python -m src.pipeline train --config {config_arg}",
        "config_file": cfg.get("_source"),
        "seed": cfg["seed"],
        "dataset": {"name": "SMD (Server Machine Dataset)", "sha256": checksum,
                    "machines": names, "n_machines": len(names), "n_metrics": data.N_METRICS,
                    "test_points": total_points, "anomaly_points": total_anom,
                    "anomaly_rate": total_anom / max(total_points, 1),
                    "train_points": int(sum(len(m.train) for m in machines.values()))},
        "folds": result["folds"],
        "models": model_keys,
        "features": {**cfg["features"],
                     "n_features": data.N_METRICS * (3 + len(cfg["features"].get("extras", [])))},
        "duration_seconds": round(elapsed, 1),
        "versions": _versions(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "git_commit": _git_commit(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "leaderboard.json").write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    (out / "folds.json").write_text(json.dumps(result["per_fold"], indent=2), encoding="utf-8")
    (out / "machines.json").write_text(json.dumps({
        m: {"fold": result["fold_of"][m], "thresholds": result["thresholds"][m],
            "test_points": int(len(machines[m].labels)), "anomaly_points": int(machines[m].labels.sum()),
            "metrics": result["machine_metrics"][m]}
        for m in names}, indent=2), encoding="utf-8")
    (out / "curves.json").write_text(json.dumps(result["curves"]), encoding="utf-8")
    print("Evaluating RCA metric ranking against SMD interpretation labels...")
    rca_eval = localisation_eval(machines, cfg["rca"].get("top_k_metrics", 6))
    (out / "rca_eval.json").write_text(json.dumps(rca_eval, indent=2), encoding="utf-8")
    for m in names:
        np.savez_compressed(out / "scores" / f"{m}.npz", **result["oof_scores"][m])

    (resolve(cfg["runs"]["dir"]) / "LATEST").write_text(run_id, encoding="utf-8")
    best = result["summary"][0]
    print(f"\nRun {run_id} saved to runs/{run_id} in {elapsed:.0f}s")
    print(f"Best model by cross-validated F1: {best['label']} (F1 {best['f1']:.3f} ± {best['f1_std']:.3f})")
    return out


def cmd_rescore(cfg: dict, run_id: str | None) -> None:
    """Recompute the per-server paper-style F1 for an existing run from its saved
    scores (no retraining). Deterministic: same scores + labels give the same numbers."""
    runs = resolve(cfg["runs"]["dir"])
    run_id = run_id or (runs / "LATEST").read_text().strip()
    rd = runs / run_id
    run_cfg = yaml.safe_load((rd / "config.yaml").read_text(encoding="utf-8"))
    raw = resolve(run_cfg["data"]["raw_dir"])
    meta = json.loads((rd / "machines.json").read_text(encoding="utf-8"))
    for m, info in meta.items():
        y = data.load_machine(raw, m).labels
        with np.load(rd / "scores" / f"{m}.npz") as z:
            for k in z.files:
                info["metrics"].setdefault(k, {})["paper_f1"] = paper_f1(y, z[k])
    lb = json.loads((rd / "leaderboard.json").read_text(encoding="utf-8"))
    add_paper_f1(lb, {m: info["metrics"] for m, info in meta.items()})
    (rd / "machines.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (rd / "leaderboard.json").write_text(json.dumps(lb, indent=2), encoding="utf-8")
    print(f"Rescored run {run_id}:")
    for r in lb:
        print(f"  {r['label']:<24} strict F1 {r['f1']:.3f}   paper-style F1 {r['paper_f1']:.3f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="OS Anomaly Analytics pipeline")
    ap.add_argument("command", choices=["download", "train", "all", "rescore"])
    ap.add_argument("--run", help="run id for `rescore` (default: latest)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", help="comma-separated subset of models to run (for quick experiments)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.models:
        keep = {m.strip() for m in args.models.split(",")}
        unknown = keep - set(cfg["models"])
        if unknown:
            ap.error(f"unknown models: {sorted(unknown)}")
        for k, v in cfg["models"].items():
            v["enabled"] = k in keep
    if args.command in ("download", "all"):
        cmd_download(cfg)
    if args.command in ("train", "all"):
        cmd_train(cfg, args.config)
    if args.command == "rescore":
        cmd_rescore(cfg, args.run)


if __name__ == "__main__":
    sys.exit(main())
