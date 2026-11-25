"""Flask API + static dashboard. Read-only over the training results: it never
trains anything, it only reads what `python -m src.pipeline train` wrote to
runs/<run_id>/. The only thing it writes is the RCA cache (runs/<run_id>/rca_cache/).

    python -m dashboard.server            # http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request, send_from_directory

from src import data
from src.config import PROJECT_ROOT, load_config, resolve
from src.data import METRIC_NAMES, segments
from src.evaluate import model_meta
from src.rca import analyse, build_evidence, detect_incidents, gemini_status

load_dotenv(PROJECT_ROOT / ".env")
STATIC = Path(__file__).parent / "static"
app = Flask(__name__, static_folder=None)
CFG = load_config()


# ------------------------------------------------------------------ run access
def runs_dir() -> Path:
    return resolve(CFG["runs"]["dir"])


def list_runs() -> list[str]:
    d = runs_dir()
    if not d.exists():
        return []
    return sorted((p.name for p in d.iterdir() if (p / "manifest.json").exists()), reverse=True)


def latest_run() -> str | None:
    pointer = runs_dir() / "LATEST"
    if pointer.exists() and (runs_dir() / pointer.read_text().strip() / "manifest.json").exists():
        return pointer.read_text().strip()
    runs = list_runs()
    return runs[0] if runs else None


def run_path(run_id: str) -> Path:
    if run_id not in list_runs():
        abort(404, f"Unknown run {run_id}")
    return runs_dir() / run_id


@lru_cache(maxsize=16)
def run_json(run_id: str, name: str, default=None):
    p = run_path(run_id) / name
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def run_config(run_id: str) -> dict:
    return yaml.safe_load((run_path(run_id) / "config.yaml").read_text(encoding="utf-8"))


@lru_cache(maxsize=32)
def machine(run_id: str, name: str) -> data.Machine:
    raw = resolve(run_config(run_id)["data"]["raw_dir"])
    return data.load_machine(raw, name)


@lru_cache(maxsize=64)
def scores(run_id: str, name: str) -> dict:
    with np.load(run_path(run_id) / "scores" / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


def labels_of(keys) -> dict:
    return {k: model_meta(k)["label"] for k in keys}


def bucket(arr: np.ndarray, size: int, how: str) -> np.ndarray:
    if size <= 1:
        return arr
    n = len(arr) // size * size
    head = arr[:n].reshape(-1, size, *arr.shape[1:])
    agg = {"max": head.max(axis=1), "mean": head.mean(axis=1)}[how]
    if n < len(arr):
        tail = arr[n:]
        agg = np.concatenate([agg, (tail.max(axis=0) if how == "max" else tail.mean(axis=0))[None]])
    return agg


def clean(v):
    """JSON-safe values (NaN -> None, numpy -> python)."""
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


def any_in_ranges(mask: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    c = np.concatenate([[0], np.cumsum(mask.astype(np.int32))])
    return np.array([c[b + 1] - c[a] > 0 for a, b in ranges], dtype=bool)


# ------------------------------------------------------------------ API
@app.get("/api/runs")
def api_runs():
    return jsonify({"runs": list_runs(), "latest": latest_run()})


@app.get("/api/status")
def api_status():
    return jsonify({"gemini": gemini_status(CFG)})


@app.get("/api/run/<run_id>")
def api_run(run_id):
    return jsonify(clean({
        "manifest": run_json(run_id, "manifest.json"),
        "leaderboard": run_json(run_id, "leaderboard.json"),
        "folds": run_json(run_id, "folds.json"),
        "machines": run_json(run_id, "machines.json"),
        "curves": run_json(run_id, "curves.json") or {},
        "rca_eval": run_json(run_id, "rca_eval.json"),
        "config_text": (run_path(run_id) / "config.yaml").read_text(encoding="utf-8"),
    }))


@app.get("/api/run/<run_id>/fleet")
def api_fleet(run_id):
    """Every machine on one time axis: share of minutes alarmed per bucket for
    one model, plus share of minutes labelled anomalous."""
    meta = run_json(run_id, "machines.json")
    model = request.args.get("model") or run_json(run_id, "leaderboard.json")[0]["model"]
    size = max(10, int(request.args.get("bucket", 60)))
    rows = []
    for name in sorted(meta, key=lambda n: [int(x) for x in n.split("-")[1:]]):
        sc = scores(run_id, name)
        if model not in sc:
            abort(400, f"Model {model} not in run")
        y = machine(run_id, name).labels
        alarm = (sc[model] >= meta[name]["thresholds"][model]).astype(np.float32)
        rows.append({"machine": name,
                     "alarm": bucket(alarm, size, "mean").round(3).tolist(),
                     "truth": bucket(y.astype(np.float32), size, "mean").round(3).tolist()})
    return jsonify({"model": model, "label": model_meta(model)["label"], "bucket_minutes": size, "rows": rows})


@app.get("/api/run/<run_id>/machine/<name>")
def api_machine(run_id, name):
    meta = run_json(run_id, "machines.json")
    if name not in meta:
        abort(404, f"Machine {name} is not in run {run_id}")
    m, sc = machine(run_id, name), scores(run_id, name)
    model = request.args.get("model") or run_json(run_id, "leaderboard.json")[0]["model"]
    if model not in sc:
        abort(400, f"Model {model} not in run")
    thr_all = meta[name]["thresholds"]
    thr = thr_all[model]
    s, y = sc[model], m.labels
    T = len(y)
    start = max(0, int(request.args.get("start", 0)))
    end = min(T - 1, int(request.args.get("end", T - 1)))
    size = max(1, int(np.ceil((end - start + 1) / int(CFG["dashboard"]["max_points"]))))
    sl = slice(start, end + 1)

    # metrics ranked by how far they drift from the normal baseline during labelled anomalies
    base_mean, base_std = m.train.mean(axis=0), np.maximum(m.train.std(axis=0), 0.01)
    z_abs = np.abs((m.test - base_mean) / base_std)
    z_rank = z_abs[y == 1].mean(axis=0) if (y == 1).any() else z_abs.max(axis=0)
    order = np.argsort(-z_rank)
    req_metrics = [x for x in request.args.get("metrics", "").split(",") if x in METRIC_NAMES]
    metric_idx = [METRIC_NAMES.index(x) for x in req_metrics] or order[:3].tolist()

    alarms = {k: v >= thr_all[k] for k, v in sc.items()}
    incidents = detect_incidents(s, thr, CFG["evaluation"]["event_gap"])
    ranges = [(i["start"], i["end"]) for i in incidents]
    if ranges:
        agree = np.sum([any_in_ranges(a, ranges) for a in alarms.values()], axis=0)
        truth_hit = any_in_ranges(y == 1, ranges)
        for i, inc in enumerate(incidents):
            inc["agreeing"] = int(agree[i])
            inc["truth_overlap"] = round(float(y[inc["start"]:inc["end"] + 1].mean()), 2) if truth_hit[i] else 0.0
    truth = [{"start": a, "end": b} for a, b in segments(y == 1)]
    mm = meta[name].get("metrics", {}).get(model, {})
    caught = sum(1 for t in truth if any(i["start"] <= t["end"] and i["end"] >= t["start"] for i in incidents))

    order_keys = [r["model"] for r in run_json(run_id, "leaderboard.json") if r["model"] in sc]
    return jsonify(clean({
        "machine": name, "model": model, "model_label": model_meta(model)["label"],
        "threshold": thr, "length": T, "start": start, "end": end, "bucket": size,
        "x": np.arange(start, end + 1, size).tolist(),
        "score": bucket(s[sl], size, "max").round(4).tolist(),
        "metrics": {METRIC_NAMES[i]: bucket(m.test[sl, i], size, "mean").round(4).tolist() for i in metric_idx},
        "metric_rank": [METRIC_NAMES[i] for i in order[:12]],
        "agreement": [{"model": k, "label": model_meta(k)["label"], "kind": model_meta(k)["kind"],
                       "alarm": bucket(alarms[k][sl].astype(np.uint8), size, "max").tolist()} for k in order_keys],
        "truth_segments": truth, "incidents": incidents,
        "stats": {"points": T, "anomaly_points": int(y.sum()), "true_incidents": len(truth),
                  "caught_incidents": caught,
                  "precision": mm.get("precision"), "recall": mm.get("recall"), "f1": mm.get("f1"),
                  "detected_incidents": len(incidents), "fold": meta[name]["fold"]},
    }))


def cache_path(run_id: str, name: str, model: str, start: int, end: int) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{name}_{model}_{start}_{end}")
    return run_path(run_id) / "rca_cache" / f"{safe}.json"


@app.post("/api/rca")
def api_rca():
    body = request.get_json(force=True, silent=True) or {}
    run_id, name, model = body.get("run"), body.get("machine"), body.get("model")
    try:
        start, end = int(body.get("start")), int(body.get("end"))
    except (TypeError, ValueError):
        abort(400, "start and end must be integers")
    meta = run_json(run_id, "machines.json")
    if name not in meta:
        abort(404, "Unknown machine")
    sc = scores(run_id, name)
    if model not in sc:
        abort(400, "Unknown model")

    cp = cache_path(run_id, name, model, start, end)
    if cp.exists() and not body.get("refresh"):
        cached = json.loads(cp.read_text(encoding="utf-8"))
        cached["cached"] = True
        return jsonify(cached)

    ev = build_evidence(machine(run_id, name), sc, meta[name]["thresholds"], labels_of(sc.keys()),
                        model, start, end, CFG["rca"]["top_k_metrics"], CFG["evaluation"]["event_gap"])
    result = analyse(CFG, ev)
    truth = ev.pop("_ground_truth")
    payload = clean({"evidence": ev, "ground_truth": truth, **result,
                     "created_at": datetime.now().isoformat(timespec="seconds"), "cached": False})
    if result["source"] == "gemini":   # only cache real LLM answers; offline is instant anyway
        cp.parent.mkdir(exist_ok=True)
        cp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return jsonify(payload)


@app.errorhandler(400)
@app.errorhandler(404)
def json_error(err):
    return jsonify({"error": getattr(err, "description", str(err))}), err.code


# ------------------------------------------------------------------ static
@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:path>")
def static_files(path):
    return send_from_directory(STATIC, path)


def main():
    ap = argparse.ArgumentParser(description="OS Anomaly Analytics dashboard")
    ap.add_argument("--host", default=CFG["dashboard"]["host"])
    ap.add_argument("--port", type=int, default=CFG["dashboard"]["port"])
    args = ap.parse_args()
    if not latest_run():
        print("No runs found. Run: python -m src.pipeline all")
    print(f"Dashboard on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
