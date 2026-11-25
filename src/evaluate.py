"""Evaluation, split by machine.

Two ideas make this work across servers that all look different:

1. Baseline normalisation. Each machine's features are turned into z-scores
   against that machine's own unlabelled `train` half ("how far is this
   minute from this server's normal?"). The train half is available in a
   real deployment, so this uses no labels.

2. Each model family gets the setup that fits it:
   * unsupervised models are fitted per machine on its own normal data, and
     their scores are rescaled against a held-back slice of that normal data;
   * supervised models are fitted on the labelled halves of *other* machines.

Cross-validation (GroupKFold over machines): for every fold, the held-out
machines are never used for fitting or for choosing alarm thresholds.
Thresholds are tuned for best F1 on calibration machines from the training side.
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score)

from .data import Machine, segments
from .features import features_from_config
from .models import make_detector, REGISTRY

METRIC_KEYS = ("precision", "recall", "f1", "pa_f1", "pr_auc", "roc_auc", "event_recall",
               "detection_delay_min", "false_alarms_per_day", "fit_seconds", "latency_ms_per_1k")
Z_CLIP = 20.0
SD_FLOOR = 0.01   # SMD values are in [0, 1]; avoids huge z for near-constant metrics


# ------------------------------------------------------------------ metrics
def best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold on `scores` that gives the highest F1 on (y, scores)."""
    if y.sum() == 0:
        return float(np.quantile(scores, 0.99))
    prec, rec, thr = precision_recall_curve(y, scores)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[int(np.argmax(f1[:-1]))])


def paper_f1(y: np.ndarray, scores: np.ndarray) -> float | None:
    """F1 at the best threshold for this one server, chosen using its test labels.
    This is the protocol used in published SMD papers (e.g. Kim et al. 2022,
    Sarfraz et al. 2024). It is optimistic, since no real system can pick its
    threshold from test labels, so it is reported only for comparison."""
    if y.sum() == 0:
        return None
    p, r, _ = precision_recall_curve(y, scores)
    return float(np.max(2 * p * r / np.clip(p + r, 1e-12, None)))


def add_paper_f1(summary: list[dict], machine_metrics: dict) -> None:
    """Mean and std over servers of the per-server paper F1, added to each leaderboard row."""
    for row in summary:
        vals = [mm[row["model"]]["paper_f1"] for mm in machine_metrics.values()
                if mm.get(row["model"], {}).get("paper_f1") is not None]
        row["paper_f1"] = float(np.mean(vals)) if vals else float("nan")
        row["paper_f1_std"] = float(np.std(vals)) if vals else float("nan")


def f1_at(y, scores, thr) -> float:
    pred = scores >= thr
    tp = float((pred & (y == 1)).sum())
    return 2 * tp / max(float(pred.sum() + (y == 1).sum()), 1.0)


def event_recall(y: np.ndarray, pred: np.ndarray) -> float:
    """Share of true anomaly incidents with at least one alarm inside them."""
    events = segments(y == 1)
    if not events:
        return float("nan")
    return float(np.mean([pred[s:e + 1].any() for s, e in events]))


def detection_delays(y: np.ndarray, pred: np.ndarray) -> list[int]:
    """For every true incident that gets caught: minutes from its start to the first alarm inside it."""
    out = []
    for s, e in segments(y == 1):
        hit = np.flatnonzero(pred[s:e + 1])
        if hit.size:
            out.append(int(hit[0]))
    return out


def false_alarm_incidents(y: np.ndarray, pred: np.ndarray, gap: int = 5) -> int:
    """Alarm incidents (alarms merged with `gap`) that do not touch any labelled anomaly."""
    c = np.concatenate([[0], np.cumsum(y == 1)])
    return sum(1 for a, b in segments(pred, gap=gap) if c[b + 1] - c[a] == 0)


def point_adjust(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """The lenient protocol used in many SMD papers: if any minute of a true
    incident is flagged, the whole incident counts as detected."""
    adj = pred.astype(bool).copy()
    for s, e in segments(y == 1):
        if adj[s:e + 1].any():
            adj[s:e + 1] = True
    return adj


def score_metrics(y: np.ndarray, scores: np.ndarray, threshold: float, groups: list | None = None,
                  gap: int = 5) -> dict:
    """`groups` = list of per-machine lengths, so incidents are never merged
    across machine boundaries."""
    pred = (scores >= threshold).astype(np.int8)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    bounds = np.cumsum([0] + (groups or [len(y)]))
    parts = list(zip(bounds[:-1], bounds[1:]))
    adj = np.concatenate([point_adjust(y[a:b], pred[a:b]) for a, b in parts])
    ev = [event_recall(y[a:b], pred[a:b]) for a, b in parts]
    n_ev = [len(segments(y[a:b] == 1)) for a, b in parts]
    delays = [d for a, b in parts for d in detection_delays(y[a:b], pred[a:b])]
    false_alarms = sum(false_alarm_incidents(y[a:b], pred[a:b], gap) for a, b in parts)
    _, _, pa_f, _ = precision_recall_fscore_support(y, adj, average="binary", zero_division=0)
    both = len(np.unique(y)) == 2
    return {
        "precision": float(p), "recall": float(r), "f1": float(f), "pa_f1": float(pa_f),
        "pr_auc": float(average_precision_score(y, scores)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y, scores)) if both else float("nan"),
        "event_recall": float(np.nansum(np.array(ev) * np.array(n_ev)) / max(sum(n_ev), 1)),
        "detection_delay_min": float(np.median(delays)) if delays else float("nan"),
        "false_alarms_per_day": false_alarms / (len(y) / 1440),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def stream(seed: int, *parts) -> np.random.Generator:
    """An independent random stream per purpose (folds, calibration split, one
    model's sampling...). Adding or removing a model never changes the random
    draws of any other model, so results stay comparable across configs."""
    return np.random.default_rng([seed, *(zlib.crc32(str(p).encode()) for p in parts)])


def make_folds(machines: list[str], n_folds: int, rng: np.random.Generator) -> list[list[str]]:
    order = [str(m) for m in rng.permutation(machines)]
    n_folds = min(n_folds, len(order))
    return [sorted(order[i::n_folds]) for i in range(n_folds)]


def cap_rows(X, y, cap, rng, keep_positive_share=0.5):
    """Random cap on rows. For labelled data, anomalies are kept preferentially
    (up to `keep_positive_share` of the cap) so the minority class survives."""
    n = len(X)
    if n <= cap:
        return X, y
    if y is None:
        return X[np.sort(rng.choice(n, cap, replace=False))], None
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = min(len(pos), int(cap * keep_positive_share))
    idx = np.r_[rng.choice(pos, n_pos, replace=False), rng.choice(neg, cap - n_pos, replace=False)]
    return X[idx], y[idx]


def as_ratio(scores: np.ndarray, thr: float) -> np.ndarray:
    """Score in units of the model's own threshold (1.0 = alarm)."""
    return scores / thr if thr > 1e-9 else scores - thr + 1.0


# ------------------------------------------------------------------ features
@dataclass
class Baseline:
    mean: np.ndarray
    std: np.ndarray

    def z(self, F: np.ndarray) -> np.ndarray:
        return np.clip((F - self.mean) / self.std, -Z_CLIP, Z_CLIP).astype(np.float32)


@dataclass
class Prepared:
    train_z: np.ndarray   # unlabelled normal half, as z-scores
    test_z: np.ndarray    # labelled half, as z-scores
    y: np.ndarray


def prepare(machine: Machine, cfg: dict) -> Prepared:
    F_train = features_from_config(machine.train, cfg)
    base = Baseline(F_train.mean(axis=0), np.maximum(F_train.std(axis=0), SD_FLOOR))
    return Prepared(base.z(F_train), base.z(features_from_config(machine.test, cfg)), machine.labels)


# ------------------------------------------------------------------ model families
def per_machine_scores(cfg, key, p: Prepared, rng):
    """Unsupervised / rule models: fit on the first 80% of this machine's
    normal half, rescale scores with the last 20% (never seen in fitting):
    0 = a typical normal minute, 1 = this machine's 99th-percentile normal minute."""
    ev = cfg["evaluation"]
    cut = int(len(p.train_z) * 0.8)
    fit_X = p.train_z[:cut][:: max(1, ev["train_stride"] // 2)]
    fit_X, _ = cap_rows(fit_X, None, cfg["models"][key].get("max_fit_rows", ev["max_fit_rows"]), rng)
    det = make_detector(cfg, key)
    t0 = time.perf_counter(); det.fit(fit_X); fit_s = time.perf_counter() - t0
    ref = det.score(p.train_z[cut:])
    q50, q99 = np.quantile(ref, [0.5, 0.99])
    t0 = time.perf_counter(); raw = det.score(p.test_z); score_s = time.perf_counter() - t0
    return (raw - q50) / max(float(q99 - q50), 1e-6), fit_s, score_s


def supervised_training_rows(prep: dict, machines: list[str], stride: int):
    Xs, ys = [], []
    for m in machines:
        y = prep[m].y
        keep = (y == 1) | (np.arange(len(y)) % stride == 0)
        Xs.append(prep[m].test_z[keep]); ys.append(y[keep])
    return np.vstack(Xs), np.concatenate(ys)


def strided(arrays: dict, machines: list[str], stride: int) -> np.ndarray:
    return np.concatenate([arrays[m][::stride] for m in machines])


# ------------------------------------------------------------------ CV loop
def cross_validate(cfg: dict, machines: dict[str, Machine], model_keys: list[str], log=print) -> dict:
    ev = cfg["evaluation"]
    stride = ev["train_stride"]
    seed = cfg["seed"]
    names = sorted(machines)

    log(f"Preparing baseline-normalised features for {len(names)} machines...")
    prep = {m: prepare(machines[m], cfg) for m in names}
    folds = make_folds(names, ev["n_folds"], stream(seed, "folds"))
    fold_of = {m: fi for fi, ms in enumerate(folds) for m in ms}
    ys = {m: prep[m].y for m in names}

    # Per-machine models see only each machine's own unlabelled half -> score once.
    local_keys = [k for k in model_keys if REGISTRY[k].kind != "supervised"]
    local_scores, local_time = {k: {} for k in local_keys}, {k: [0.0, 0.0, 0] for k in local_keys}
    for key in local_keys:
        log(f"Fitting {REGISTRY[key].label} on each machine's own normal data...")
        for m in names:
            s, fit_s, score_s = per_machine_scores(cfg, key, prep[m], stream(seed, "local", key, m))
            local_scores[key][m] = s.astype(np.float32)
            local_time[key][0] += fit_s; local_time[key][1] += score_s; local_time[key][2] += len(s)

    per_fold = {k: [] for k in model_keys}
    oof_scores = {m: {} for m in names}
    thresholds = {m: {} for m in names}

    for fi, test_ms in enumerate(folds):
        rest = [m for m in names if m not in test_ms]
        n_cal = max(1, int(round(len(rest) * ev["calibration_fraction"])))
        cal_ms = sorted(str(m) for m in stream(seed, "calibration", fi).choice(rest, n_cal, replace=False))
        fit_ms = [m for m in rest if m not in cal_ms]
        log(f"Fold {fi + 1}/{len(folds)}: test={test_ms}")
        test_y = np.concatenate([ys[m] for m in test_ms])
        lengths = [len(ys[m]) for m in test_ms]
        cal_y = strided(ys, cal_ms, stride)
        cal_X = np.vstack([prep[m].test_z[::stride] for m in cal_ms])

        for key in model_keys:
            spec = REGISTRY[key]
            cap = cfg["models"][key].get("max_fit_rows", ev["max_fit_rows"])
            if spec.kind == "supervised" and ev.get("calibration", "holdout") == "crossfit":
                # Cross-fitting: every training machine gets an out-of-sample score from a
                # model that did not see it, so the threshold is tuned on all of them; the
                # final model is then trained on all training machines.
                parts = [sorted(str(m) for m in p) for p in np.array_split(stream(seed, "crossfit", fi).permutation(rest), ev.get("crossfit_parts", 3))]
                c_s, c_y = [], []
                for pi, part in enumerate(parts):
                    X, y = supervised_training_rows(prep, [m for m in rest if m not in part], stride)
                    X, y = cap_rows(X, y, cap, stream(seed, "fit", key, fi, pi))
                    d = make_detector(cfg, key).fit(X, y)
                    c_s.append(np.concatenate([d.score(prep[m].test_z[::stride]) for m in part]))
                    c_y.append(strided(ys, part, stride))
                cal_s, c_y = np.concatenate(c_s), np.concatenate(c_y)
                thr = best_f1_threshold(c_y, cal_s)
                X, y = supervised_training_rows(prep, rest, stride)
                X, y = cap_rows(X, y, cap, stream(seed, "fit", key, fi))
                det = make_detector(cfg, key)
                t0 = time.perf_counter(); det.fit(X, y); fit_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                scores = {m: det.score(prep[m].test_z) for m in test_ms}
                score_s = time.perf_counter() - t0
                n_fit = len(X)
            elif spec.kind == "supervised":
                # Hold-out: fit on some training machines, tune the threshold on the others.
                X, y = supervised_training_rows(prep, fit_ms, stride)
                X, y = cap_rows(X, y, cap, stream(seed, "fit", key, fi))
                det = make_detector(cfg, key)
                t0 = time.perf_counter(); det.fit(X, y); fit_s = time.perf_counter() - t0
                cal_s, c_y = det.score(cal_X), cal_y
                thr = best_f1_threshold(c_y, cal_s)
                t0 = time.perf_counter()
                scores = {m: det.score(prep[m].test_z) for m in test_ms}
                score_s = time.perf_counter() - t0
                n_fit = len(X)
            else:
                # No labels were used to fit these, so every non-test machine can calibrate.
                c_y, cal_s = strided(ys, rest, stride), strided(local_scores[key], rest, stride)
                thr = best_f1_threshold(c_y, cal_s)
                scores = {m: local_scores[key][m] for m in test_ms}
                tf, ts, tn = local_time[key]
                fit_s = tf / len(names)                     # average fit time per machine
                score_s = ts / tn * sum(len(s) for s in scores.values())
                n_fit = None

            all_scores = np.concatenate([scores[m] for m in test_ms])
            res = score_metrics(test_y, all_scores, thr, lengths, ev.get("event_gap", 5))
            res.update(fold=fi, threshold=thr, fit_seconds=fit_s, calibration_f1=f1_at(c_y, cal_s, thr),
                       latency_ms_per_1k=1e6 * score_s / len(all_scores), n_fit_rows=n_fit)
            per_fold[key].append(res)
            for m in test_ms:
                oof_scores[m][key] = scores[m].astype(np.float32)
                thresholds[m][key] = thr
            log(f"   {spec.label:<24} F1={res['f1']:.3f}  PR-AUC={res['pr_auc']:.3f}  PA-F1={res['pa_f1']:.3f}")

    machine_metrics = per_machine_metrics(ys, oof_scores, thresholds, ev.get("event_gap", 5))
    summary = summarise(per_fold)
    add_paper_f1(summary, machine_metrics)
    return {"folds": folds, "fold_of": fold_of, "per_fold": per_fold,
            "oof_scores": oof_scores, "thresholds": thresholds,
            "summary": summary,
            "machine_metrics": machine_metrics,
            "curves": pr_curves(ys, oof_scores, thresholds)}


# ------------------------------------------------------------------ summaries
def model_meta(key: str) -> dict:
    s = REGISTRY[key]
    return {"label": s.label, "kind": s.kind, "family": s.family, "neural": s.neural}


def summarise(per_fold: dict) -> list[dict]:
    rows = []
    for key, folds in per_fold.items():
        row = {"model": key, **model_meta(key)}
        for mk in METRIC_KEYS:
            vals = np.array([f[mk] for f in folds], dtype=float)
            row[mk] = float(np.nanmean(vals))
            row[mk + "_std"] = float(np.nanstd(vals))
        row["confusion"] = np.sum([f["confusion"] for f in folds], axis=0).tolist()
        rows.append(row)
    rows.sort(key=lambda r: r["f1"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def per_machine_metrics(ys: dict, scores: dict, thresholds: dict, gap: int) -> dict:
    out = {}
    for m, y in ys.items():
        true_inc = segments(y == 1)
        out[m] = {"true_incidents": len(true_inc)}
        for k, s in scores[m].items():
            pred = s >= thresholds[m][k]
            p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
            delays = detection_delays(y, pred)
            out[m][k] = {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4),
                         "event_recall": round(event_recall(y, pred), 4) if true_inc else None,
                         "alarm_incidents": len(segments(pred, gap=gap)),
                         "detection_delay_min": float(np.median(delays)) if delays else None,
                         "false_alarms_per_day": round(false_alarm_incidents(y, pred, gap) / (len(y) / 1440), 2),
                         "paper_f1": paper_f1(y, s)}
    return out


def pr_curves(ys: dict, scores: dict, thresholds: dict, n_points: int = 80) -> dict:
    """Pooled precision-recall curve per model, using scores in units of each
    machine's threshold (so 1.0 is the operating point everywhere)."""
    names = sorted(ys)
    y = np.concatenate([ys[m] for m in names])
    out = {}
    for k in scores[names[0]]:
        s = np.concatenate([as_ratio(scores[m][k], thresholds[m][k]) for m in names])
        prec, rec, _ = precision_recall_curve(y, s)
        grid = np.linspace(0, 1, n_points + 1)
        # best precision achievable at each recall level (upper envelope, standard for plotting)
        order = np.argsort(rec)
        rec_s, prec_s = rec[order], prec[order]
        env = np.maximum.accumulate(prec_s[::-1])[::-1]
        p_grid = np.interp(grid, rec_s, env)
        pred = s >= 1.0
        tp = float((pred & (y == 1)).sum())
        out[k] = {"recall": grid.round(4).tolist(), "precision": p_grid.round(4).tolist(),
                  "operating_point": {"recall": tp / max(float(y.sum()), 1), "precision": tp / max(float(pred.sum()), 1)}}
    return out
