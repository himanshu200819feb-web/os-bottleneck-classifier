"""Root-cause analysis (RCA) for a detected incident.

1. build_evidence(): numbers only, no LLM.
   * top metrics: which metrics deviate most from the machine's normal baseline
   * onset order: which metric crossed 3 standard deviations first (causes
     usually move before their effects)
   * co-moving groups: top metrics that are strongly correlated during the incident
   * shape: spike / level shift / gradual drift / intermittent (simple heuristics)
   * model agreement: which detectors raised an alarm in the window
   * recurrence: earlier or later incidents on the same machine with a similar
     metric fingerprint
   Ground-truth interpretation labels are attached for display only and are
   never sent to the LLM.
2. analyse(): sends the evidence to a chain of Gemini models and returns a
   structured, Pydantic-validated report. If every model fails, or there is no
   key, it returns a rule-based report clearly marked as offline.
3. localisation_eval(): measures how well the metric ranking matches SMD's
   interpretation labels (HitRate@100% / @150%, as in the OmniAnomaly paper).
"""
from __future__ import annotations

import json
import os
import time
from typing import List

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from .data import Machine, METRIC_NAMES, segments

SEVERITIES = ["Low", "Medium", "High", "Critical"]
ONSET_Z = 3.0
RANK_METHODS = ("mean_abs_z", "abs_mean_z", "max_abs_z")
# Peak deviation ranks SMD's labelled culprit metrics best (HitRate@100% 0.73 vs
# 0.61 for mean |z| on the 8-machine profile); all three are reported in rca_eval.json.
DEFAULT_RANK = "max_abs_z"


class RCAReport(BaseModel):
    severity: str = Field(description="One of Low, Medium, High, Critical")
    headline: str = Field(description="One-sentence summary of what happened")
    summary: str = Field(description="2-3 sentences an on-call engineer can read in 10 seconds")
    likely_cause: str = Field(description="Most plausible root cause given the evidence")
    pattern: str = Field(description="Shape of the anomaly, e.g. sudden spike, level shift, gradual drift")
    timeline: List[str] = Field(description="2-5 short, ordered steps of how the incident unfolded, citing minutes")
    evidence: List[str] = Field(description="3-5 short bullet points citing the numbers")
    investigation_steps: List[str] = Field(description="3-5 read-only checks an operator should run next")
    confidence: str = Field(description="Low, Medium or High")
    caveats: str = Field(description="What the evidence cannot tell us")


# ------------------------------------------------------------------ helpers
def baseline_stats(train: np.ndarray):
    return train.mean(axis=0), np.maximum(train.std(axis=0), 0.01)


def rank_metrics(z_window: np.ndarray, method: str = DEFAULT_RANK) -> np.ndarray:
    """z_window: (minutes, 38) z-scores during the incident. Returns metric
    indices, most deviating first."""
    if method == "mean_abs_z":
        s = np.abs(z_window).mean(axis=0)
    elif method == "abs_mean_z":
        s = np.abs(z_window.mean(axis=0))
    elif method == "max_abs_z":
        s = np.abs(z_window).max(axis=0)
    else:
        raise ValueError(f"Unknown rank method {method}")
    return np.argsort(-s, kind="stable")


def classify_shape(z: np.ndarray) -> str:
    """z: |z| trajectory of the lead metric over the incident."""
    n = len(z)
    if n <= 5:
        return "short spike"
    above = z > ONSET_Z
    if above.mean() < 0.5:
        return "intermittent bursts"
    half = n // 2
    first, second = z[:half].mean(), z[half:].mean()
    onset = int(np.argmax(above)) if above.any() else n
    if onset <= max(3, n // 10) and second >= 0.7 * first:
        return "sudden level shift that persists"
    if second > 1.5 * max(first, 1e-6):
        return "gradual drift that keeps growing"
    if second < 0.5 * first:
        return "sharp jump that fades"
    return "sustained elevation"


def co_moving_groups(window: np.ndarray, idx: list[int], thr: float = 0.7) -> list[list[str]]:
    """Connected components of |corr| >= thr among the given metrics."""
    if len(window) < 5 or len(idx) < 2:
        return []
    sub = window[:, idx]
    sd = sub.std(axis=0)
    valid = sd > 1e-9
    if valid.sum() < 2:
        return []
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.nan_to_num(np.corrcoef(sub[:, valid].T))
    names = [METRIC_NAMES[i] for i, v in zip(idx, valid) if v]
    seen, groups = set(), []
    for a in range(len(names)):
        if a in seen:
            continue
        stack, comp = [a], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u); comp.append(names[u])
            stack.extend(v for v in range(len(names)) if v not in seen and abs(c[u, v]) >= thr)
        if len(comp) > 1:
            groups.append(sorted(comp))
    return groups


def detect_incidents(scores: np.ndarray, threshold: float, gap: int, min_len: int = 1) -> list[dict]:
    out = []
    for a, b in segments(scores >= threshold, gap=gap):
        if b - a + 1 >= min_len:
            out.append({"start": a, "end": b, "duration": b - a + 1,
                        "peak": round(float(scores[a:b + 1].max() / threshold), 2) if threshold > 0 else None})
    return out


# ------------------------------------------------------------------ evidence
def build_evidence(machine: Machine, scores: dict, thresholds: dict, labels_of: dict,
                   model_key: str, start: int, end: int, top_k: int = 6, gap: int = 5,
                   rank_by: str = DEFAULT_RANK) -> dict:
    test = machine.test
    start, end = max(0, int(start)), min(len(test) - 1, int(end))
    mean, std = baseline_stats(machine.train)
    window = test[start:end + 1]
    z_win = (window - mean) / std
    pre_a = max(0, start - 30)
    z_pre = (test[pre_a:start] - mean) / std
    order = rank_metrics(z_win, rank_by)[:top_k]
    inc_mean = window.mean(axis=0)
    zm = z_win.mean(axis=0)

    # onset: first minute (relative to incident start, can be negative) a metric passes |z| > 3
    z_ext = np.vstack([z_pre, z_win]) if len(z_pre) else z_win
    offset = start - pre_a

    top = []
    for i in order:
        hit = np.flatnonzero(np.abs(z_ext[:, i]) > ONSET_Z)
        top.append({
            "metric": METRIC_NAMES[i],
            "z_score": round(float(zm[i]), 2),
            "peak_abs_z": round(float(np.abs(z_win[:, i]).max()), 2),
            "direction": "up" if zm[i] > 0 else "down",
            "baseline_mean": round(float(mean[i]), 4),
            "incident_mean": round(float(inc_mean[i]), 4),
            "incident_peak": round(float(window[:, i].max()), 4),
            "before_mean_30min": round(float(test[pre_a:start, i].mean()), 4) if start > pre_a else None,
            "onset_minute": int(hit[0] - offset) if hit.size else None,
        })
    first_movers = sorted((t for t in top if t["onset_minute"] is not None), key=lambda t: t["onset_minute"])

    # model agreement
    agreement = []
    for k, s in scores.items():
        frac = float((s[start:end + 1] >= thresholds[k]).mean())
        if frac > 0:
            agreement.append({"model": labels_of[k], "share_of_window_alarmed": round(frac, 2)})
    agreement.sort(key=lambda a: -a["share_of_window_alarmed"])

    # recurrence: other incidents (same detector) with a similar z fingerprint
    fp = zm / max(np.linalg.norm(zm), 1e-9)
    similar = []
    for inc in detect_incidents(scores[model_key], thresholds[model_key], gap):
        if inc["end"] >= start and inc["start"] <= end:
            continue
        v = ((test[inc["start"]:inc["end"] + 1] - mean) / std).mean(axis=0)
        sim = float(fp @ (v / max(np.linalg.norm(v), 1e-9)))
        if sim >= 0.8:
            similar.append({"start_minute": inc["start"], "duration_minutes": inc["duration"],
                            "similarity": round(sim, 2), "when": "before" if inc["start"] < start else "after"})
    similar.sort(key=lambda s: -s["similarity"])

    lead = order[0]
    s_sel = scores[model_key][start:end + 1]
    thr = thresholds[model_key]

    truth_dims, truth_overlap = [], float(machine.labels[start:end + 1].mean())
    for a, b, dims in machine.interpretation:
        if a <= end and b >= start:
            truth_dims.extend(METRIC_NAMES[d] for d in dims if d < len(METRIC_NAMES))

    return {
        "machine": machine.name,
        "start_minute": start, "end_minute": end, "duration_minutes": end - start + 1,
        "detector": labels_of[model_key],
        "peak_score_vs_threshold": round(float(s_sel.max() / thr), 2) if thr > 0 else None,
        "models_agreeing": [a["model"] for a in agreement],
        "agreement": f"{len(agreement)}/{len(scores)}",
        "agreement_detail": agreement,
        "metrics_deviating_more_than_3sd": int((np.abs(zm) > 3).sum()),
        "top_metrics": top,
        "first_movers": [{"metric": t["metric"], "onset_minute": t["onset_minute"]} for t in first_movers[:4]],
        "co_moving_groups": co_moving_groups(window, order.tolist()),
        "shape": classify_shape(np.abs(z_win[:, lead])),
        "similar_incidents": similar[:3],
        # display only, never sent to the LLM:
        "_ground_truth": {"labelled_anomaly_share": round(truth_overlap, 3),
                          "interpretation_metrics": sorted(set(truth_dims))},
    }


def llm_view(evidence: dict) -> dict:
    return {k: v for k, v in evidence.items() if not k.startswith("_")}


# ------------------------------------------------------------------ offline report
def offline_report(ev: dict) -> RCAReport:
    n_models = int(ev["agreement"].split("/")[1]) or 1
    agree = len(ev["models_agreeing"]) / n_models
    max_z = max((abs(t["z_score"]) for t in ev["top_metrics"]), default=0)
    level = (agree >= 0.6) + (max_z >= 6) + (ev["duration_minutes"] >= 30) + (ev["metrics_deviating_more_than_3sd"] >= 8)
    severity = SEVERITIES[min(level, 3)]
    top = ev["top_metrics"][:3]
    names = ", ".join(t["metric"] for t in top)
    broad = ev["metrics_deviating_more_than_3sd"] >= 8
    first = ev["first_movers"][0] if ev["first_movers"] else None
    timeline = []
    if first:
        timeline.append(f"{first['metric']} is the first to pass 3 SD (minute {first['onset_minute']:+d} from the alarm).")
    for g in ev["co_moving_groups"][:2]:
        timeline.append(f"{', '.join(g)} move together during the incident.")
    timeline.append(f"The incident lasts {ev['duration_minutes']} minutes ({ev['shape']}).")
    return RCAReport(
        severity=severity,
        headline=f"{ev['duration_minutes']}-minute {ev['shape']} on {ev['machine']}, led by {names}.",
        summary=(f"{ev['agreement']} detectors flagged this window. {ev['metrics_deviating_more_than_3sd']} metrics "
                 f"are more than 3 SD from this machine's normal level"
                 + (f"; {first['metric']} moved first." if first else ".")),
        likely_cause=("A system-wide event (e.g. load surge, failover or resource exhaustion) "
                      "is the most likely explanation." if broad else
                      f"A component-level issue around {first['metric'] if first else top[0]['metric']}; "
                      "check the process or subsystem that owns these metrics."),
        pattern=ev["shape"] + (", broad and correlated" if broad else ", limited to a few metrics"),
        timeline=timeline,
        evidence=[f"{t['metric']}: {t['z_score']:+.1f} SD vs baseline "
                  f"({t['baseline_mean']} → {t['incident_mean']})" for t in top]
                 + [f"{ev['agreement']} models raised an alarm in this window"]
                 + ([f"{len(ev['similar_incidents'])} similar incident(s) on this machine"] if ev["similar_incidents"] else []),
        investigation_steps=["Check top consumers: `top -o %CPU` / `ps aux --sort=-%mem | head`",
                             "Look for pressure: `vmstat 1 10` and `iostat -xz 1 5`",
                             "Review kernel and service logs for the window: `journalctl --since ... --until ...`",
                             "Compare with deploys or cron jobs scheduled at this time"],
        confidence="Medium" if agree >= 0.5 else "Low",
        caveats="Rule-based summary generated without an LLM. SMD metric names are anonymised, "
                "so the cause is inferred from the pattern, not from metric meaning.",
    )


# ------------------------------------------------------------------ Gemini
def model_chain(cfg: dict) -> list[str]:
    env = os.getenv("GEMINI_MODELS")
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]
    chain = cfg["rca"].get("models") or [cfg["rca"].get("model", "gemini-2.5-flash")]
    return list(chain)


def gemini_status(cfg: dict) -> dict:
    chain = model_chain(cfg)
    return {"configured": bool(os.getenv("GEMINI_API_KEY")), "models": chain, "model": chain[0]}


PROMPT = """You are a senior site-reliability engineer writing a root-cause analysis for an on-call colleague.

Data: the Server Machine Dataset (SMD). 38 per-minute metrics from one production
server, scaled to [0, 1]. Metric names are anonymised (m01..m38), so do NOT invent
what a metric measures; reason from the pattern.

How to read the evidence:
- z_score: how many standard deviations the incident mean is from this machine's
  normal baseline; peak_abs_z is the largest single-minute deviation.
- onset_minute / first_movers: minute (relative to the alarm start) at which a metric
  first passed 3 SD. Negative = it moved before the alarm. Earlier movers are better
  root-cause candidates than later ones.
- co_moving_groups: metrics strongly correlated during the incident (likely the same
  subsystem or a cause and its effect).
- shape: heuristic shape of the lead metric (spike, level shift, drift, ...).
- agreement_detail: which detectors alarmed and for what share of the window.
- similar_incidents: other alarms on this machine with a similar metric fingerprint
  (a recurring problem is worth calling out).

Rules:
- Cite the numbers you use. Keep every bullet short.
- investigation_steps must be read-only diagnostics (e.g. top, vmstat, iostat,
  ss, dmesg, journalctl). Never suggest killing processes or changing settings.
- severity is one of Low, Medium, High, Critical; confidence is Low, Medium or High.

Incident evidence (JSON):
{evidence}
"""


def _call_gemini(client, types, model: str, cfg: dict, evidence: dict) -> RCAReport:
    resp = client.models.generate_content(
        model=model,
        contents=PROMPT.format(evidence=json.dumps(llm_view(evidence), indent=2)),
        config=types.GenerateContentConfig(
            temperature=cfg["rca"].get("temperature", 0.2),
            response_mime_type="application/json",
            response_schema=RCAReport,
        ),
    )
    report = resp.parsed if isinstance(resp.parsed, RCAReport) else RCAReport.model_validate_json(resp.text)
    if report.severity not in SEVERITIES:
        report.severity = "Medium"
    return report


def analyse(cfg: dict, evidence: dict, caller=None) -> dict:
    """Try each Gemini model in the chain; first valid report wins. If all fail
    (or no key), fall back to the offline report.

    Returns {"source": "gemini"|"offline", "model", "error", "attempts", "report"}
    where attempts = [{"model", "ok", "seconds", "error"}] in the order tried.
    `caller(model) -> RCAReport` can replace the real API call (used in tests)."""
    chain = model_chain(cfg)
    if not os.getenv("GEMINI_API_KEY") and caller is None:
        return {"source": "offline", "model": None, "attempts": [],
                "error": "GEMINI_API_KEY is not set (add it to .env).",
                "report": offline_report(evidence).model_dump()}

    if caller is None:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                              http_options=types.HttpOptions(timeout=int(cfg["rca"].get("timeout_seconds", 60) * 1000)))
        caller = lambda model: _call_gemini(client, types, model, cfg, evidence)  # noqa: E731

    attempts = []
    for model in chain:
        t0 = time.perf_counter()
        try:
            report = caller(model)
            attempts.append({"model": model, "ok": True, "seconds": round(time.perf_counter() - t0, 1), "error": None})
            return {"source": "gemini", "model": model, "error": None, "attempts": attempts,
                    "report": report.model_dump()}
        except (ValidationError, Exception) as exc:  # network, auth, quota, unknown model, bad JSON
            attempts.append({"model": model, "ok": False, "seconds": round(time.perf_counter() - t0, 1),
                             "error": f"{type(exc).__name__}: {str(exc)[:200]}"})

    return {"source": "offline", "model": None, "attempts": attempts,
            "error": f"All {len(chain)} Gemini models failed.",
            "report": offline_report(evidence).model_dump()}


# ------------------------------------------------------------------ RCA evaluation
def hit_rate(ranked: np.ndarray, truth: set, p: float) -> float:
    k = int(np.floor(p * len(truth)))
    return len(set(ranked[:k].tolist()) & truth) / len(truth)


def localisation_eval(machines: dict[str, Machine], top_k: int = 6) -> dict:
    """For every labelled incident that has interpretation labels, rank the 38
    metrics and check how many of the labelled culprit metrics are near the top.

    HitRate@P% = share of the |G| labelled metrics found in the top floor(P% x |G|).
    A random ranking scores about |G|/38 at P=100%, which is reported for comparison."""
    res = {m: {"hr100": [], "hr150": [], "top_k": []} for m in RANK_METHODS}
    rand100, n_seg, per_machine = [], 0, {}
    for name, mach in machines.items():
        mean, std = baseline_stats(mach.train)
        rows = []
        for a, b, dims in mach.interpretation:
            truth = {d for d in dims if d < len(METRIC_NAMES)}
            if not truth or b < a:
                continue
            z = (mach.test[a:b + 1] - mean) / std
            for method in RANK_METHODS:
                r = rank_metrics(z, method)
                res[method]["hr100"].append(hit_rate(r, truth, 1.0))
                res[method]["hr150"].append(hit_rate(r, truth, 1.5))
                res[method]["top_k"].append(len(set(r[:top_k].tolist()) & truth) / min(top_k, len(truth)))
            rows.append(res[DEFAULT_RANK]["hr100"][-1])
            rand100.append(min(1.0, len(truth) / len(METRIC_NAMES)))
            n_seg += 1
        if rows:
            per_machine[name] = {"incidents": len(rows), "hit_rate_100": round(float(np.mean(rows)), 4)}
    summary = {m: {"hit_rate_100": round(float(np.mean(v["hr100"])), 4),
                   "hit_rate_150": round(float(np.mean(v["hr150"])), 4),
                   f"top_{top_k}_precision": round(float(np.mean(v["top_k"])), 4)}
               for m, v in res.items() if v["hr100"]}
    return {"incidents": n_seg, "used_method": DEFAULT_RANK, "top_k": top_k, "methods": summary,
            "random_hit_rate_100": round(float(np.mean(rand100)), 4) if rand100 else None,
            "per_machine": per_machine}
