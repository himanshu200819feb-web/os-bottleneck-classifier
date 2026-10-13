"""Modern Streamlit dashboard for live and recorded OS bottleneck classification."""

from __future__ import annotations

import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st
import yaml

st.set_page_config(
    page_title="OS Bottleneck Monitor",
    page_icon="◫",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    CONFIG = yaml.safe_load(handle) or {}

ARTIFACTS_DIR = PROJECT_ROOT / CONFIG.get("paths", {}).get("artifacts_dir", "artifacts")
DASHBOARD_CFG = CONFIG.get("dashboard", {})
TELEMETRY_CFG = CONFIG.get("telemetry", {})
HEURISTIC_CFG = CONFIG.get("models", {}).get("baseline_heuristic", {}).get("thresholds", {})
HISTORY_LIMIT = max(int(DASHBOARD_CFG.get("telemetry_history_window", 60)), 30)
DEFAULT_REFRESH = float(DASHBOARD_CFG.get("refresh_rate_sec", 1.0))


from src.llm_agent import diagnose_bottleneck, gemini_status

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
:root { --accent:#0f766e; --accent-soft:rgba(15,118,110,.10); --line:rgba(120,120,120,.18); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
[data-testid="stAppViewContainer"] { background:color-mix(in srgb, var(--background-color) 97%, #0f766e 3%); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { border-right:1px solid var(--line); }
.block-container { max-width:1440px; padding-top:2rem; padding-bottom:3rem; }
h1 { font-size:clamp(1.7rem,2vw,2.2rem) !important; letter-spacing:-.035em; }
h2, h3 { letter-spacing:-.02em; }
div[data-testid="stMetric"] { background:var(--secondary-background-color); border:1px solid var(--line); border-radius:12px; padding:1rem; }
div[data-testid="stMetricValue"] { font-family:'JetBrains Mono',monospace; font-size:1.55rem; }
.stButton > button { min-height:44px; border-radius:9px; font-weight:600; transition:transform .15s ease, box-shadow .15s ease; }
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 6px 18px rgba(15,118,110,.12); border-color:var(--accent); }
.stButton > button:active { transform:translateY(0); }
[data-testid="stStatusWidget"], [data-testid="stAlert"] { border-radius:10px; }
.live-row { display:flex; align-items:center; gap:.55rem; color:var(--text-color); font-size:.9rem; }
.live-dot { width:8px; height:8px; border-radius:50%; background:#16a34a; box-shadow:0 0 0 4px rgba(22,163,74,.12); }
.paused-dot { width:8px; height:8px; border-radius:50%; background:#94a3b8; }
.status-pill { display:inline-block; padding:.28rem .65rem; border-radius:999px; background:var(--accent-soft); color:#0f766e; font-weight:700; font-size:.78rem; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
code { font-family:'JetBrains Mono',monospace !important; }
@media (max-width: 700px) { .block-container { padding:1rem; } div[data-testid="stMetricValue"] { font-size:1.25rem; } }
</style>
""",
    unsafe_allow_html=True,
)


CANONICAL_CLASSES = ["CPU_Bound", "IO_Contention", "Memory_Pressure", "Normal"]
UI_NUMERIC_COLUMNS = [
    "cpu_user_pct", "cpu_system_pct", "cpu_iowait_pct", "cpu_idle_pct",
    "load_avg_1m_per_cpu", "context_switches_per_sec", "interrupts_per_sec",
    "mem_used_pct", "mem_available_mb", "swap_used_pct", "swap_in_kb_per_sec",
    "swap_out_kb_per_sec", "disk_read_kb_per_sec", "disk_write_kb_per_sec",
    "disk_busy_pct", "net_rx_kb_per_sec", "net_tx_kb_per_sec",
]


if nn is not None:
    class ConvNet1D(nn.Module):
        def __init__(self, input_dim: int, num_classes: int):
            super().__init__()
            pooled = min(4, input_dim)
            self.features = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(pooled),
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * pooled, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, num_classes)
            )

        def forward(self, values):
            return self.classifier(self.features(values).flatten(1))


def normalise_class_name(value: Any) -> str:
    aliases = {
        "memorythrashing": "Memory_Pressure", "memorypressure": "Memory_Pressure",
        "cpubound": "CPU_Bound", "iocontention": "IO_Contention", "normal": "Normal",
    }
    text = str(value)
    return aliases.get(text.replace("_", "").replace("-", "").lower(), text)


def decode_label(value: Any, encoder: Any = None) -> str:
    if encoder is not None and isinstance(value, (int, np.integer, float, np.floating)):
        try:
            return normalise_class_name(encoder.inverse_transform([int(value)])[0])
        except (ValueError, TypeError, IndexError):
            pass
    return normalise_class_name(value)


@st.cache_resource
def load_system_artifacts() -> Dict[str, Any]:
    assets: Dict[str, Any] = {"models": {}, "ready": False, "errors": []}
    if not ARTIFACTS_DIR.is_dir():
        assets["errors"].append(f"Artifact directory not found: {ARTIFACTS_DIR}")
        return assets

    path_cfg = CONFIG.get("paths", {})
    optional_files = {
        "le": PROJECT_ROOT / path_cfg.get("label_encoder_path", "artifacts/label_encoder.joblib"),
        "features": PROJECT_ROOT / path_cfg.get("feature_names_path", "artifacts/feature_names.joblib"),
        "cnn_preprocessor": PROJECT_ROOT / path_cfg.get("scaler_path", "artifacts/cnn_preprocessor.joblib"),
    }
    for key, path in optional_files.items():
        if path.is_file():
            try:
                assets[key] = joblib.load(path)
            except Exception as exc:
                assets["errors"].append(f"Could not load {path.name}: {exc}")

    for path in ARTIFACTS_DIR.glob("*.joblib"):
        if path.name in {item.name for item in optional_files.values()}:
            continue
        try:
            candidate = joblib.load(path)
            if hasattr(candidate, "predict"):
                assets["models"][path.stem] = ("sklearn", candidate)
        except Exception:
            continue

    if "features" not in assets:
        for kind, model in assets["models"].values():
            inferred = getattr(model, "feature_names_in_", None) if kind == "sklearn" else None
            if inferred is not None:
                assets["features"] = list(inferred)
                break

    cnn_path = ARTIFACTS_DIR / "CNN_1D.pt"
    if cnn_path.is_file():
        if torch is None:
            assets["errors"].append("CNN_1D.pt exists, but PyTorch is not installed.")
        elif not all(key in assets for key in ("features", "le", "cnn_preprocessor")):
            assets["errors"].append("CNN requires its preprocessor, feature names, and label encoder.")
        else:
            try:
                checkpoint = torch.load(cnn_path, map_location="cpu", weights_only=True)
                input_dim = int(checkpoint.get("input_dim", len(assets["features"])))
                num_classes = int(checkpoint.get("num_classes", len(assets["le"].classes_)))
                cnn = ConvNet1D(input_dim, num_classes)
                cnn.load_state_dict(checkpoint["state_dict"])
                cnn.eval()
                assets["models"]["1D_CNN_Net"] = ("pytorch", cnn)
            except Exception as exc:
                assets["errors"].append(f"Could not load CNN_1D.pt: {exc}")

    assets["ready"] = bool(assets["models"] and "features" in assets)
    if not assets["models"]:
        assets["errors"].append("No trained models found; rule-based inference is available.")
    return assets


@st.cache_data(show_spinner=False)
def read_csv_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def find_project_datasets() -> Dict[str, Path]:
    candidates = [
        PROJECT_ROOT / "data" / "processed" / "test.csv",
        PROJECT_ROOT / "data" / "processed" / "train.csv",
        PROJECT_ROOT / "data" / "raw" / "telemetry_raw.csv",
        PROJECT_ROOT / "telemetry_raw.csv",
    ]
    return {str(path.relative_to(PROJECT_ROOT)): path for path in candidates if path.is_file()}


def normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("The selected data is empty.")
    data = frame.copy()
    cores = max(psutil.cpu_count(logical=True) or 1, 1)
    if "load_avg_1m_per_cpu" not in data:
        data["load_avg_1m_per_cpu"] = pd.to_numeric(data.get("load_avg_1m", 0.0), errors="coerce") / cores
    for column in UI_NUMERIC_COLUMNS:
        if column not in data:
            data[column] = 0.0
        data[column] = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "timestamp" not in data:
        data["timestamp"] = np.arange(len(data)).astype(str)
    return data


def _process_snapshot() -> tuple[dict[int, tuple[float, float]], list[dict[str, Any]]]:
    counters: dict[int, tuple[float, float]] = {}
    rows: list[dict[str, Any]] = []
    attrs = ["pid", "name", "cpu_times", "memory_info", "create_time"]
    for proc in psutil.process_iter(attrs, ad_value=None):
        info = proc.info
        cpu_times = info.get("cpu_times")
        memory_info = info.get("memory_info")
        created = info.get("create_time")
        if cpu_times is None or created is None:
            continue
        total_cpu = float(getattr(cpu_times, "user", 0.0) + getattr(cpu_times, "system", 0.0))
        counters[int(info["pid"])] = (total_cpu, float(created))
        rows.append({
            "pid": int(info["pid"]),
            "process": info.get("name") or "Unknown",
            "memory_mb": float(getattr(memory_info, "rss", 0.0)) / 1024**2 if memory_info else 0.0,
        })
    return counters, rows


def initial_counter_snapshot() -> Dict[str, Any]:
    process_counters, _ = _process_snapshot()
    return {
        "time": time.monotonic(), "cpu": psutil.cpu_times(), "cpu_stats": psutil.cpu_stats(),
        "disk": psutil.disk_io_counters(), "net": psutil.net_io_counters(),
        "swap": psutil.swap_memory(), "processes": process_counters,
    }


def get_telemetry_snapshot(previous: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    elapsed = time.monotonic() - previous["time"]
    if elapsed < 0.20:
        time.sleep(0.20 - elapsed)
    now = time.monotonic()
    elapsed = max(now - previous["time"], 0.1)
    cores = max(psutil.cpu_count(logical=True) or 1, 1)

    cpu = psutil.cpu_times()
    stats = psutil.cpu_stats()
    disk = psutil.disk_io_counters()
    network = psutil.net_io_counters()
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    process_counters, process_rows = _process_snapshot()

    def delta(current: Any, old: Any, attribute: str, divisor: float = 1.0) -> float:
        if current is None or old is None:
            return 0.0
        return max(0.0, float(getattr(current, attribute, 0.0) - getattr(old, attribute, 0.0))) / divisor

    cpu_fields = set(getattr(cpu, "_fields", []))
    deltas = {field: delta(cpu, previous["cpu"], field) for field in cpu_fields}
    total_cpu_delta = max(sum(deltas.values()), 1e-9)
    pct = lambda field: 100.0 * deltas.get(field, 0.0) / total_cpu_delta
    idle_pct = pct("idle")

    try:
        load_1m = float(psutil.getloadavg()[0])
    except (AttributeError, OSError):
        load_1m = (100.0 - idle_pct) / 100.0 * cores

    disk_busy = min(100.0, delta(disk, previous["disk"], "busy_time") / elapsed / 10.0)
    sample: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host_os": platform.system(),
        "cpu_user_pct": pct("user"), "cpu_system_pct": pct("system"),
        "cpu_iowait_pct": pct("iowait"), "cpu_idle_pct": idle_pct,
        "load_avg_1m": load_1m, "load_avg_1m_per_cpu": load_1m / cores,
        "context_switches_per_sec": delta(stats, previous["cpu_stats"], "ctx_switches") / elapsed,
        "interrupts_per_sec": delta(stats, previous["cpu_stats"], "interrupts") / elapsed,
        "mem_used_pct": float(memory.percent), "mem_available_mb": float(memory.available) / 1024**2,
        "swap_used_pct": float(swap.percent),
        "swap_in_kb_per_sec": delta(swap, previous["swap"], "sin", 1024.0) / elapsed,
        "swap_out_kb_per_sec": delta(swap, previous["swap"], "sout", 1024.0) / elapsed,
        "disk_read_kb_per_sec": delta(disk, previous["disk"], "read_bytes", 1024.0) / elapsed,
        "disk_write_kb_per_sec": delta(disk, previous["disk"], "write_bytes", 1024.0) / elapsed,
        "disk_busy_pct": disk_busy,
        "net_rx_kb_per_sec": delta(network, previous["net"], "bytes_recv", 1024.0) / elapsed,
        "net_tx_kb_per_sec": delta(network, previous["net"], "bytes_sent", 1024.0) / elapsed,
    }
    sample = {key: round(value, 3) if isinstance(value, float) else value for key, value in sample.items()}

    old_processes = previous.get("processes", {})
    for row in process_rows:
        pid = row["pid"]
        current_counter = process_counters.get(pid)
        old_counter = old_processes.get(pid)
        if current_counter and old_counter and current_counter[1] == old_counter[1]:
            row["cpu_pct"] = max(0.0, (current_counter[0] - old_counter[0]) / elapsed * 100.0)
        else:
            row["cpu_pct"] = 0.0
    processes = pd.DataFrame(process_rows)
    if not processes.empty:
        processes = processes.sort_values(["cpu_pct", "memory_mb"], ascending=False).head(12).reset_index(drop=True)
        processes[["cpu_pct", "memory_mb"]] = processes[["cpu_pct", "memory_mb"]].round(1)

    next_snapshot = {
        "time": now, "cpu": cpu, "cpu_stats": stats, "disk": disk, "net": network,
        "swap": swap, "processes": process_counters,
    }
    return sample, next_snapshot, processes


def heuristic_inference(row: Dict[str, Any], classes: Iterable[str]) -> Tuple[str, list[str], list[float]]:
    cpu_limit = float(HEURISTIC_CFG.get("cpu_bound_user_pct", 75.0))
    memory_limit = float(HEURISTIC_CFG.get("memory_pressure_used_pct", 85.0))
    swap_out_limit = float(HEURISTIC_CFG.get("memory_pressure_swap_out_kb_per_sec", 1024.0))
    io_wait_limit = float(HEURISTIC_CFG.get("io_contention_iowait_pct", 20.0))
    io_write_limit = float(HEURISTIC_CFG.get("io_contention_write_kb_per_sec", 30000.0))

    if float(row.get("cpu_user_pct", 0.0)) >= cpu_limit:
        predicted = "CPU_Bound"
    elif float(row.get("mem_used_pct", 0.0)) >= memory_limit or float(row.get("swap_out_kb_per_sec", 0.0)) >= swap_out_limit:
        predicted = "Memory_Pressure"
    elif float(row.get("cpu_iowait_pct", 0.0)) >= io_wait_limit or float(row.get("disk_write_kb_per_sec", 0.0)) >= io_write_limit:
        predicted = "IO_Contention"
    else:
        predicted = "Normal"

    names = [normalise_class_name(item) for item in classes]
    distribution = {name: 0.04 for name in names}
    distribution[predicted] = 0.88
    values = [distribution.get(name, 0.04) for name in names]
    total = sum(values)
    return predicted, names, [value / total for value in values]


def prepare_model_input(row: Dict[str, Any], features: Iterable[str]) -> pd.DataFrame:
    values = {}
    for feature in features:
        try:
            numeric = float(row.get(feature, 0.0))
            values[feature] = numeric if np.isfinite(numeric) else 0.0
        except (TypeError, ValueError):
            values[feature] = 0.0
    return pd.DataFrame([values], columns=list(features))


def model_inference(row: Dict[str, Any], model_name: str, assets: Dict[str, Any]) -> Tuple[str, list[str], list[float]]:
    fallback_classes = [decode_label(item, assets.get("le")) for item in getattr(assets.get("le"), "classes_", CANONICAL_CLASSES)]
    if model_name == "Rule_Based_Heuristic" or model_name not in assets["models"]:
        return heuristic_inference(row, fallback_classes)

    model_type, model = assets["models"][model_name]
    encoder = assets.get("le")
    raw_input = prepare_model_input(row, assets["features"])
    if model_type == "pytorch":
        transformed = assets["cnn_preprocessor"].transform(raw_input)
        tensor = torch.tensor(np.asarray(transformed), dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
        encoded_classes = list(range(len(probabilities)))
        predicted_value = int(np.argmax(probabilities))
    else:
        predicted_value = model.predict(raw_input)[0]
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(raw_input)[0], dtype=float)
            encoded_classes = list(getattr(model, "classes_", range(len(probabilities))))
        else:
            encoded_classes = list(getattr(model, "classes_", getattr(encoder, "classes_", CANONICAL_CLASSES)))
            probabilities = np.zeros(len(encoded_classes), dtype=float)
            probabilities[encoded_classes.index(predicted_value)] = 1.0
    return (
        decode_label(predicted_value, encoder),
        [decode_label(value, encoder) for value in encoded_classes],
        probabilities.astype(float).tolist(),
    )


def pressure_signals(row: Dict[str, Any]) -> list[tuple[str, float, float, str]]:
    signals = [
        ("CPU user", float(row.get("cpu_user_pct", 0.0)), float(HEURISTIC_CFG.get("cpu_bound_user_pct", 75.0)), "%"),
        ("Memory used", float(row.get("mem_used_pct", 0.0)), float(HEURISTIC_CFG.get("memory_pressure_used_pct", 85.0)), "%"),
        ("I/O wait", float(row.get("cpu_iowait_pct", 0.0)), float(HEURISTIC_CFG.get("io_contention_iowait_pct", 20.0)), "%"),
        ("Swap out", float(row.get("swap_out_kb_per_sec", 0.0)), float(HEURISTIC_CFG.get("memory_pressure_swap_out_kb_per_sec", 1024.0)), "KiB/s"),
        ("Disk write", float(row.get("disk_write_kb_per_sec", 0.0)), float(HEURISTIC_CFG.get("io_contention_write_kb_per_sec", 30000.0)), "KiB/s"),
    ]
    return sorted(signals, key=lambda item: item[1] / max(item[2], 1e-9), reverse=True)


def probability_figure(classes: list[str], probabilities: list[float]) -> go.Figure:
    order = np.argsort(probabilities)
    colors = ["#0f766e" if probabilities[index] == max(probabilities) else "#94a3b8" for index in order]
    fig = go.Figure(go.Bar(
        x=[probabilities[index] for index in order], y=[classes[index].replace("_", " ") for index in order],
        orientation="h", marker_color=colors,
        text=[f"{probabilities[index]:.1%}" for index in order], textposition="auto",
    ))
    fig.update_layout(height=250, margin=dict(l=0, r=8, t=8, b=0), xaxis=dict(range=[0, 1], tickformat=".0%", showgrid=False), yaxis_title=None, showlegend=False)
    return fig


def history_figure(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    palette = {"CPU user": "#0f766e", "Memory used": "#64748b", "I/O wait": "#d97706"}
    for name, column in (("CPU user", "cpu_user_pct"), ("Memory used", "mem_used_pct"), ("I/O wait", "cpu_iowait_pct")):
        fig.add_trace(go.Scatter(x=history["timestamp"], y=history[column], name=name, mode="lines", line=dict(width=2, color=palette[name])))
    fig.update_layout(height=300, margin=dict(l=0, r=8, t=20, b=0), yaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="rgba(128,128,128,.12)"), xaxis=dict(showgrid=False), legend=dict(orientation="h", y=1.08, x=0))
    return fig


ASSETS = load_system_artifacts()
PROJECT_DATASETS = find_project_datasets()
state_defaults = {
    "history": pd.DataFrame(), "live_snapshot": None, "processes": pd.DataFrame(),
    "live_running": False, "last_rca": None, "last_rca_timestamp": None, "last_mode": None,
}
for key, value in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


with st.sidebar:
    st.markdown("## OS Monitor")
    st.caption("Live telemetry and bottleneck classification")
    modes = ["Live host"]
    if PROJECT_DATASETS:
        modes.append("Recorded dataset")
    modes.append("Upload CSV")
    mode = st.radio("Data source", modes)

    if mode != st.session_state.last_mode:
        st.session_state.history = pd.DataFrame()
        st.session_state.live_snapshot = None
        st.session_state.processes = pd.DataFrame()
        st.session_state.last_rca = None
        st.session_state.last_mode = mode

    model_options = list(ASSETS["models"])
    model_options.append("Rule_Based_Heuristic")
    default_name = DASHBOARD_CFG.get("default_model", "Random_Forest")
    default_index = model_options.index(default_name) if default_name in model_options else 0
    selected_model_name = st.selectbox("Inference model", model_options, index=default_index)

    refresh_seconds = DEFAULT_REFRESH
    if mode == "Live host":
        refresh_seconds = st.slider("Refresh interval", 0.5, 5.0, DEFAULT_REFRESH, 0.5)
        control_a, control_b = st.columns(2)
        if control_a.button("Start", type="primary", width="stretch", disabled=st.session_state.live_running):
            st.session_state.live_running = True
            st.rerun()
        if control_b.button("Stop", width="stretch", disabled=not st.session_state.live_running):
            st.session_state.live_running = False
            st.rerun()
        if st.button("Refresh once", width="stretch", disabled=st.session_state.live_running):
            st.rerun()

    dataset_path = None
    if mode == "Recorded dataset":
        selected_dataset = st.selectbox("Dataset", list(PROJECT_DATASETS))
        dataset_path = PROJECT_DATASETS[selected_dataset]

    uploaded_file = None
    if mode == "Upload CSV":
        uploaded_file = st.file_uploader("Telemetry CSV", type=["csv"])

    st.divider()
    st.markdown("#### Runtime")
    if ASSETS["ready"]:
        st.success(f"{len(ASSETS['models'])} trained model(s) loaded")
    else:
        st.info("Rule-based inference active")
    llm_state = gemini_status()
    st.caption(f"Gemini: {'ready' if llm_state['ready'] else 'fallback'} · {llm_state['model']}")
    with st.expander("Diagnostics"):
        for error in ASSETS["errors"]:
            st.write(f"• {error}")
        st.write(f"Host: {platform.system()} · {psutil.cpu_count(logical=True)} logical CPUs")


if mode == "Recorded dataset":
    source_frame = normalise_frame(read_csv_file(str(dataset_path)))
elif mode == "Upload CSV":
    if uploaded_file is None:
        st.title("OS Bottleneck Monitor")
        st.info("Upload a telemetry CSV from the sidebar to begin.")
        st.stop()
    try:
        source_frame = normalise_frame(pd.read_csv(uploaded_file))
    except Exception as exc:
        st.error(f"The CSV could not be read: {exc}")
        st.stop()
else:
    source_frame = None

run_every = refresh_seconds if mode == "Live host" and st.session_state.live_running else None
fragment = getattr(st, "fragment", None)
if fragment is None:
    fragment = lambda **_: (lambda function: function)


@fragment(run_every=run_every)
def render_dashboard() -> None:
    if mode == "Live host":
        if st.session_state.live_snapshot is None:
            st.session_state.live_snapshot = initial_counter_snapshot()
        current_row, st.session_state.live_snapshot, st.session_state.processes = get_telemetry_snapshot(st.session_state.live_snapshot)
        current_frame = normalise_frame(pd.DataFrame([current_row]))
        history = pd.concat([st.session_state.history, current_frame], ignore_index=True).tail(HISTORY_LIMIT)
        st.session_state.history = normalise_frame(history)
        selected_index = len(st.session_state.history) - 1
    else:
        selected_index = st.slider("Sample", 0, len(source_frame) - 1, len(source_frame) - 1)
        current_frame = source_frame.iloc[[selected_index]]
        st.session_state.history = normalise_frame(source_frame.iloc[max(0, selected_index - HISTORY_LIMIT + 1):selected_index + 1].copy())

    current_row = current_frame.iloc[0].to_dict()
    try:
        predicted_class, classes, probabilities = model_inference(current_row, selected_model_name, ASSETS)
        inference_error = None
    except Exception as exc:
        predicted_class, classes, probabilities = heuristic_inference(current_row, CANONICAL_CLASSES)
        inference_error = str(exc)

    title_col, status_col = st.columns([4, 1])
    with title_col:
        st.title("OS Bottleneck Monitor")
        st.caption(f"{mode} · {selected_model_name.replace('_', ' ')} · Updated {current_row['timestamp']}")
    with status_col:
        indicator = "live-dot" if st.session_state.live_running and mode == "Live host" else "paused-dot"
        text = "Live" if st.session_state.live_running and mode == "Live host" else "Snapshot"
        st.markdown(f'<div class="live-row"><span class="{indicator}"></span>{text}</div>', unsafe_allow_html=True)

    if inference_error:
        st.warning(f"Model inference failed; this sample uses the local heuristic. {inference_error}")

    cpu_total = max(0.0, 100.0 - float(current_row["cpu_idle_pct"]))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("CPU active", f"{cpu_total:.1f}%", f"{float(current_row['cpu_user_pct']):.1f}% user")
    k2.metric("Memory used", f"{float(current_row['mem_used_pct']):.1f}%", f"{float(current_row['mem_available_mb']):,.0f} MiB available")
    k3.metric("Disk write", f"{float(current_row['disk_write_kb_per_sec']) / 1024:.2f} MiB/s", f"{float(current_row['cpu_iowait_pct']):.1f}% I/O wait")
    k4.metric("Current state", predicted_class.replace("_", " "), f"{max(probabilities):.1%} confidence")

    overview_tab, process_tab, assistant_tab = st.tabs(["Overview", "Processes", "Root-cause assistant"])

    with overview_tab:
        left, right = st.columns([1.05, 1.4])
        with left:
            st.markdown("### Classification")
            severity_label = "Healthy" if predicted_class == "Normal" else "Attention"
            st.markdown(f'<span class="status-pill">{severity_label} · {predicted_class.replace("_", " ")}</span>', unsafe_allow_html=True)
            st.plotly_chart(probability_figure(classes, probabilities), width="stretch", config={"displayModeBar": False})
        with right:
            st.markdown("### Pressure signals")
            signal_rows = []
            for name, value, threshold, unit in pressure_signals(current_row):
                signal_rows.append({"Signal": name, "Current": f"{value:,.1f} {unit}", "Reference": f"{threshold:,.1f} {unit}", "Pressure": min(value / max(threshold, 1e-9), 9.99)})
            signal_df = pd.DataFrame(signal_rows)
            st.dataframe(
                signal_df,
                hide_index=True,
                width="stretch",
                column_config={"Pressure": st.column_config.ProgressColumn("Threshold ratio", min_value=0.0, max_value=1.0, format="%.2f×")},
            )
        st.markdown("### Resource history")
        st.plotly_chart(history_figure(st.session_state.history), width="stretch", config={"displayModeBar": False})

    with process_tab:
        st.markdown("### Process inspector")
        if mode != "Live host":
            st.info("Per-process attribution is available in Live host mode.")
        elif st.session_state.processes.empty:
            st.info("Collecting process deltas. The next sample will include CPU attribution.")
        else:
            st.dataframe(
                st.session_state.processes,
                hide_index=True,
                width="stretch",
                column_config={
                    "pid": st.column_config.NumberColumn("PID", format="%d"),
                    "process": "Process",
                    "cpu_pct": st.column_config.NumberColumn("CPU", format="%.1f%%"),
                    "memory_mb": st.column_config.NumberColumn("Memory", format="%.1f MiB"),
                },
            )
            st.caption("Process CPU is measured between consecutive samples and may exceed 100% for multithreaded processes.")

    with assistant_tab:
        st.markdown("### Root-cause assistant")
        st.caption("Gemini receives the selected telemetry sample only when you click Analyze current sample.")
        if st.button("Analyze current sample", type="primary", key="analyze_sample"):
            top_features = {f"{name} ({unit})": round(value, 3) for name, value, _, unit in pressure_signals(current_row)[:4]}
            telemetry_for_rca = {key: value for key, value in current_row.items() if key != "label"}
            if mode == "Live host" and not st.session_state.processes.empty:
                telemetry_for_rca["top_processes"] = st.session_state.processes.head(5).to_dict("records")
            with st.spinner("Analyzing telemetry…"):
                st.session_state.last_rca = diagnose_bottleneck(
                    predicted_class=predicted_class,
                    confidence=float(max(probabilities)),
                    telemetry_snapshot=telemetry_for_rca,
                    top_features=top_features,
                )
                st.session_state.last_rca_timestamp = str(current_row["timestamp"])

        if st.session_state.last_rca is None:
            st.info("Run an analysis to get evidence, investigation steps, and cautious remediation guidance.")
        else:
            report = st.session_state.last_rca
            status = report.get("status", "unknown")
            online = status == "online_gemini_agent"
            st.caption(f"Analyzed sample {st.session_state.last_rca_timestamp} · {'Gemini' if online else 'Local fallback'}")
            if not online and report.get("fallback_reason"):
                st.warning(report["fallback_reason"])
            detail, commands = st.columns([1.15, 1.0])
            with detail:
                st.markdown(f"**Severity:** `{report.get('severity_level', 'N/A')}`")
                st.markdown(f"**Likely cause:** {report.get('root_cause', 'N/A')}")
                st.markdown(f"**Evidence:** {report.get('technical_analysis', 'N/A')}")
            with commands:
                st.markdown("**Investigate**")
                for command in report.get("investigation_commands", []):
                    st.code(command, language="powershell" if platform.system() == "Windows" else "bash")
                st.markdown("**Remediate carefully**")
                for command in report.get("remediation_commands", []):
                    st.code(command, language=None)


render_dashboard()

if run_every and getattr(st, "fragment", None) is None:
    st.warning("Live auto-refresh requires Streamlit 1.37 or newer. Upgrade Streamlit to enable it.")
