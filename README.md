# os-bottleneck-classifier
<div align="center">

# 🖥️ OS Bottleneck Classifier

### A Modular Telemetry-Driven Framework for Operating System Bottleneck Classification and LLM-Assisted Root Cause Remediation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Linux-informational?logo=linux&logoColor=white)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM%20%7C%20PyTorch-orange)
![LLM](https://img.shields.io/badge/LLM-Google%20Gemini-8E75B2)
![UI](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

</div>

---

## 📌 Overview

Traditional monitoring tools (`top`, `vmstat`, `iostat`, Nagios, Zabbix) rely on **single-variable static thresholds** such as "alert if CPU > 80%". These break down in real systems, where bottlenecks **propagate across subsystems** (for example, heavy disk writes trigger page-cache writeback, which raises kernel CPU time and puts processes into uninterruptible D-state sleep).

**OS Bottleneck Classifier** is an end-to-end, reproducible, non-intrusive pipeline that:

1. 📡 Samples **15 kernel performance counters at 1 Hz**
2. 🧠 Classifies the active **bottleneck regime** with sub-millisecond inference
3. 🤖 Uses an **LLM (Google Gemini)** to produce a structured root-cause diagnosis plus **bash remediation runbooks**
4. 📊 Shows everything in a live **Streamlit observability dashboard**

---

## 🎯 Target Classes

| Class | Description |
|---|---|
| ✅ **Normal** | Nominal state; CPU, memory and storage have healthy headroom |
| 🔥 **CPU_Bound** | Compute-heavy processes saturate execution pipelines and scheduler queues |
| 🧮 **Memory_Pressure** | RAM exhaustion forces aggressive page reclamation and swap churn |
| 💾 **IO_Contention** | Storage saturation with processes blocked in uninterruptible sleep |

---

## 📡 Telemetry Features (15 counters @ 1 Hz)

| Group | Feature | Description |
|---|---|---|
| **CPU** | `cpu_user_pct` | User-space CPU execution % |
| | `cpu_system_pct` | Kernel-space CPU execution % |
| | `cpu_iowait_pct` | % of time CPU stalls waiting for disk I/O |
| | `cpu_idle_pct` | Unallocated CPU headroom % |
| | `load_avg_1m_per_cpu` | 1-min load average normalized by core count |
| | `context_switches_per_sec` | Rate of thread context switches |
| | `interrupts_per_sec` | Rate of hardware interrupts handled |
| **Memory** | `mem_used_pct` | Resident RAM consumption % |
| | `mem_available_mb` | Physical memory available for allocation (MB) |
| | `swap_used_pct` | Swap space consumption % |
| **Disk** | `disk_read_kb_per_sec` | Block read throughput (KB/s) |
| | `disk_write_kb_per_sec` | Block write throughput (KB/s) |
| | `disk_busy_pct` | Disk service utilization % |
| **Network** | `net_rx_kb_per_sec` | Ingress throughput (KB/s) |
| | `net_tx_kb_per_sec` | Egress throughput (KB/s) |

---

## 🏗️ System Architecture

```text
  [Workload Generator (generate_dataset.py)]
                    │
                    ▼
  [Raw Telemetry Log (data/raw/telemetry_raw.csv)]
                    │
                    ▼
  [Group-Based Temporal Splitter & Noise Injector (preprocess.py)]
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
    [train.csv]           [test.csv]
         │                     │
         ▼                     │
  [GridSearchCV (5-Fold Stratified Group CV)]
         │
         ▼
  [Model Artifacts: RF, XGB, LGBM, SVM, LR, MLP, 1D-CNN (artifacts/)]
         │
         ▼
  [Interactive Streamlit Dashboard (app.py)] ◄── [Live psutil / CSV Playback]
         │
         ▼
  [Gemini Root Cause Copilot (llm_agent.py)] ──► [Bash Remediation Runbooks]
```

### 🔒 Leakage-Safe Data Partitioning

- **Splitting unit:** independent run batches (`run_id`), **never random rows**
- **Split:** 80% train (26 runs) / 20% test (6 held-out runs)
- **Preprocessing:** `StandardScaler` and `LabelEncoder` are fitted **only on training folds**
- **Cross-validation:** 5-fold `StratifiedGroupKFold` with full run isolation

---

## 📊 Benchmark Results

Evaluated on **held-out test runs** across 8 models (1 legacy baseline, 5 classical ML, 2 deep learning):

| Model | Type | CV Macro F1 | Test Acc | Precision | Recall | Test F1 | Latency (ms) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Static Rule Baseline | Legacy Heuristic | N/A | 0.5511 | 0.5677 | 0.6633 | 0.5541 | 0.0314 |
| Logistic Regression | Classical ML (Linear) | 0.9653 | 0.9978 | 0.9967 | 0.9983 | 0.9975 | 0.0057 |
| Random Forest | Classical ML (Bagged) | 0.9993 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1230 |
| **XGBoost** 🏆 | Classical ML (Boosted) | 0.9935 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.0061** |
| LightGBM | Classical ML (Boosted) | 0.9988 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0078 |
| SVM (RBF) | Classical ML (Kernel) | 0.9556 | 0.9956 | 0.9951 | 0.9950 | 0.9950 | 0.0140 |
| MLP Deep Net | Deep Learning (MLP) | 0.9613 | 0.9978 | 0.9967 | 0.9983 | 0.9975 | 0.0633 |
| 1D-CNN | Deep Learning (CNN) | N/A | 0.9844 | 0.9777 | 0.9850 | 0.9810 | 0.0907 |

### 🔍 Key Insights

- **Static rules fail:** the threshold baseline reaches only **55.41% Macro F1**, because coupled events (such as writeback overhead during heavy I/O) fool single-variable thresholds.
- **Trees beat deep nets on tabular data:** axis-aligned splits handle abrupt telemetry step-changes cleanly.
- **Pareto champion:** **XGBoost** reaches **1.0000 Macro F1** at **0.0061 ms** inference, which is under 0.001% of the 1,000 ms sampling interval.
- **Generalization:** grouped 5-fold CV scores (0.9556 to 0.9993) support generalization without temporal leakage.

---

## 🤖 LLM SysAdmin Agent

- **Provider:** Google Gemini API, with an **offline deterministic fallback** if the API is unavailable
- **Decoding:** temperature `0.1`, JSON output (`application/json`)
- **Validation:** client-side **Pydantic** schema (`BottleneckDiagnosisReport`)
- **Model:** configurable in `config.yaml`

<details>
<summary><b>📄 Sample output (IO_Contention incident)</b></summary>

```json
{
  "severity_level": "HIGH",
  "root_cause": "Block I/O storage saturation caused by sustained write throughput (~98.4 MB/s), driving request queues into backlog.",
  "technical_analysis": "Elevated cpu_iowait_pct of 58.6% with near-zero cpu_idle_pct (0.9%) indicates CPU tasks are stalled in D-state awaiting storage flushes.",
  "investigation_commands": [
    "iotop -b -n 3 -o -d 2",
    "iostat -xz 1 5",
    "pidstat -d 1 5"
  ],
  "remediation_commands": [
    "renice -n 19 -p $(iotop -b -n 1 -o | awk 'NR>3 {print $1; exit}')",
    "ionice -c 3 -p $(iotop -b -n 1 -o | awk 'NR>3 {print $1; exit}')",
    "sysctl -w vm.dirty_background_ratio=5 vm.dirty_ratio=10"
  ],
  "status": "online_gemini_agent",
  "confidence_pct": 98.5
}
```

</details>

---

## 📁 Project Structure

```text
os-bottleneck-classifier/
├── app.py                  # Streamlit dashboard (live streaming + CSV playback)
├── config.yaml             # Central configuration
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Dashboard theme
├── src/
│   ├── generate_dataset.py # Workload injection + telemetry collection
│   ├── preprocess.py       # Run-grouped split + noise injection
│   ├── train.py            # Training, GridSearchCV, benchmarking
│   └── llm_agent.py        # Gemini RCA + remediation agent
├── data/
│   ├── raw/                # Raw telemetry logs
│   └── processed/          # train.csv / test.csv
└── artifacts/              # Trained models, scaler, encoder, metrics
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/os-bottleneck-classifier.git
cd os-bottleneck-classifier
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. (Optional) Enable the Gemini agent

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> Without a key, the dashboard still works and uses the deterministic offline fallback.

### 4. Run the pipeline

```bash
# Collect telemetry under injected workloads (Linux)
python src/generate_dataset.py

# Run-grouped train/test split
python src/preprocess.py

# Train and benchmark all models
python src/train.py
```

> Pre-trained models and processed data are already included in `artifacts/` and `data/`, so you can skip straight to the dashboard.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

---

## ⚠️ Limitations (Threats to Validity)

- Near-perfect scores come from **controlled synthetic workloads**. Real multi-tenant production systems will show more variance.
- The 1D-CNN adds latency without improving accuracy over tree models.
- Cloud LLM calls are subject to rate limits and availability (for example 503 errors), which the offline fallback is designed to handle.

## 🔭 Future Scope

- Replace `psutil` with **eBPF** probes for microsecond-granularity telemetry
- Execute non-destructive remediation runbooks through authenticated local agent daemons
- Extend the taxonomy to **network socket starvation** and **thread deadlock** states
- Deploy as a **Kubernetes sidecar** monitoring agent

---

## 📚 References

1. Gregg, B. (2020). *Systems Performance: Enterprise and the Cloud* (2nd ed.). Addison-Wesley.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* ACM SIGKDD.
3. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
4. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5-32.
5. Google DeepMind. (2024). *Gemini: A Family of Highly Capable Multimodal Models.*
6. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825-2830.

---

## 👥 Authors

- **Your Name** (add team members here)

## 📄 License

This project is licensed under the MIT License. Add a `LICENSE` file if you choose this.
