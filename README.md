# OS Anomaly Analytics

Anomaly detection and root-cause analysis for server OS metrics. The project compares
**10 ML models** (2 of them neural networks) for detecting anomalies on production
servers, and explains incidents with **Gemini-powered root-cause analysis (RCA)**.
Everything is shown in a Flask + HTML/CSS/JS dashboard.

It grew out of an earlier "OS analytics and bottleneck classifier" project. That
project's labels were the stress test being run, so every model scored near-perfectly
and the comparison meant nothing. This version uses real labelled incidents from
production servers instead.

**Dataset:** [SMD, the Server Machine Dataset](https://github.com/NetManAIOps/OmniAnomaly)
(Su et al., KDD 2019). It covers 28 production servers with 38 OS-level metrics each
(CPU, memory, disk, network and so on, names anonymised), sampled once a minute over
about 5 weeks. That is 708,420 labelled minutes, 4.2% of them anomalies.

## Models

| # | Model | Type | Idea |
|---|---|---|---|
| 1 | Logistic Regression | supervised | linear boundary, class-balanced |
| 2 | Random Forest | supervised | many independent decision trees vote |
| 3 | XGBoost | supervised | gradient-boosted trees |
| 4 | LightGBM | supervised | gradient-boosted trees, leaf-wise growth (fast on large data) |
| 5 | K-Nearest Neighbours | supervised | vote of the 15 most similar labelled minutes |
| 6 | **Neural Net (MLP)** | supervised | feed-forward network, layers 128 → 64 |
| 7 | Isolation Forest | unsupervised | unusual minutes are isolated in fewer random splits |
| 8 | One-Class SVM | unsupervised | curved boundary around normal data |
| 9 | Local Outlier Factor | unsupervised | flags minutes in sparse regions |
| 10 | **Autoencoder (NN)** | unsupervised | network that rebuilds normal minutes (114 → 64 → 16 → 64 → 114); high rebuild error = anomaly |

*Supervised* models learn from labelled anomalies on **other** servers. *Unsupervised*
models learn each server's **own** normal behaviour and need no labels.

## Results (all 28 servers, 4-fold cross-validation by server)

| # | Model | Type | F1 (mean ± std) | Paper-style F1 | Precision | Recall | PR-AUC | Incidents caught | First alarm | False alarms / day |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Random Forest | supervised | 0.357 ± 0.139 | 0.474 | 0.30 | 0.47 | 0.338 | 82% | 1.5 min | 4.0 |
| 2 | Logistic Regression | supervised | 0.354 ± 0.159 | 0.508 | 0.46 | 0.31 | 0.374 | 63% | 3.0 min | 2.5 |
| 3 | K-Nearest Neighbours | supervised | 0.319 ± 0.162 | 0.409 | 0.33 | 0.32 | 0.244 | 81% | 2.1 min | 3.3 |
| 4 | XGBoost | supervised | 0.306 ± 0.131 | 0.490 | 0.38 | 0.35 | 0.342 | 68% | 2.4 min | 3.7 |
| 5 | LightGBM | supervised | 0.299 ± 0.113 | 0.512 | 0.39 | 0.36 | 0.369 | 71% | 2.4 min | 2.5 |
| 6 | Neural Net (MLP) | supervised | 0.228 ± 0.169 | 0.415 | 0.18 | 0.44 | 0.218 | 79% | 0.8 min | 9.5 |
| 7 | Autoencoder (NN) | unsupervised | 0.217 ± 0.131 | **0.529** | 0.39 | 0.20 | 0.289 | 39% | 1.8 min | 0.9 |
| 8 | Local Outlier Factor | unsupervised | 0.175 ± 0.083 | 0.445 | 0.14 | 0.28 | 0.173 | 38% | 1.6 min | 3.0 |
| 9 | Isolation Forest | unsupervised | 0.172 ± 0.073 | 0.456 | 0.30 | 0.18 | 0.235 | 39% | 2.8 min | 2.9 |
| 10 | One-Class SVM | unsupervised | 0.135 ± 0.062 | 0.420 | 0.08 | 0.52 | 0.129 | 68% | 0.3 min | 6.2 |

There are 327 real incidents in total.

### Two ways of scoring F1

| | **F1** (our main number) | **Paper-style F1** |
|---|---|---|
| Alarm threshold | Tuned on *other* servers, never on test labels | The best threshold for each server, picked **using its test labels** |
| Averaged over | All held-out minutes in a fold | Servers (one score per server) |
| Realistic? | Yes: this is what a deployed monitor would get | No: optimistic, since a real system can't see the answers |
| Use it for | Judging the models | Comparing with published work |

Published SMD results under the paper-style protocol (strict per-minute F1, without
point-adjust):

| Method | F1 on SMD | Source |
|---|---|---|
| Random scores | 0.080 | Kim et al. 2022; Sarfraz et al. 2024 |
| DAGMM | 0.238 / 0.435 | Kim et al. 2022 / Sarfraz et al. 2024 |
| USAD | 0.426 | both |
| Anomaly Transformer | 0.426 | Sarfraz et al. 2024 |
| OmniAnomaly | 0.474 / 0.415 | Kim et al. 2022 / Sarfraz et al. 2024 |
| TranAD | 0.457 | Sarfraz et al. 2024 |
| GDN | 0.529 / 0.526 | Kim et al. 2022 / Sarfraz et al. 2024 |
| PCA reconstruction error | 0.572 | Sarfraz et al. 2024 |

Our models score 0.41–0.53 on the same protocol, in line with these. Many papers quote
0.9+ on SMD, but those numbers use point-adjust, under which random scores already
reach 0.80–0.89.

Sources: [Kim et al., AAAI 2022](https://arxiv.org/abs/2109.05257) ·
[Sarfraz et al., ICML 2024](https://arxiv.org/pdf/2405.02678)

**Root-cause ranking:** on those 327 incidents, ranking metrics by peak deviation puts
**77%** of SMD's labelled culprit metrics at the top (HitRate@100%), and 88% at
HitRate@150%. A random ranking manages 19%.

## Testing criteria

Anomalies are rare (4.2% of minutes), so **accuracy is useless**: a model that never
raises an alarm would score 95.8%. The project reports:

| Criterion | Question it answers |
|---|---|
| **Precision** | Of the minutes we flagged, how many were really anomalous? |
| **Recall** | Of the anomalous minutes, how many did we flag? |
| **F1** (strict, point-wise) | Balance of the two; the main ranking metric |
| **PR-AUC** | Ranking quality at every threshold, and the right curve for imbalanced data |
| **ROC-AUC** | Ranking quality, but looks optimistic when anomalies are rare |
| **Incidents caught** | Share of true incidents with at least one alarm (the operator's view) |
| **Detection delay** | Median minutes from incident start to the first alarm |
| **False alarms / day** | Alarm incidents per server per day that touch no labelled anomaly |
| **Paper-style F1** | Per-server F1 at the best threshold picked from test labels; for comparing with published work |
| **PA-F1** (lenient) | Point-adjusted F1 used in many papers. Random scores reach about 0.8 with it, so it is shown for reference only and never used for ranking |
| **Fit time / latency** | Cost to train, and to score 1,000 minutes |
| **Mean ± std over folds** | How stable the result is across different held-out servers |
| **RCA HitRate@100%** | How many of SMD's labelled culprit metrics the RCA ranks at the top |

Every score comes from **servers the model never saw**: 4-fold cross-validation by
server, with alarm thresholds tuned on separate calibration servers.

## Quick start

Requires **Python 3.13** (PyYAML has no prebuilt wheel for 3.15 on Windows yet).

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # then paste your GEMINI_API_KEY (optional)

python -m src.pipeline all        # download SMD (~460 MB) + train/evaluate all 28 servers
python -m dashboard.server        # open http://127.0.0.1:8000
```

For a fast demo with 8 servers (about 2 minutes):
`python -m src.pipeline train --config config.quick.yaml`

Tests use synthetic data, so no download is needed: `python -m pytest -q`

## Method

1. **Features** (`src/features.py`): per minute, the raw value plus the 10-minute rolling
   mean and rolling std of each of the 38 metrics (114 features). Only past data is used.
2. **Baseline normalisation** (`src/evaluate.py`): each feature becomes a z-score against
   that server's *own unlabelled normal period*, which answers "is this unusual for this
   server?". No labels are used in this step.
3. **Models** (`src/models.py`): the 10 above, behind one `fit` / `score` interface.
4. **Evaluation**: GroupKFold over servers (4 folds); the test servers are touched once.
   - **Supervised thresholds, by cross-fitting:** the training servers are split into 3
     groups. Each group is scored by a model trained on the other two, so every training
     server gets an honest score, and the alarm threshold (best F1) is tuned on all of
     them. The final model is then trained on all training servers.
   - **Unsupervised thresholds:** tuned on all training servers, since no labels were
     used to fit these models.
   - **Independent random streams:** each model has its own random stream, so adding or
     removing a model never changes another model's results.
5. **RCA** (`src/rca.py`):
   - **Evidence** (numbers only, no LLM): peak and average deviation of every metric,
     which metric moved first, which metrics move together, the anomaly's shape, which
     models agree, and similar past incidents on the same server.
   - **Gemini chain**: `gemini-4-hydrogen` → `gemini-3.8-flash` → `gemini-3.6-flash` →
     `gemini-3.1-flash-lite` → offline rules. Each model gets 40 s, and every attempt is
     shown in the dashboard.
   - **Report**: Pydantic-validated, with severity, summary, timeline, likely cause,
     evidence and **read-only** next steps.
   - **Caching and export**: answers are cached per incident and can be exported as Markdown.
   - **Ground truth stays separate**: SMD's interpretation labels are shown next to the
     report for comparison and are never sent to Gemini.

## What was tried to improve the models

Each idea was tested on all 28 servers with the same folds, and kept only if it helped.
Supervised average F1 is the mean over the 6 supervised models.

| Change | Result | Kept? |
|---|---|---|
| Smooth scores over 3–10 minutes | No gain (the rolling-window features already smooth) | No |
| Rescale supervised scores per server against its normal period | Average F1 0.294 → 0.245, worse for all 6 models | No |
| Add "sudden jump" and "slow drift" features | Average F1 0.294 → 0.279 (Random Forest and Logistic Regression up, 4 models down) | No |
| Peak instead of average deviation to rank RCA metrics | Culprit HitRate@100% 0.61 → 0.73 (random: 0.22) | **Yes** |
| Cross-fitted thresholds for supervised models | Average F1 0.294 → 0.311, PR-AUC up for all 6, KNN false alarms 7.2 → 3.3 a day | **Yes** |

## Dashboard

| Page | What it shows |
|---|---|
| **Overview** | Four headline numbers, one model-ranking chart that switches between F1 / incidents caught / PR-AUC / false alarms, and four plain-language findings |
| **Models** | Filter (all / supervised / unsupervised / neural), sortable table (extra metrics behind a switch), PR curve, confusion matrix, per-fold results, per-server F1 heatmap |
| **Fleet** | All servers on one timeline (alarms vs labelled anomalies), sortable per-server table |
| **Explorer** | Score timeline, an optional strip comparing when all 10 models alarm, raw metrics, incidents, and the RCA panel (main findings first, details in collapsible sections) |

## Reproducibility

- **One command:** `python -m src.pipeline all --config config.yaml`. It never prompts.
- **One config file** for everything (`config.yaml`). `config.quick.yaml` inherits from it.
- **Fixed seed (42)** for every random step. Re-running gives identical metrics.
- **Dataset checksum:** a SHA-256 for every file is stored in `data/raw/SMD/MANIFEST.json`.
- **Every run is saved** to `runs/<run_id>/`:
  - `config.yaml` (snapshot)
  - `manifest.json` (seed, checksum, versions, timing, platform)
  - `leaderboard.json`, `folds.json`, `machines.json`, `curves.json`, `rca_eval.json`
  - `scores/*.npz` (out-of-fold scores)
- **Pinned** `requirements.txt`, a Dockerfile, and CI (`.github/workflows/ci.yml`).

## Modularity

```
config.yaml / config.quick.yaml   all settings
src/
  config.py     load config (with `extends:`), resolve paths
  data.py       download + checksum + load SMD
  features.py   window features (one function, used everywhere)
  models.py     model registry: 10 detectors behind one fit/score interface
  evaluate.py   baseline normalisation, cross-validation, all testing criteria
  rca.py        evidence builder, Gemini chain, offline fallback, RCA evaluation
  pipeline.py   CLI: download | train | all  → runs/<run_id>/
dashboard/
  server.py     Flask JSON API (read-only over runs/, plus the RCA cache)
  static/       index.html, styles.css, app.js (Chart.js)
tests/          pytest on synthetic data
```

**Adding a model** takes one entry in `REGISTRY` (`src/models.py`) and one block in
`config.yaml`. Set `enabled: false` to drop a model.

## Limitations

- SMD metric names are anonymised (`m01`–`m38`). The RCA therefore reasons from
  patterns (which metrics move, how much, in what order) rather than from what each
  metric means.
- Strict point-wise F1 is harsh: an alarm one minute late counts as a miss. Incidents
  caught and detection delay give the operational view.
- Supervised models need labelled anomalies from other servers. Unsupervised models need
  none, which is why both families are compared.
