/* OS Anomaly Analytics: front-end. Plain JS + Chart.js, no build step. */
"use strict";

/* ================================================================ utils */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 3) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const pct = (v, d = 0) => (v == null || Number.isNaN(v) ? "–" : (100 * v).toFixed(d) + "%");
const big = (n) => Number(n).toLocaleString();
const md = (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>");
const minuteLabel = (m) => {
  const d = Math.floor(m / 1440), h = Math.floor((m % 1440) / 60), mm = m % 60;
  return `day ${d}, ${String(h).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
};
const shortTime = (m) => { const d = Math.floor(m / 1440), h = Math.floor((m % 1440) / 60); return `d${d} ${String(h).padStart(2, "0")}h`; };
const shortM = (m) => m.replace("machine-", "");
const natural = (a, b) => a.localeCompare(b, undefined, { numeric: true });

const KIND = { supervised: { label: "Supervised", v: "--sup" }, unsupervised: { label: "Unsupervised", v: "--uns" } };
const kindColor = (k) => css((KIND[k] || KIND.supervised).v);
const nnBadge = (r) => (r.neural ? `<span class="nn" title="Neural network">NN</span>` : "");
const kindTag = (k) => `<span class="tag"><i style="background:${kindColor(k)}"></i>${(KIND[k] || KIND.supervised).label}</span>`;
const delayTxt = (v, d = 1) => (v == null ? "–" : `${fmt(v, d)} min`);

/* sequential ramp (dark mode: low = near surface, high = light blue) */
const RAMP = ["#152029", "#173a63", "#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4"];
function hex2rgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function seq(t) {
  t = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
  const i = Math.min(Math.floor(t), RAMP.length - 2), f = t - i;
  const a = hex2rgb(RAMP[i]), b = hex2rgb(RAMP[i + 1]);
  return `rgb(${a.map((x, k) => Math.round(x + (b[k] - x) * f)).join(",")})`;
}
const rampCss = `linear-gradient(90deg, ${RAMP.join(",")})`;

/* metrics the ranking chart can show */
const RANK = {
  f1: { label: "F1", sub: "Strict F1 per minute on servers the model never saw. Whiskers show the spread across the 4 folds.", show: (v) => fmt(v), sd: (v) => fmt(v, 2), higher: true },
  paper_f1: { label: "Paper F1", sub: "F1 the way published SMD papers report it: each server gets its best threshold, picked using its test labels. Optimistic, so use it only to compare with the literature. Whiskers show the spread across servers.", show: (v) => fmt(v), sd: (v) => fmt(v, 2), higher: true },
  event_recall: { label: "Incidents", sub: "Share of real incidents that got at least one alarm. Whiskers show the spread across folds.", show: (v) => pct(v), sd: (v) => pct(v), higher: true },
  pr_auc: { label: "PR-AUC", sub: "How well the scores rank anomalous minutes above normal ones, at any threshold. Whiskers show the spread across folds.", show: (v) => fmt(v), sd: (v) => fmt(v, 2), higher: true },
  false_alarms_per_day: { label: "False alarms", sub: "Alarms per server per day that match no real anomaly, so lower is better. Whiskers show the spread across folds.", show: (v) => fmt(v, 1), sd: (v) => fmt(v, 1), higher: false },
};

const state = {
  run: null, data: null, geminiModels: [], rankMetric: "f1",
  selectedModel: null, modelFilter: "all", moreCols: false, lbSort: { key: "f1", dir: -1 },
  fleetModel: null, fleet: null, fleetSort: { key: "machine", dir: 1 },
  machine: null, model: null, metrics: [], zoom: null, mdata: null, incSort: { key: "start", dir: 1 },
  showAgree: false,
};
const charts = {};
const labelW = () => (innerWidth < 600 ? 92 : 150);

async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || r.statusText);
  return body;
}

/* ================================================================ tooltip */
const tip = $("#tip");
function showTip(html, ev) {
  tip.innerHTML = html; tip.style.opacity = 1;
  const x = Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 8);
  const y = Math.min(ev.clientY + 14, innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => (tip.style.opacity = 0);

/* ================================================================ sortable tables */
function renderTable(el, cols, rows, opt = {}) {
  const s = opt.sort;
  if (s) {
    const col = cols.find((c) => c.key === s.key);
    const val = col?.sortVal || ((r) => r[s.key]);
    rows = [...rows].sort((a, b) => {
      const x = val(a), y = val(b);
      if (x == null) return 1; if (y == null) return -1;
      return (typeof x === "string" ? natural(x, y) : x - y) * s.dir;
    });
  }
  const best = {};
  for (const c of cols) if (c.best) {
    const vals = rows.map((r) => r[c.key]).filter((v) => v != null);
    if (vals.length) best[c.key] = c.best === "min" ? Math.min(...vals) : Math.max(...vals);
  }
  el.innerHTML = `<thead><tr>${cols.map((c) => `<th class="${c.num ? "num" : ""} ${s && c.sort !== false ? "sortable" : ""}" data-k="${c.key}" ${c.title ? `title="${esc(c.title)}"` : ""}>${c.label}${s && s.key === c.key ? `<span class="arrow">${s.dir > 0 ? "↑" : "↓"}</span>` : ""}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r, i) => `<tr data-i="${i}" class="${opt.onRow ? "clickable" : ""} ${opt.selected && opt.selected(r) ? "selected" : ""}">${cols.map((c) =>
      `<td class="${c.num ? "num" : ""} ${c.best && r[c.key] != null && r[c.key] === best[c.key] ? "best-cell" : ""}">${c.fmt ? c.fmt(r) : esc(r[c.key])}</td>`).join("")}</tr>`).join("")}</tbody>`;
  if (s) $$("th.sortable", el).forEach((th) => th.addEventListener("click", () => {
    const c = cols.find((x) => x.key === th.dataset.k);
    s.dir = s.key === th.dataset.k ? -s.dir : (c.num ? (c.best === "min" ? 1 : -1) : 1);
    s.key = th.dataset.k; renderTable(el, cols, rows, opt);
  }));
  if (opt.onRow) $$("tbody tr", el).forEach((tr) => tr.addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    opt.onRow(rows[+tr.dataset.i]);
  }));
  if (opt.after) opt.after(el, rows);
}

/* ================================================================ navigation */
function go(view) {
  $$("nav.tabs button").forEach((x) => x.classList.toggle("active", x.dataset.view === view));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + view));
  if (view === "explorer") { if (!state.mdata) loadMachine(); else requestAnimationFrame(drawAgreement); }
  if (view === "fleet") { if (!state.fleet) loadFleet(); else requestAnimationFrame(drawFleet); }
  scrollTo({ top: 0 });
}
$$("nav.tabs button").forEach((b) => b.addEventListener("click", () => go(b.dataset.view)));
$$("[data-go]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); go(a.dataset.go); }));
let resizeT;
addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(() => { drawFleet(); drawAgreement(); }, 120); });

/* ================================================================ boot */
async function boot() {
  Chart.defaults.color = css("--ink-2");
  Chart.defaults.font.family = css("--font-text");
  const [{ runs, latest }, status] = await Promise.all([api("/api/runs"), api("/api/status")]);
  const g = status.gemini, pill = $("#gemini-pill");
  state.geminiModels = g.configured ? g.models : [];
  pill.className = "status" + (g.configured ? " on" : "");
  pill.lastElementChild.textContent = g.configured ? "Gemini ready" : "Gemini off";
  pill.title = g.configured ? "Root-cause analysis tries, in order:\n" + g.models.map((m, i) => `${i + 1}. ${m}`).join("\n") + "\nthen offline rules"
                            : "Add GEMINI_API_KEY to .env to enable Gemini";
  if (!runs.length) {
    $("#no-runs").hidden = false;
    $$(".view").forEach((v) => (v.hidden = true));
    return;
  }
  const box = $("#run-box");
  if (runs.length > 1) {
    box.innerHTML = `<select id="run-select" aria-label="Training run">${runs.map((r) => `<option value="${r}">Run ${r}${r === latest ? " (latest)" : ""}</option>`).join("")}</select>`;
    $("#run-select").value = latest;
    $("#run-select").addEventListener("change", (e) => loadRun(e.target.value));
  } else {
    box.textContent = `Run ${latest}`;
  }
  await loadRun(latest);
}

async function loadRun(runId) {
  state.run = runId;
  state.data = await api(`/api/run/${runId}`);
  const lb = state.data.leaderboard;
  state.selectedModel = state.fleetModel = state.model = lb[0].model;
  state.mdata = null; state.fleet = null; state.zoom = null; state.metrics = [];
  renderOverview(); renderModels(); setupFleet(); setupExplorer(); resetRCA();
  const active = $(".view.active")?.id.replace("view-", "");
  if (active === "explorer") loadMachine();
  if (active === "fleet") loadFleet();
}

/* ================================================================ overview */
function renderOverview() {
  const { manifest: m, leaderboard: lb, rca_eval: re } = state.data;
  const ds = m.dataset, best = lb[0];
  const nTrue = Object.values(state.data.machines).reduce((a, x) => a + (x.metrics?.true_incidents || 0), 0);
  const nSup = lb.filter((r) => r.kind === "supervised").length;
  $("#ov-intro").textContent = `${lb.length} models (${nSup} supervised, ${lb.length - nSup} unsupervised) on ${ds.n_machines} production servers: ` +
    `${big(ds.test_points)} labelled minutes, ${pct(ds.anomaly_rate, 1)} of them anomalous. Every score comes from servers the model never saw.`;

  const usedRe = re && re.methods[re.used_method];
  const tiles = [
    ["Best model", best.label, `F1 <b>${fmt(best.f1)}</b> ± ${fmt(best.f1_std, 2)}${best.paper_f1 != null ? `, paper F1 <b>${fmt(best.paper_f1)}</b>` : ""}`],
    ["Incidents caught", pct(best.event_recall), `of ${nTrue ? big(nTrue) + " " : ""}real incidents`],
    ["Time to first alarm", delayTxt(best.detection_delay_min), `median, with <b>${fmt(best.false_alarms_per_day, 1)}</b> false alarms per server a day`],
    usedRe ? ["Root cause found", pct(usedRe.hit_rate_100), `of labelled culprit metrics ranked on top, against <b>${pct(re.random_hit_rate_100)}</b> by chance`]
           : ["Labelled minutes", big(ds.test_points), `${pct(ds.anomaly_rate, 1)} anomalous`],
  ];
  $("#kpis").innerHTML = tiles.map(([l, v, n]) => `<div class="kpi"><div class="label">${l}</div>
    <div class="value${/[a-z]{3,}/i.test(String(v)) && !/min$/.test(v) ? " text" : ""}">${esc(v)}</div><div class="note">${n}</div></div>`).join("");

  // findings: four, each with a label
  const f = [];
  f.push(["Most accurate", `<b>${esc(best.label)}</b> has the best F1 (${fmt(best.f1)}) and catches ${pct(best.event_recall)} of incidents.`]);
  const bestPaper = [...lb].filter((r) => r.paper_f1 != null).sort((a, b) => b.paper_f1 - a.paper_f1)[0];
  if (bestPaper) f.push(["Against published work", `Scored the way papers score SMD, our models reach <b>${fmt(Math.min(...lb.map((r) => r.paper_f1)), 2)}–${fmt(bestPaper.paper_f1, 2)}</b> (best: ${esc(bestPaper.label)}). Published deep models such as OmniAnomaly, USAD, TranAD and GDN score 0.42–0.53 under the same protocol.`]);
  const bestEv = [...lb].sort((a, b) => b.event_recall - a.event_recall)[0];
  if (bestEv.model !== best.model) f.push(["Widest coverage", `<b>${esc(bestEv.label)}</b> catches the most incidents (${pct(bestEv.event_recall)}), at the cost of ${fmt(bestEv.false_alarms_per_day, 1)} false alarms a day.`]);
  const quiet = [...lb].sort((a, b) => a.false_alarms_per_day - b.false_alarms_per_day)[0];
  f.push(["Quietest", `<b>${esc(quiet.label)}</b> raises the fewest false alarms (${fmt(quiet.false_alarms_per_day, 1)} per server per day) and catches ${pct(quiet.event_recall)} of incidents.`]);
  const bestUns = lb.find((r) => r.kind === "unsupervised");
  if (bestUns) f.push(["Without labels", `The best unsupervised model, <b>${esc(bestUns.label)}</b>, reaches F1 ${fmt(bestUns.f1)} using no labelled data at all.`]);
  const nns = lb.filter((r) => r.neural);
  if (nns.length && f.length < 5) f.push(["Neural nets", nns.map((r) => `<b>${esc(r.label)}</b> ranks ${r.rank} of ${lb.length}`).join("; ") + "."]);
  $("#findings").innerHTML = f.slice(0, 5).map(([k, t]) => `<li><span class="k">${k}</span>${t}</li>`).join("");

  $$(".kind-legend").forEach((el) => (el.innerHTML = Object.keys(KIND).map((k) => `<span><i class="swatch" style="background:${kindColor(k)}"></i>${KIND[k].label}</span>`).join("")
    + `<span><i class="swatch line" style="background:var(--ink-2)"></i>± 1 std</span>`));
  $("#rank-metric").innerHTML = Object.entries(RANK).map(([k, v]) => `<button data-k="${k}" class="${k === state.rankMetric ? "on" : ""}">${v.label}</button>`).join("");
  $$("#rank-metric button").forEach((b) => b.addEventListener("click", () => { state.rankMetric = b.dataset.k; renderOverview(); }));
  renderRanking();
}

function renderRanking() {
  const key = state.rankMetric, spec = RANK[key];
  const rows = [...state.data.leaderboard].sort((a, b) => (spec.higher ? b[key] - a[key] : a[key] - b[key]));
  $("#rank-sub").textContent = spec.sub;
  const top = Math.max(...rows.map((r) => r[key] + (r[key + "_std"] || 0)));
  const maxV = key === "false_alarms_per_day" ? Math.ceil(top) || 1 : Math.min(1, Math.ceil(top * 10) / 10) || 1;
  const x = (v) => `${(Math.max(0, Math.min(v, maxV)) / maxV) * 100}%`;
  const el = $("#ranking");
  el.innerHTML = `<div class="bars">${rows.map((r, i) => {
    const sd = r[key + "_std"] || 0;
    return `<div class="bar-row" data-i="${i}">
      <div class="bar-name" title="${esc(r.label)}">${esc(r.label)} ${nnBadge(r)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${x(r[key])};background:${kindColor(r.kind)}"></div>
        ${sd ? `<div class="whisker" style="left:${x(r[key] - sd)};width:calc(${x(r[key] + sd)} - ${x(r[key] - sd)})"></div>` : ""}</div>
      <div class="bar-val">${spec.show(r[key])} <small>±${spec.sd(sd)}</small></div></div>`;
  }).join("")}</div>
    <div class="bar-axis"><span></span><div class="ticks">${[0, maxV / 2, maxV].map((t) => `<span>${key === "event_recall" ? pct(t) : key === "false_alarms_per_day" ? t.toFixed(1) : t.toFixed(2)}</span>`).join("")}</div><span></span></div>`;
  $$(".bar-row", el).forEach((row) => {
    const r = rows[+row.dataset.i];
    row.addEventListener("mousemove", (ev) => showTip(`<b>${esc(r.label)}</b><span class="muted">${KIND[r.kind].label}${r.neural ? " neural network" : ""}</span><br>
      F1 ${fmt(r.f1)} (paper-style ${fmt(r.paper_f1)}) · precision ${fmt(r.precision, 2)} · recall ${fmt(r.recall, 2)}<br>
      Incidents caught ${pct(r.event_recall)} · first alarm ${delayTxt(r.detection_delay_min)}<br>
      False alarms ${fmt(r.false_alarms_per_day, 1)} per day<br><span class="muted">Click for details</span>`, ev));
    row.addEventListener("mouseleave", hideTip);
    row.addEventListener("click", () => { hideTip(); state.selectedModel = r.model; renderModels(); go("models"); });
  });
}

/* ================================================================ models */
const FILTERS = [["all", "All"], ["supervised", "Supervised"], ["unsupervised", "Unsupervised"], ["neural", "Neural nets"]];
function filteredLB() {
  const lb = state.data.leaderboard, f = state.modelFilter;
  if (f === "all") return lb;
  if (f === "neural") return lb.filter((r) => r.neural);
  return lb.filter((r) => r.kind === f);
}

$("#more-cols").addEventListener("change", (e) => { state.moreCols = e.target.checked; renderModels(); });

function renderModels() {
  $("#model-filter").innerHTML = FILTERS.map(([k, l]) => `<button data-f="${k}" class="${state.modelFilter === k ? "on" : ""}">${l}</button>`).join("");
  $$("#model-filter button").forEach((b) => b.addEventListener("click", () => { state.modelFilter = b.dataset.f; renderModels(); }));
  const rows = filteredLB();
  if (!rows.some((r) => r.model === state.selectedModel)) state.selectedModel = rows[0].model;
  const pm = (k) => (r) => `${fmt(r[k])}<span class="pm">±${fmt(r[k + "_std"], 2)}</span>`;
  const cols = [
    { key: "rank", label: "#", num: true },
    { key: "label", label: "Model", fmt: (r) => `<span class="model-cell">${esc(r.label)} ${nnBadge(r)}</span>` },
    { key: "kind", label: "Type", fmt: (r) => kindTag(r.kind) },
    { key: "f1", label: "F1", num: true, best: "max", fmt: pm("f1"), title: "Strict F1 per minute, threshold tuned without test labels, on unseen servers" },
    { key: "paper_f1", label: "Paper F1", num: true, best: "max", fmt: (r) => fmt(r.paper_f1), title: "F1 per server at the best threshold picked from its test labels, averaged over servers: the protocol published SMD papers use. Optimistic; for comparison only." },
    { key: "precision", label: "Precision", num: true, best: "max", fmt: (r) => fmt(r.precision) },
    { key: "recall", label: "Recall", num: true, best: "max", fmt: (r) => fmt(r.recall) },
    { key: "pr_auc", label: "PR-AUC", num: true, best: "max", fmt: (r) => fmt(r.pr_auc) },
    { key: "event_recall", label: "Incidents", num: true, best: "max", fmt: (r) => pct(r.event_recall), title: "Share of real incidents with at least one alarm" },
    { key: "detection_delay_min", label: "First alarm", num: true, best: "min", fmt: (r) => delayTxt(r.detection_delay_min), title: "Median minutes from the start of a caught incident to its first alarm" },
    { key: "false_alarms_per_day", label: "False / day", num: true, best: "min", fmt: (r) => fmt(r.false_alarms_per_day, 1), title: "Alarm incidents per server per day that match no real anomaly" },
  ];
  if (state.moreCols) cols.push(
    { key: "roc_auc", label: "ROC-AUC", num: true, best: "max", fmt: (r) => fmt(r.roc_auc) },
    { key: "pa_f1", label: "PA-F1", num: true, best: "max", fmt: (r) => fmt(r.pa_f1), title: "Point-adjusted F1, the lenient score many papers report. For comparison only." },
    { key: "fit_seconds", label: "Train s", num: true, best: "min", fmt: (r) => fmt(r.fit_seconds, 2), title: "Training time per fold (per server for unsupervised models)" },
    { key: "latency_ms_per_1k", label: "ms / 1k", num: true, best: "min", fmt: (r) => fmt(r.latency_ms_per_1k, 1), title: "Time to score 1,000 minutes" },
  );
  renderTable($("#lb-table"), cols, rows, {
    sort: state.lbSort, selected: (r) => r.model === state.selectedModel,
    onRow: (r) => { state.selectedModel = r.model; renderModels(); },
  });
  renderModelDetail();
  renderHeatmap(rows);
}

function renderModelDetail() {
  const r = state.data.leaderboard.find((x) => x.model === state.selectedModel);
  const color = kindColor(r.kind), curves = state.data.curves || {};
  $("#pr-title").textContent = `Precision–recall · ${r.label}`;
  const ds = [];
  for (const o of state.data.leaderboard) {
    const c = curves[o.model];
    if (!c || o.model === r.model) continue;
    ds.push({ label: o.label, data: c.recall.map((x, i) => ({ x, y: c.precision[i] })), borderColor: "rgba(214,235,228,.12)", borderWidth: 1, pointRadius: 0, order: 3 });
  }
  const c = curves[r.model];
  if (c) {
    ds.push({ label: r.label, data: c.recall.map((x, i) => ({ x, y: c.precision[i] })), borderColor: color, borderWidth: 2.4, pointRadius: 0, order: 1 });
    ds.push({ label: "Tuned threshold", type: "scatter", data: [{ x: c.operating_point.recall, y: c.operating_point.precision }], backgroundColor: color, borderColor: css("--surface"), borderWidth: 2, pointRadius: 7, pointHoverRadius: 8, order: 0 });
  }
  if (charts.pr) charts.pr.destroy();
  charts.pr = new Chart($("#pr-chart"), {
    type: "line", data: { datasets: ds },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
      interaction: { mode: "nearest", intersect: false },
      scales: {
        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "Recall", color: css("--muted") }, grid: { color: css("--grid") }, border: { color: css("--axis") } },
        y: { min: 0, max: 1, title: { display: true, text: "Precision", color: css("--muted") }, grid: { color: css("--grid") }, border: { display: false } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { filter: (i) => i.dataset.label === r.label || i.dataset.label === "Tuned threshold", backgroundColor: css("--raised"), borderColor: css("--line-2"), borderWidth: 1,
          callbacks: { label: (i) => ` ${i.dataset.label}: precision ${i.parsed.y.toFixed(2)} at recall ${i.parsed.x.toFixed(2)}` } },
      },
    },
  });

  const [[tn, fp], [fn, tp]] = r.confusion;
  const shade = (v, tot) => `background:rgba(57,135,229,${0.1 + (tot ? v / tot : 0) * 0.7})`;
  $("#cm-title").textContent = `Confusion matrix · ${r.label}`;
  $("#cm").innerHTML = `<div class="cm">
      <div></div><div class="h">Predicted normal</div><div class="h">Predicted anomaly</div>
      <div class="h">Actually normal</div>
      <div class="cell" style="${shade(tn, tn + fp)}"><b>${big(tn)}</b><span>${pct(tn / (tn + fp), 1)} of normal</span></div>
      <div class="cell" style="${shade(fp, tn + fp)}"><b>${big(fp)}</b><span>false alarms · ${pct(fp / (tn + fp), 1)}</span></div>
      <div class="h">Actually anomalous</div>
      <div class="cell" style="${shade(fn, fn + tp)}"><b>${big(fn)}</b><span>missed · ${pct(fn / (fn + tp), 1)}</span></div>
      <div class="cell" style="${shade(tp, fn + tp)}"><b>${big(tp)}</b><span>caught · ${pct(tp / (fn + tp), 1)}</span></div>
    </div>`;
  const folds = state.data.folds[r.model] || [], groups = state.data.manifest.folds;
  $("#fold-table").innerHTML = `<div class="table-wrap"><table><thead><tr><th>Fold</th><th>Held-out servers</th><th class="num">F1</th><th class="num">Incidents</th><th class="num">False / day</th></tr></thead>
    <tbody>${folds.map((f, i) => `<tr><td>${i + 1}</td><td style="white-space:normal;font-size:12px;color:var(--ink-2)">${groups[i].map(shortM).join(", ")}</td>
      <td class="num">${fmt(f.f1)}</td><td class="num">${pct(f.event_recall)}</td><td class="num">${fmt(f.false_alarms_per_day, 1)}</td></tr>`).join("")}</tbody></table></div>`;
}

function renderHeatmap(rows) {
  const machines = Object.keys(state.data.machines).sort(natural);
  const el = $("#heatmap");
  if (!machines.some((m) => state.data.machines[m].metrics)) { el.innerHTML = `<p class="sub">Per-server results are not available for this run.</p>`; return; }
  el.style.gridTemplateColumns = `180px repeat(${machines.length}, minmax(18px, 1fr))`;
  let html = "";
  rows.forEach((r) => {
    html += `<div class="hl" title="${esc(r.label)}"><i class="swatch" style="background:${kindColor(r.kind)};border-radius:50%"></i>${esc(r.label)}</div>`;
    machines.forEach((m) => {
      const v = state.data.machines[m].metrics?.[r.model]?.f1;
      html += `<div class="hc" data-m="${m}" data-k="${r.model}" style="background:${v == null ? "transparent" : seq(v)}"></div>`;
    });
  });
  html += `<div></div>` + machines.map((m) => `<div class="hx">${shortM(m)}</div>`).join("");
  el.innerHTML = html;
  $("#hm-legend").innerHTML = `F1 0 <span class="ramp" style="background:${rampCss}"></span> 1`;
  $$(".hc", el).forEach((c) => {
    const m = c.dataset.m, k = c.dataset.k, v = state.data.machines[m].metrics[k], lab = state.data.leaderboard.find((x) => x.model === k).label;
    c.addEventListener("mousemove", (ev) => showTip(`<b>${esc(lab)} · ${m}</b>F1 ${fmt(v.f1)} · precision ${fmt(v.precision, 2)} · recall ${fmt(v.recall, 2)}<br>
      Incidents caught ${pct(v.event_recall)} · ${fmt(v.false_alarms_per_day, 1)} false alarms a day<br><span class="muted">Click to open in the Explorer</span>`, ev));
    c.addEventListener("mouseleave", hideTip);
    c.addEventListener("click", () => { hideTip(); openExplorer(m, k); });
  });
}

/* ================================================================ fleet */
function setupFleet() {
  const sel = $("#fleet-model");
  sel.innerHTML = state.data.leaderboard.map((r) => `<option value="${r.model}">${esc(r.label)}</option>`).join("");
  sel.value = state.fleetModel;
  sel.onchange = () => { state.fleetModel = sel.value; loadFleet(); };
  $("#fleet-count").textContent = `All ${Object.keys(state.data.machines).length} servers`;
}

async function loadFleet() {
  const card = $("#fleet-canvas").parentElement;
  card.classList.add("loading");
  try {
    state.fleet = await api(`/api/run/${state.run}/fleet?model=${state.fleetModel}&bucket=60`);
    drawFleet(); renderFleetTable();
  } finally { card.classList.remove("loading"); }
}

const FLEET = { label: 58, cell: 11, lane: 3, gap: 6, axis: 22 };
function drawFleet() {
  const f = state.fleet, cv = $("#fleet-canvas");
  if (!f || !cv.offsetParent) return;
  const W = cv.parentElement.clientWidth - 44, rows = f.rows, pitch = FLEET.cell + FLEET.lane + FLEET.gap;
  const H = rows.length * pitch + FLEET.axis, dpr = devicePixelRatio || 1;
  const nb = Math.max(...rows.map((r) => r.alarm.length));
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + "px";
  const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, W, H);
  const x0 = FLEET.label, bw = (W - x0) / nb;
  ctx.font = `11px ${css("--font-text")}`; ctx.textBaseline = "middle"; ctx.textAlign = "right";
  rows.forEach((r, i) => {
    const y = i * pitch;
    ctx.fillStyle = css("--muted"); ctx.fillText(shortM(r.machine), x0 - 8, y + FLEET.cell / 2);
    ctx.fillStyle = css("--surface-2"); ctx.fillRect(x0, y, bw * r.alarm.length, FLEET.cell);
    r.alarm.forEach((a, b) => { if (a > 0) { ctx.fillStyle = seq(0.25 + 0.75 * a); ctx.fillRect(x0 + b * bw, y, Math.max(bw, 1), FLEET.cell); } });
    ctx.fillStyle = css("--truth-solid");
    r.truth.forEach((t, b) => { if (t > 0) ctx.fillRect(x0 + b * bw, y + FLEET.cell + 1, Math.max(bw, 1.5), FLEET.lane); });
  });
  const per = f.bucket_minutes, days = Math.floor((nb * per) / 1440);
  ctx.textAlign = "center"; ctx.fillStyle = css("--muted");
  const yAx = rows.length * pitch + 10, step = days > 14 ? 5 : 2;
  for (let d = 0; d <= days; d += step) ctx.fillText(`day ${d}`, x0 + ((d * 1440) / per) * bw, yAx);
  cv._geo = { x0, bw, pitch, rows, per };
}
function fleetHit(ev) {
  const g = ev.currentTarget._geo; if (!g) return null;
  const rect = ev.currentTarget.getBoundingClientRect();
  const r = g.rows[Math.floor((ev.clientY - rect.top) / g.pitch)], b = Math.floor((ev.clientX - rect.left - g.x0) / g.bw);
  return r && b >= 0 && b < r.alarm.length ? { g, r, b } : null;
}
$("#fleet-canvas").addEventListener("mousemove", (ev) => {
  const h = fleetHit(ev); if (!h) return hideTip();
  const { g, r, b } = h;
  showTip(`<b>${r.machine}</b>${minuteLabel(b * g.per)} → ${minuteLabel((b + 1) * g.per)}<br>
    In alarm ${pct(r.alarm[b])} of the hour · labelled anomaly ${pct(r.truth[b])}<br><span class="muted">Click to open in the Explorer</span>`, ev);
});
$("#fleet-canvas").addEventListener("mouseleave", hideTip);
$("#fleet-canvas").addEventListener("click", (ev) => {
  const h = fleetHit(ev); if (!h) return;
  hideTip(); openExplorer(h.r.machine, state.fleetModel, [Math.max(0, h.b * h.g.per - 360), (h.b + 1) * h.g.per + 360]);
});

function renderFleetTable() {
  const k = state.fleetModel;
  const rows = Object.entries(state.data.machines).map(([m, v]) => ({
    machine: m, rate: v.anomaly_points / v.test_points, true_incidents: v.metrics?.true_incidents, ...(v.metrics?.[k] || {}),
  }));
  renderTable($("#fleet-table"), [
    { key: "machine", label: "Server", fmt: (r) => `<span class="model-cell">${r.machine}</span>` },
    { key: "rate", label: "Anomalous", num: true, fmt: (r) => pct(r.rate, 1) },
    { key: "true_incidents", label: "Real incidents", num: true },
    { key: "event_recall", label: "Caught", num: true, fmt: (r) => pct(r.event_recall) },
    { key: "false_alarms_per_day", label: "False / day", num: true, fmt: (r) => fmt(r.false_alarms_per_day, 1) },
    { key: "f1", label: "F1", num: true, fmt: (r) => fmt(r.f1) },
  ], rows, { sort: state.fleetSort, onRow: (r) => openExplorer(r.machine, k) });
}

/* ================================================================ explorer */
function setupExplorer() {
  const machines = Object.keys(state.data.machines).sort(natural);
  const ms = $("#ex-machine"), mo = $("#ex-model");
  ms.innerHTML = machines.map((m) => { const x = state.data.machines[m]; return `<option value="${m}">${m} · ${pct(x.anomaly_points / x.test_points, 1)} anomalous</option>`; }).join("");
  mo.innerHTML = state.data.leaderboard.map((r) => `<option value="${r.model}">${esc(r.label)}</option>`).join("");
  state.machine = state.machine && machines.includes(state.machine) ? state.machine : machines[0];
  ms.value = state.machine; mo.value = state.model;
  ms.onchange = () => { state.machine = ms.value; state.zoom = null; state.metrics = []; resetRCA(); loadMachine(); };
  mo.onchange = () => { state.model = mo.value; loadMachine(); };
  $("#show-agree").onchange = (e) => { state.showAgree = e.target.checked; $("#agree-canvas").hidden = !state.showAgree; drawAgreement(); };
}

function openExplorer(machine, model, zoom = null) {
  state.machine = machine; state.model = model; state.zoom = zoom; state.metrics = [];
  $("#ex-machine").value = machine; $("#ex-model").value = model;
  resetRCA(); state.mdata = null; go("explorer");
}

async function loadMachine() {
  const q = new URLSearchParams({ model: state.model });
  if (state.zoom) { q.set("start", state.zoom[0]); q.set("end", state.zoom[1]); }
  if (state.metrics.length) q.set("metrics", state.metrics.join(","));
  const left = $("#ex-left");
  left.classList.add("loading");
  try {
    const d = await api(`/api/run/${state.run}/machine/${state.machine}?${q}`);
    state.mdata = d;
    if (!state.metrics.length) state.metrics = Object.keys(d.metrics);
    const s = d.stats;
    $("#ex-stats").innerHTML = `<span><b>${s.caught_incidents}/${s.true_incidents}</b> real incidents caught</span>
      <span><b>${s.detected_incidents}</b> alarm incidents</span><span>F1 <b>${fmt(s.f1)}</b></span>`;
    renderScoreChart(d); renderMetricChips(d); renderMetricChart(d); renderIncidents(d); renderZoomNote(d);
    requestAnimationFrame(drawAgreement);
  } finally { left.classList.remove("loading"); }
}

/* Chart.js plugin: shaded labelled-anomaly bands */
const bandsPlugin = {
  id: "bands",
  beforeDatasetsDraw(chart, _a, opts) {
    const { ctx, chartArea: a, scales: { x } } = chart;
    ctx.save(); ctx.fillStyle = css("--truth");
    for (const s of opts.segments || []) {
      if (s.end < x.min || s.start > x.max) continue;
      const x0 = Math.max(a.left, x.getPixelForValue(s.start)), x1 = Math.min(a.right, x.getPixelForValue(s.end + 1));
      ctx.fillRect(x0, a.top, Math.max(1.5, x1 - x0), a.bottom - a.top);
    }
    ctx.restore();
  },
};

function timeOptions(d, yOpts) {
  return {
    responsive: true, maintainAspectRatio: false, animation: false, parsing: false, normalized: true,
    interaction: { mode: "index", intersect: false },
    layout: { padding: { right: 8 } },
    scales: {
      x: { type: "linear", min: d.start, max: d.end, grid: { color: css("--grid"), drawTicks: false }, border: { color: css("--axis") },
           ticks: { color: css("--muted"), maxTicksLimit: 7, callback: (v) => shortTime(v), padding: 6 } },
      y: { grid: { color: css("--grid") }, border: { display: false }, ticks: { color: css("--muted"), maxTicksLimit: 5 },
           afterFit: (axis) => { axis.width = labelW(); }, ...yOpts },
    },
    plugins: {
      legend: { display: false },
      bands: { segments: d.truth_segments },
      tooltip: { backgroundColor: css("--raised"), titleColor: css("--ink"), bodyColor: css("--ink-2"), borderColor: css("--line-2"), borderWidth: 1,
        padding: 10, usePointStyle: true, boxPadding: 4, callbacks: { title: (it) => minuteLabel(it[0].parsed.x) } },
    },
  };
}

function drawChart(id, config) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart($("#" + id), { ...config, plugins: [bandsPlugin] });
}

function renderScoreChart(d) {
  const thr = d.threshold, CAP = 3;
  const pts = d.score.map((s, i) => { const r = thr > 0 ? s / thr : s; return { x: d.x[i], y: Math.max(0, Math.min(CAP, r)), r }; });
  const o = timeOptions(d, { min: 0, max: CAP, ticks: { color: css("--muted"), stepSize: 1, callback: (v) => v + "×" } });
  o.plugins.tooltip.filter = (c) => c.datasetIndex === 0;
  o.plugins.tooltip.callbacks.label = (c) => ` ${d.model_label}: ${c.raw.r.toFixed(2)}× threshold${c.raw.r >= 1 ? " · alarm" : ""}`;
  o.onClick = (e, _els, chart) => {
    const x = chart.scales.x.getValueForPixel(e.x);
    if (x == null || (state.zoom && state.zoom[1] - state.zoom[0] <= 400)) return;
    state.zoom = [Math.max(0, Math.round(x - 180)), Math.min(d.length - 1, Math.round(x + 180))];
    loadMachine();
  };
  const kc = kindColor(state.data.leaderboard.find((r) => r.model === d.model)?.kind);
  $("#score-swatch").style.background = kc;
  drawChart("score-chart", {
    type: "line",
    data: { datasets: [
      { data: pts, borderColor: kc, backgroundColor: kc + "26", fill: "origin", borderWidth: 1.4, pointRadius: 0, pointHoverRadius: 4, tension: 0 },
      { data: [{ x: d.start, y: 1 }, { x: d.end, y: 1 }], borderColor: css("--ink-2"), borderDash: [5, 4], borderWidth: 1.2, pointRadius: 0 },
    ] },
    options: o,
  });
}

/* per-model alarm strip, x-aligned with the score chart */
function drawAgreement() {
  const d = state.mdata, cv = $("#agree-canvas"), ch = charts["score-chart"];
  if (!d || !ch || !state.showAgree || !cv.offsetParent) return;
  const rows = d.agreement, rowH = 9, gap = 4, W = cv.parentElement.clientWidth - 44;
  const H = rows.length * (rowH + gap), dpr = devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + "px";
  const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, W, H);
  const a = ch.chartArea, xs = ch.scales.x, px = (v) => xs.getPixelForValue(v);
  ctx.fillStyle = css("--truth");
  for (const s of d.truth_segments) {
    if (s.end < d.start || s.start > d.end) continue;
    const x0 = Math.max(a.left, px(s.start)), x1 = Math.min(a.right, px(s.end + 1));
    ctx.fillRect(x0, 0, Math.max(1.5, x1 - x0), H);
  }
  ctx.font = `11px ${css("--font-text")}`; ctx.textBaseline = "middle"; ctx.textAlign = "right";
  rows.forEach((r, i) => {
    const y = i * (rowH + gap), sel = r.model === d.model;
    ctx.fillStyle = sel ? css("--ink") : css("--muted");
    let lab = r.label; while (ctx.measureText(lab).width > a.left - 12 && lab.length > 4) lab = lab.slice(0, -2);
    ctx.fillText(lab === r.label ? lab : lab + "…", a.left - 8, y + rowH / 2);
    ctx.fillStyle = css("--surface-2"); ctx.fillRect(a.left, y, a.right - a.left, rowH);
    ctx.fillStyle = kindColor(r.kind); ctx.globalAlpha = sel ? 1 : 0.7;
    r.alarm.forEach((v, j) => { if (v) { const x0 = px(d.x[j]), x1 = px(d.x[j] + d.bucket); ctx.fillRect(x0, y, Math.max(1.2, x1 - x0), rowH); } });
    ctx.globalAlpha = 1;
  });
  cv._geo = { rows, rowH, gap, a, xs };
}
$("#agree-canvas").addEventListener("mousemove", (ev) => {
  const g = ev.currentTarget._geo; if (!g) return;
  const rect = ev.currentTarget.getBoundingClientRect(), x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  const r = g.rows[Math.floor(y / (g.rowH + g.gap))];
  if (!r || x < g.a.left || x > g.a.right) return hideTip();
  const t = Math.round(g.xs.getValueForPixel(x)), d = state.mdata, j = Math.min(d.x.length - 1, Math.max(0, Math.floor((t - d.start) / d.bucket)));
  showTip(`<b>${esc(r.label)}</b>${minuteLabel(t)} · ${r.alarm[j] ? "in alarm" : "quiet"}`, ev);
});
$("#agree-canvas").addEventListener("mouseleave", hideTip);

const SERIES = ["--sup", "--uns", "--m3"];
function renderMetricChips(d) {
  $("#metric-chips").innerHTML = d.metric_rank.slice(0, 8).map((m) => {
    const i = state.metrics.indexOf(m);
    return `<button class="chip${i >= 0 ? " on" : ""}" data-m="${m}">${i >= 0 ? `<i style="background:${css(SERIES[i])}"></i>` : ""}${m}</button>`;
  }).join("");
  $$("#metric-chips .chip").forEach((b) => b.addEventListener("click", () => {
    const m = b.dataset.m, i = state.metrics.indexOf(m);
    if (i >= 0) { if (state.metrics.length > 1) state.metrics.splice(i, 1); }
    else { if (state.metrics.length >= 3) state.metrics.shift(); state.metrics.push(m); }
    loadMachine();
  }));
}

function renderMetricChart(d) {
  const names = state.metrics.filter((m) => d.metrics[m]);
  const o = timeOptions(d, { min: 0, suggestedMax: 1 });
  o.plugins.tooltip.callbacks.label = (c) => ` ${c.dataset.label}: ${c.parsed.y.toFixed(3)}`;
  drawChart("metric-chart", {
    type: "line",
    data: { datasets: names.map((m) => ({ label: m, data: d.metrics[m].map((v, j) => ({ x: d.x[j], y: v })),
      borderColor: css(SERIES[state.metrics.indexOf(m)]), borderWidth: 1.5, pointRadius: 0, pointHoverRadius: 4 })) },
    options: o,
  });
}

function renderIncidents(d) {
  const el = $("#inc-table"), n = d.agreement.length;
  if (!d.incidents.length) { el.innerHTML = `<tbody><tr><td style="color:var(--muted)">This model raised no alarms on this server.</td></tr></tbody>`; return; }
  renderTable(el, [
    { key: "start", label: "When", fmt: (r) => minuteLabel(r.start) },
    { key: "duration", label: "Length", num: true, fmt: (r) => `${r.duration} min` },
    { key: "agreeing", label: "Models", num: true, fmt: (r) => `${r.agreeing}/${n}`, title: "How many of the models raise an alarm in this window" },
    { key: "truth_overlap", label: "Real anomaly?", fmt: (r) => (r.truth_overlap > 0 ? `<span class="ok-tag">Yes</span>` : `<span class="bad-tag">False alarm</span>`) },
    { key: "act", label: "", sort: false, fmt: () => `<button class="btn small primary">Analyse</button>` },
  ], d.incidents, {
    sort: state.incSort,
    onRow: (r) => { state.zoom = zoomFor(r); loadMachine(); },
    after: (tbl, sorted) => $$("tbody tr", tbl).forEach((tr) => tr.querySelector("button")?.addEventListener("click", (e) => analyseIncident(sorted[+tr.dataset.i], e.currentTarget))),
  });
}

const zoomFor = (inc) => { const pad = Math.max(60, inc.duration * 2); return [Math.max(0, inc.start - pad), Math.min(state.mdata.length - 1, inc.end + pad)]; };

function renderZoomNote(d) {
  const z = $("#zoom-note");
  if (!state.zoom) { z.hidden = true; return; }
  z.hidden = false;
  z.innerHTML = `${shortTime(d.start)} → ${shortTime(d.end)} <button class="btn small">Show all</button>`;
  z.querySelector("button").onclick = () => { state.zoom = null; loadMachine(); };
}

/* ================================================================ root-cause analysis */
function resetRCA() {
  const re = state.data?.rca_eval, used = re && re.methods[re.used_method];
  $("#rca").innerHTML = `<h2>Root-cause analysis</h2>
    <div class="rca-empty"><p class="sub" style="margin:4px 0 0">Press <b>Analyse</b> on an incident. The analysis:</p>
      <ol><li>ranks the 38 metrics by how far they moved from this server's normal level,</li>
      <li>finds which metric moved first and which ones move together,</li>
      <li>checks how many models agree and whether it happened before,</li>
      <li>asks Gemini to write it up, with offline rules as a fallback.</li></ol></div>
    ${used ? `<div class="rca-trust"><b>${pct(used.hit_rate_100)}</b>of the culprit metrics named in SMD's labels are ranked at the top by step 1, across ${re.incidents} labelled incidents. A random ranking finds ${pct(re.random_hit_rate_100)}.</div>` : ""}`;
}

async function analyseIncident(inc, btn, refresh = false) {
  state.zoom = zoomFor(inc);
  if (btn) btn.disabled = true;
  const chain = state.geminiModels.length ? `Asking ${state.geminiModels.map(esc).join(", then ")}. A busy model can take up to 40 s.` : "Gemini is off, so the offline rules will write the report.";
  $("#rca").innerHTML = `<h2>Root-cause analysis</h2><p class="sub">${state.machine} · ${minuteLabel(inc.start)} · ${inc.duration} min<br>${chain}</p>${'<div class="skeleton"></div>'.repeat(7)}`;
  loadMachine();
  try {
    const res = await api("/api/rca", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run: state.run, machine: state.machine, model: state.model, start: inc.start, end: inc.end, refresh }) });
    renderRCA(res, inc);
  } catch (e) {
    $("#rca").innerHTML = `<h2>Root-cause analysis</h2><div class="notice">The analysis failed: ${esc(e.message)}. Try again, or pick another incident.</div>`;
  } finally { if (btn) btn.disabled = false; }
}

function shortErr(e) {
  const s = String(e || "");
  if (s.includes("504") || /timeout|deadline/i.test(s)) return "timed out";
  if (s.includes("503") || /high demand|unavailable/i.test(s)) return "busy";
  if (s.includes("429")) return "rate limited";
  if (s.includes("404")) return "not available";
  if (s.includes("401") || s.includes("403") || /api key/i.test(s)) return "key rejected";
  return s.slice(0, 60);
}

function renderRCA(res, inc) {
  const r = res.report, ev = res.evidence, gt = res.ground_truth, gtSet = new Set(gt.interpretation_metrics);
  const src = res.source === "gemini" ? `<span class="pill on"><span class="dot"></span>${esc(res.model)}</span>` : `<span class="pill warn"><span class="dot"></span>Offline rules</span>`;
  const peak = (t) => t.peak_abs_z ?? Math.abs(t.z_score);
  const maxZ = Math.max(3, ...ev.top_metrics.map(peak));
  const zbars = ev.top_metrics.map((t) => {
    const up = t.direction ? t.direction === "up" : t.z_score >= 0, w = (Math.min(peak(t), maxZ) / maxZ) * 50;
    return `<div class="zrow" title="normal ${t.baseline_mean} → incident average ${t.incident_mean}, peak ${t.incident_peak}">
      <div class="zn">${t.metric}${gtSet.has(t.metric) ? '<span class="gt" title="Named in SMD ground truth">✓</span>' : ""}</div>
      <div class="ztrack"><div class="zfill" style="left:${up ? 50 : 50 - w}%;width:${w}%;background:${up ? css("--up") : css("--down")}"></div></div>
      <div class="zv">${peak(t).toFixed(1)} SD${t.onset_minute != null ? ` <small>· ${t.onset_minute >= 0 ? "+" : ""}${t.onset_minute} min</small>` : ""}</div></div>`;
  }).join("");
  const attempts = (res.attempts || []).map((a) => `<li title="${esc(a.error || "")}"><span class="${a.ok ? "s-ok" : "s-bad"}">${a.ok ? "✓" : "✗"}</span>
      <span class="m">${esc(a.model)}</span><span>${a.seconds}s</span><span class="why">${a.ok ? "answered" : esc(shortErr(a.error))}</span></li>`).join("")
    + (res.source === "offline" ? `<li><span class="s-off">↓</span><span class="m">offline rules</span><span class="why">used as the fallback</span></li>` : "");
  const hits = gt.interpretation_metrics.filter((m) => ev.top_metrics.some((t) => t.metric === m));
  const similar = ev.similar_incidents.map((s, i) => `<li><button class="link" data-sim="${i}">${minuteLabel(s.start_minute)}</button>, ${s.duration_minutes} min, ${pct(s.similarity)} alike (${s.when})</li>`).join("");
  const groups = ev.co_moving_groups.map((g) => `<li>${g.join(", ")}</li>`).join("");

  $("#rca").innerHTML = `<h2>Root-cause analysis</h2>
    <p class="sub">${esc(ev.machine)} · ${minuteLabel(ev.start_minute)} · ${ev.duration_minutes} min · ${ev.agreement} models alarm</p>
    ${res.error ? `<div class="notice">${esc(res.error)} The report below comes from the offline rules.</div>` : ""}
    <div class="rca-head"><span class="pill sev ${esc(r.severity)}"><i></i>${esc(r.severity)}</span>${src}
      <span class="pill">${esc(r.confidence)} confidence</span>${res.cached ? `<span class="pill" title="Saved ${esc(res.created_at)}">Saved</span>` : ""}
      <span class="rca-actions"><button class="btn small ghost" id="rca-rerun" title="Ask Gemini again">Re-run</button><button class="btn small ghost" id="rca-export">Export</button></span></div>
    <h3>${esc(r.headline)}</h3>
    <p class="summary">${md(r.summary || "")}</p>
    <h4>Likely cause</h4><div>${md(r.likely_cause)}</div>
    <h4>Metrics that moved most</h4><div class="zbars">${zbars}</div>
    <p class="sub" style="margin:8px 0 0">Peak distance from normal in standard deviations · red = above normal, blue = below · minutes = when it first passed 3 SD, relative to the alarm · ✓ = named in SMD's labels</p>
    ${r.timeline?.length ? `<details open><summary>How it unfolded</summary><ol class="timeline">${r.timeline.map((x) => `<li>${md(x)}</li>`).join("")}</ol></details>` : ""}
    <details open><summary>What to check next</summary><ul>${r.investigation_steps.map((x) => `<li>${md(x)}</li>`).join("")}</ul></details>
    <details><summary>Evidence and pattern</summary><p style="margin:0 0 8px;color:var(--ink-2)">${md(r.pattern)}</p><ul>${r.evidence.map((x) => `<li>${md(x)}</li>`).join("")}</ul>
      ${groups ? `<h4>Metrics moving together</h4><ul>${groups}</ul>` : ""}
      ${similar ? `<h4>Similar incidents on this server</h4><ul>${similar}</ul>` : ""}</details>
    ${attempts ? `<details><summary>Which Gemini model answered</summary><ul class="attempts">${attempts}</ul></details>` : ""}
    <details><summary>What Gemini was sent</summary><pre class="code">${esc(JSON.stringify(ev, null, 2))}</pre></details>
    <div class="truth-line"><b>Check against SMD labels</b> (never sent to Gemini): ${gt.labelled_anomaly_share > 0
      ? `${pct(gt.labelled_anomaly_share)} of this window is a real anomaly. The labels name ${gt.interpretation_metrics.length} culprit metrics, and ${hits.length} of the top ${ev.top_metrics.length} above are among them.`
      : "no real anomaly in this window, so this is probably a false alarm."}</div>
    <p class="caveat">${esc(r.caveats)}</p>`;

  $("#rca-rerun").onclick = () => analyseIncident(inc, null, true);
  $("#rca-export").onclick = () => exportRCA(res);
  $$("[data-sim]", $("#rca")).forEach((b) => b.addEventListener("click", () => {
    const s = ev.similar_incidents[+b.dataset.sim];
    state.zoom = zoomFor({ start: s.start_minute, end: s.start_minute + s.duration_minutes - 1, duration: s.duration_minutes });
    loadMachine();
  }));
}

function exportRCA(res) {
  const r = res.report, ev = res.evidence;
  const lines = [
    `# Root-cause analysis: ${ev.machine}, ${minuteLabel(ev.start_minute)} (${ev.duration_minutes} min)`, "",
    `**Severity:** ${r.severity} · **Confidence:** ${r.confidence} · **Written by:** ${res.source === "gemini" ? res.model : "offline rules"} · **Detector:** ${ev.detector} (${ev.agreement} models alarm)`, "",
    `## ${r.headline}`, "", r.summary || "", "",
    "## Likely cause", r.likely_cause, "",
    "## How it unfolded", ...(r.timeline || []).map((x, i) => `${i + 1}. ${x}`), "",
    "## Metrics that moved most", "| Metric | Peak (SD) | Average (SD) | Normal | Incident | First past 3 SD (min) |", "|---|---|---|---|---|---|",
    ...ev.top_metrics.map((t) => `| ${t.metric} | ${t.peak_abs_z ?? "–"} | ${t.z_score} | ${t.baseline_mean} | ${t.incident_mean} | ${t.onset_minute ?? "–"} |`), "",
    "## Evidence", ...r.evidence.map((x) => `- ${x}`), "",
    "## What to check next", ...r.investigation_steps.map((x) => `- ${x}`), "",
    `_${r.caveats}_`, "",
  ];
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/markdown" })), download: `rca_${ev.machine}_${ev.start_minute}.md` });
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

boot().catch((e) => { $("main").insertAdjacentHTML("afterbegin", `<div class="notice">The dashboard could not load: ${esc(e.message)}. Is the server running?</div>`); });
