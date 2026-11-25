"""Fast tests on synthetic data: no download needed (runs in CI)."""
import copy

import numpy as np
import pytest

from src.config import load_config
from src.data import Machine, segments, N_METRICS
from src.evaluate import cross_validate, best_f1_threshold, event_recall
from src.features import window_features
from src.models import REGISTRY, make_detector, enabled_models
from src.rca import analyse, build_evidence, detect_incidents


def fake_machine(name, seed, n=600):
    rng = np.random.default_rng(seed)
    train = rng.normal(0.3, 0.05, (n, N_METRICS)).clip(0, 1).astype(np.float32)
    test = rng.normal(0.3, 0.05, (n, N_METRICS)).clip(0, 1).astype(np.float32)
    labels = np.zeros(n, dtype=np.int8)
    for start in (100, 350):
        test[start:start + 25, :5] += 0.5          # anomaly: metrics m01..m05 jump
        labels[start:start + 25] = 1
    return Machine(name, train, test.clip(0, 1), labels, [(100, 124, [0, 1, 2, 3, 4])])


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config())
    c["evaluation"].update(n_folds=3, train_stride=2, max_fit_rows=800)
    for k in c["models"]:
        c["models"][k]["max_fit_rows"] = 800
    c["models"]["random_forest"]["params"]["n_estimators"] = 20
    c["models"]["xgboost"]["params"]["n_estimators"] = 20
    c["models"]["isolation_forest"]["params"]["n_estimators"] = 50
    return c


@pytest.fixture
def machines():
    return {f"machine-1-{i}": fake_machine(f"machine-1-{i}", i) for i in range(1, 7)}


def test_window_features_shape_and_causal():
    x = np.arange(40, dtype=np.float32).reshape(20, 2)
    f = window_features(x, 5)
    assert f.shape == (20, 6)
    x2 = x.copy(); x2[15:] = 999                # changing the future...
    assert np.allclose(window_features(x2, 5)[:15], f[:15])  # ...never changes the past


def test_segments_merge_gaps():
    m = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1], dtype=bool)
    assert segments(m) == [(1, 2), (5, 5), (10, 10)]
    assert segments(m, gap=2) == [(1, 5), (10, 10)]


def test_threshold_and_event_recall():
    y = np.array([0, 0, 1, 1, 0, 0])
    s = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.1])
    assert 0.3 < best_f1_threshold(y, s) <= 0.8
    assert event_recall(y, np.array([0, 0, 0, 1, 0, 0])) == 1.0


def test_config_lists_ten_ml_models_including_neural_nets():
    keys = enabled_models(load_config())
    assert len(keys) == 10
    assert all(REGISTRY[k].kind in ("supervised", "unsupervised") for k in keys)
    assert sum(REGISTRY[k].neural for k in keys) == 2
    assert "lightgbm" in keys and "pca" not in keys


def test_point_adjust_marks_whole_incident():
    from src.evaluate import point_adjust
    y = np.array([0, 1, 1, 1, 0, 1, 1, 0])
    pred = np.array([0, 0, 1, 0, 0, 0, 0, 0])
    assert point_adjust(y, pred).tolist() == [0, 1, 1, 1, 0, 0, 0, 0]


def test_detection_delay_and_false_alarms():
    from src.evaluate import detection_delays, false_alarm_incidents
    y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0], dtype=np.int8)
    pred = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8)
    assert detection_delays(y, pred) == [2]                  # caught 2 min after start; 2nd incident missed
    assert false_alarm_incidents(y, pred, gap=0) == 2        # minutes 0 and 7 touch no anomaly


def test_paper_f1_picks_best_threshold():
    from src.evaluate import paper_f1
    y = np.array([0, 0, 0, 1, 1, 0])
    assert paper_f1(y, np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.1])) == pytest.approx(1.0)
    assert paper_f1(np.zeros(5, dtype=int), np.arange(5.0)) is None


def test_feature_extras_add_columns():
    x = np.random.default_rng(0).random((50, 4)).astype(np.float32)
    assert window_features(x, 5).shape == (50, 12)
    assert window_features(x, 5, ("diff", "long_dev")).shape == (50, 20)
    with pytest.raises(ValueError):
        window_features(x, 5, ("nope",))


def test_quick_config_extends_base():
    q = load_config("config.quick.yaml")
    assert len(q["data"]["machines"]) == 8 and q["seed"] == load_config()["seed"]


@pytest.mark.parametrize("key", list(REGISTRY))
def test_every_model_fits_and_scores(key, cfg):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 12)).astype(np.float32)
    y = (rng.random(300) < 0.2).astype(int)
    X[y == 1] += 2
    det = make_detector(cfg, key)
    if REGISTRY[key].kind == "supervised":
        det.fit(X, y)
    else:
        det.fit(X[y == 0])                          # unsupervised models only see normal data
    s = det.score(X)
    assert s.shape == (300,) and np.isfinite(s).all()
    assert s[y == 1].mean() > s[y == 0].mean()     # anomalies score higher


def test_cross_validation_is_leak_free_and_reproducible(cfg, machines):
    keys = ["logistic_regression", "isolation_forest"]
    r1 = cross_validate(cfg, machines, keys, log=lambda *_: None)
    r2 = cross_validate(cfg, machines, keys, log=lambda *_: None)
    flat = [m for fold in r1["folds"] for m in fold]
    assert sorted(flat) == sorted(machines)                 # every machine held out exactly once
    assert len(flat) == len(set(flat))
    for a, b in zip(r1["summary"], r2["summary"]):          # same seed -> same numbers
        assert a["model"] == b["model"] and a["f1"] == pytest.approx(b["f1"])
    assert r1["summary"][0]["pr_auc"] > 0.5                  # easy synthetic anomalies are found
    assert set(r1["per_fold"]) == set(keys)
    assert set(r1["machine_metrics"]) == set(machines)
    assert set(r1["curves"]) == set(keys)
    assert r1["summary"][0]["false_alarms_per_day"] >= 0
    assert all(0 < r["paper_f1"] <= 1 for r in r1["summary"])


def test_crossfit_calibration_runs_and_models_are_independent(cfg, machines):
    cfg["evaluation"]["calibration"] = "crossfit"
    both = cross_validate(cfg, machines, ["logistic_regression", "isolation_forest"], log=lambda *_: None)
    alone = cross_validate(cfg, machines, ["logistic_regression"], log=lambda *_: None)
    f1 = lambda r, k: next(x["f1"] for x in r["summary"] if x["model"] == k)
    # dropping a model must not change another model's result
    assert f1(both, "logistic_regression") == pytest.approx(f1(alone, "logistic_regression"))


def test_localisation_eval_beats_random(machines):
    from src.rca import localisation_eval
    res = localisation_eval(machines)
    assert res["incidents"] == len(machines)                 # one interpretation label per fake machine
    assert res["methods"]["max_abs_z"]["hit_rate_100"] > res["random_hit_rate_100"]


def _evidence(machines):
    m = machines["machine-1-1"]
    s = {"a": m.labels.astype(np.float32)}
    ev = build_evidence(m, s, {"a": 0.5}, {"a": "A"}, "a", 100, 124)
    ev.pop("_ground_truth")
    return ev


def test_rca_tries_models_in_order_then_succeeds(cfg, machines, monkeypatch):
    from src.rca import RCAReport, offline_report
    monkeypatch.setenv("GEMINI_MODELS", "first,second,third")
    ev = _evidence(machines)
    tried = []

    def caller(model):
        tried.append(model)
        if model != "third":
            raise RuntimeError("404 model not found")
        return offline_report(ev)

    out = analyse(cfg, ev, caller=caller)
    assert tried == ["first", "second", "third"]
    assert out["source"] == "gemini" and out["model"] == "third"
    assert [a["ok"] for a in out["attempts"]] == [False, False, True]


def test_rca_falls_back_to_offline_when_all_models_fail(cfg, machines, monkeypatch):
    monkeypatch.setenv("GEMINI_MODELS", "a,b,c")
    out = analyse(cfg, _evidence(machines), caller=lambda m: (_ for _ in ()).throw(TimeoutError("slow")))
    assert out["source"] == "offline" and len(out["attempts"]) == 3
    assert all(not a["ok"] for a in out["attempts"])


def test_config_chain_has_three_models():
    assert len(load_config()["rca"]["models"]) >= 3


def test_rca_offline_without_key(cfg, machines, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    m = machines["machine-1-1"]
    scores = {"a": m.labels.astype(np.float32), "b": m.labels.astype(np.float32)}
    ev = build_evidence(m, scores, {"a": 0.5, "b": 0.5}, {"a": "A", "b": "B"}, "a", 100, 124)
    assert ev["top_metrics"][0]["metric"] in {"m01", "m02", "m03", "m04", "m05"}
    assert "_ground_truth" in ev
    assert ev["first_movers"] and ev["shape"]
    assert ev["similar_incidents"] and ev["similar_incidents"][0]["start_minute"] == 350  # the other injected anomaly
    assert "_ground_truth" not in __import__("src.rca", fromlist=["llm_view"]).llm_view(ev)
    out = analyse(cfg, ev)
    assert out["source"] == "offline" and "GEMINI_API_KEY" in out["error"]
    assert out["report"]["severity"] in {"Low", "Medium", "High", "Critical"}
    assert detect_incidents(scores["a"], 0.5, gap=5)[0]["start"] == 100
