"""Model registry. Every model is wrapped as a Detector with the same two
methods, fit(X, y) and score(X) -> anomaly score (higher = more anomalous).

Adding a model = adding one entry to REGISTRY and one block in config.yaml.

Kinds:
  supervised    learns from labelled anomalies on *other* machines
  unsupervised  learns one machine's normal behaviour, needs no labels
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier

@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    kind: str                       # "supervised" | "unsupervised"
    family: str                     # "linear" | "trees" | "neighbours" | "kernel" | "neural"
    build: Callable[[dict, int], object]
    score: Callable[[object, np.ndarray], np.ndarray]
    neural: bool = False


# ---------------------------------------------------------------- custom detectors
class AutoencoderDetector:
    """Neural autoencoder (MLP 114 -> 64 -> 16 -> 64 -> 114). It learns to
    rebuild normal minutes; minutes it cannot rebuild well are anomalous.
    Anomaly score = mean squared reconstruction error."""

    def __init__(self, hidden=(64, 16, 64), alpha=1e-4, max_iter=80, learning_rate_init=1e-3, random_state=0):
        self.scaler = StandardScaler()
        self.net = MLPRegressor(hidden_layer_sizes=tuple(hidden), activation="relu", alpha=alpha,
                                max_iter=max_iter, learning_rate_init=learning_rate_init,
                                early_stopping=True, n_iter_no_change=8, random_state=random_state)

    def fit(self, X):
        Xs = self.scaler.fit_transform(X)
        self.net.fit(Xs, Xs)
        return self

    def anomaly_score(self, X):
        Xs = self.scaler.transform(X)
        return ((self.net.predict(Xs) - Xs) ** 2).mean(axis=1)


def _proba(est, X):
    return est.predict_proba(X)[:, 1]


def _neg_score_samples(est, X):
    return -est.score_samples(X)


def _neg_decision(est, X):
    return -est.decision_function(X)


def _anomaly_score(est, X):
    return est.anomaly_score(X)


REGISTRY: dict[str, ModelSpec] = {
    # ---- supervised
    "logistic_regression": ModelSpec(
        "logistic_regression", "Logistic Regression", "supervised", "linear",
        lambda p, s: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=s, **p)),
        _proba),
    "random_forest": ModelSpec(
        "random_forest", "Random Forest", "supervised", "trees",
        lambda p, s: RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=s, **p),
        _proba),
    "xgboost": ModelSpec(
        "xgboost", "XGBoost", "supervised", "trees",
        lambda p, s: XGBClassifier(tree_method="hist", n_jobs=-1, random_state=s, eval_metric="logloss", **p),
        _proba),
    "lightgbm": ModelSpec(
        "lightgbm", "LightGBM", "supervised", "trees",
        lambda p, s: LGBMClassifier(n_jobs=-1, random_state=s, verbose=-1, **p),
        _proba),
    "knn": ModelSpec(
        "knn", "K-Nearest Neighbours", "supervised", "neighbours",
        lambda p, s: make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=-1, **p)),
        _proba),
    "mlp": ModelSpec(
        "mlp", "Neural Net (MLP)", "supervised", "neural",
        lambda p, s: make_pipeline(StandardScaler(), MLPClassifier(early_stopping=True, random_state=s,
                                                                   **{**p, "hidden_layer_sizes": tuple(p.get("hidden_layer_sizes", (128, 64)))})),
        _proba, neural=True),
    # ---- unsupervised
    "isolation_forest": ModelSpec(
        "isolation_forest", "Isolation Forest", "unsupervised", "trees",
        lambda p, s: IsolationForest(n_jobs=-1, random_state=s, **p),
        _neg_score_samples),
    "one_class_svm": ModelSpec(
        "one_class_svm", "One-Class SVM", "unsupervised", "kernel",
        lambda p, s: make_pipeline(StandardScaler(), OneClassSVM(**p)),
        _neg_decision),
    "local_outlier_factor": ModelSpec(
        "local_outlier_factor", "Local Outlier Factor", "unsupervised", "neighbours",
        lambda p, s: make_pipeline(StandardScaler(), LocalOutlierFactor(novelty=True, n_jobs=-1, **p)),
        _neg_score_samples),
    "autoencoder": ModelSpec(
        "autoencoder", "Autoencoder (NN)", "unsupervised", "neural",
        lambda p, s: AutoencoderDetector(random_state=s, **p),
        _anomaly_score, neural=True),
}


class Detector:
    """Uniform wrapper: supervised models fit on labelled rows, unsupervised
    ones fit on (mostly normal) rows and ignore y."""

    def __init__(self, spec: ModelSpec, params: dict, seed: int):
        self.spec = spec
        self.params = dict(params or {})
        self.seed = seed
        self.est = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Detector":
        params = dict(self.params)
        if self.spec.key in ("xgboost", "lightgbm") and y is not None:
            pos = max(int(y.sum()), 1)
            params.setdefault("scale_pos_weight", (len(y) - pos) / pos)
        self.est = self.spec.build(params, self.seed)
        if self.spec.kind == "supervised":
            if y is None or len(np.unique(y)) < 2:
                raise ValueError(f"{self.spec.label} needs both classes in its training data")
            self.est.fit(X, y)
        else:
            self.est.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.spec.score(self.est, X), dtype=np.float32)


def enabled_models(cfg: dict) -> list[str]:
    keys = [k for k, v in cfg["models"].items() if v.get("enabled", True)]
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        raise ValueError(f"config.yaml lists unknown models: {unknown}. Known: {list(REGISTRY)}")
    return keys


def make_detector(cfg: dict, key: str) -> Detector:
    return Detector(REGISTRY[key], cfg["models"][key].get("params", {}), cfg["seed"])
