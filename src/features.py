"""Window features. One function, used by training, evaluation and the
dashboard, so there is no training/serving skew. Every feature is causal:
it only looks at the current and past minutes."""
from __future__ import annotations

import numpy as np
import pandas as pd

EXTRAS = ("diff", "long_dev")


def window_features(x: np.ndarray, window: int, extras: tuple | list = (), long_window: int = 60) -> np.ndarray:
    """Per minute and metric:
      raw value, rolling mean and rolling std over the last `window` minutes,
      plus optional extras:
        diff      change since the previous minute (sudden jumps)
        long_dev  value minus the mean of the last `long_window` minutes (slow drift)
    Output shape is (T, k * n_metrics), float32."""
    unknown = set(extras) - set(EXTRAS)
    if unknown:
        raise ValueError(f"Unknown feature extras {sorted(unknown)}; choose from {EXTRAS}")
    df = pd.DataFrame(x)
    roll = df.rolling(window, min_periods=1)
    parts = [x.astype(np.float32, copy=False),
             roll.mean().to_numpy(dtype=np.float32),
             roll.std(ddof=0).fillna(0.0).to_numpy(dtype=np.float32)]
    if "diff" in extras:
        parts.append(df.diff().fillna(0.0).to_numpy(dtype=np.float32))
    if "long_dev" in extras:
        parts.append((df - df.rolling(long_window, min_periods=1).mean()).to_numpy(dtype=np.float32))
    return np.hstack(parts)


def feature_names(metric_names: list[str], extras: tuple | list = ()) -> list[str]:
    names = list(metric_names) + [f"{m}_mean" for m in metric_names] + [f"{m}_std" for m in metric_names]
    if "diff" in extras:
        names += [f"{m}_diff" for m in metric_names]
    if "long_dev" in extras:
        names += [f"{m}_longdev" for m in metric_names]
    return names


def features_from_config(x: np.ndarray, cfg: dict) -> np.ndarray:
    f = cfg["features"]
    return window_features(x, f["window"], tuple(f.get("extras", ())), f.get("long_window", 60))
