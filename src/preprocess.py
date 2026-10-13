"""Validate telemetry and create a leakage-safe, run-grouped holdout split."""

import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


CONFIG = load_config()
PATHS = CONFIG["paths"]
PREP = CONFIG["preprocessing"]
SEED = CONFIG["project"]["random_seed"]


def fail(message: str) -> None:
    print(f"[!] {message}")
    sys.exit(1)


def run_preprocessing() -> None:
    raw_path = PATHS["raw_csv"]
    if not os.path.exists(raw_path):
        fail(f"Raw telemetry file does not exist: {raw_path}")

    df = pd.read_csv(raw_path)
    group_column = PREP["group_column"]
    metadata_columns = list(PREP["metadata_columns"])
    required = {"timestamp", "label", group_column}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        fail(
            f"Missing required grouped-dataset columns: {missing_required}. "
            "Regenerate telemetry with the fixed generate_dataset.py; a random-row fallback is intentionally disabled."
        )

    if df.empty:
        fail("Raw telemetry dataset is empty.")
    if df["label"].isna().any() or df[group_column].isna().any():
        fail("Labels and run IDs must not be missing.")

    feature_columns = [column for column in df.columns if column not in metadata_columns]
    if not feature_columns:
        fail("No candidate feature columns remain after excluding metadata.")

    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)

    all_missing = [column for column in feature_columns if df[column].isna().all()]
    if all_missing:
        fail(f"Features contain no usable values: {all_missing}")

    runs_per_class = df.groupby("label")[group_column].nunique()
    minimum_runs = int(PREP.get("minimum_runs_per_class", 6))
    too_few = runs_per_class[runs_per_class < minimum_runs]
    if not too_few.empty:
        fail(
            "Insufficient independent runs per class. Required at least "
            f"{minimum_runs}; observed {too_few.to_dict()}."
        )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])
    groups = df[group_column].astype(str).to_numpy()
    requested_folds = max(2, round(1.0 / float(PREP.get("test_size", 0.20))))
    split_folds = min(requested_folds, int(runs_per_class.min()))

    splitter = StratifiedGroupKFold(
        n_splits=split_folds,
        shuffle=True,
        random_state=SEED,
    )
    train_index, test_index = next(splitter.split(df[feature_columns], y, groups))
    train_df = df.iloc[train_index].copy()
    test_df = df.iloc[test_index].copy()

    train_groups = set(train_df[group_column].astype(str))
    test_groups = set(test_df[group_column].astype(str))
    overlap = train_groups & test_groups
    if overlap:
        fail(f"Internal split error: {len(overlap)} run IDs occur in both partitions.")

    if PREP.get("drop_constant_features", True):
        usable_features = [
            column for column in feature_columns
            if train_df[column].nunique(dropna=True) > 1
        ]
    else:
        usable_features = feature_columns

    dropped_features = sorted(set(feature_columns) - set(usable_features))
    keep_metadata = [column for column in metadata_columns if column in df.columns and column != "label"]
    output_columns = keep_metadata + usable_features + ["label"]

    train_out = train_df[output_columns].copy()
    test_out = test_df[output_columns].copy()
    train_out["label"] = label_encoder.transform(train_out["label"])
    test_out["label"] = label_encoder.transform(test_out["label"])

    os.makedirs(os.path.dirname(PATHS["train_csv"]) or ".", exist_ok=True)
    os.makedirs(PATHS["artifacts_dir"], exist_ok=True)
    train_out.to_csv(PATHS["train_csv"], index=False)
    test_out.to_csv(PATHS["test_csv"], index=False)
    joblib.dump(label_encoder, PATHS["label_encoder_path"])
    joblib.dump(usable_features, PATHS["feature_names_path"])

    manifest_rows = []
    for partition_name, partition in (("train", train_df), ("test", test_df)):
        grouped = partition.groupby("label").agg(
            rows=("label", "size"),
            runs=(group_column, "nunique"),
        )
        for label, row in grouped.iterrows():
            manifest_rows.append({
                "partition": partition_name,
                "label": label,
                "rows": int(row["rows"]),
                "runs": int(row["runs"]),
            })
    pd.DataFrame(manifest_rows).to_csv(PATHS["split_manifest_path"], index=False)

    print("=" * 78)
    print("LEAKAGE-SAFE GROUPED PREPROCESSING COMPLETE")
    print(f"Train: {len(train_out)} rows from {len(train_groups)} complete runs")
    print(f"Test : {len(test_out)} rows from {len(test_groups)} complete runs")
    print(f"Features retained: {len(usable_features)}")
    print(f"Constant/unsupported features dropped: {dropped_features or 'none'}")
    print("Run overlap: 0")
    print("No label-aware noise or augmentation was applied.")
    print("Scaling and imputation are deferred to model pipelines.")
    print("=" * 78)


if __name__ == "__main__":
    run_preprocessing()