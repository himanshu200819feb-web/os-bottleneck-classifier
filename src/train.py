"""Benchmark heuristic, ML, and DL models with grouped CV and untouched test runs."""

import copy
import os
from pathlib import Path
import random
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


CONFIG = load_config()
PATHS = CONFIG["paths"]
SEED = CONFIG["project"]["random_seed"]
CV_FOLDS = CONFIG["evaluation"]["cv_folds"]
SCORING = CONFIG["evaluation"]["scoring_metric"]
GROUP_COLUMN = CONFIG["preprocessing"]["group_column"]


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prefixed_grid(parameters: dict) -> dict:
    return {f"model__{name}": values for name, values in parameters.items()}


def metrics_record(model_name, model_type, parameters, y_true, y_pred, latency, cv_score=None):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "Model": model_name,
        "Type": model_type,
        "Best_Params": str(parameters),
        "CV_Macro_F1": np.nan if cv_score is None else round(float(cv_score), 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Macro_F1": round(f1, 4),
        "Latency_ms": round(latency, 4),
    }


def evaluate_static_heuristic(test_df: pd.DataFrame, label_encoder):
    thresholds = CONFIG["models"]["baseline_heuristic"]["thresholds"]
    available_labels = set(label_encoder.classes_)
    required_labels = {"Normal", "CPU_Bound", "Memory_Pressure", "IO_Contention"}
    if not required_labels.issubset(available_labels):
        raise ValueError(f"Baseline expects labels {sorted(required_labels)}")

    predictions = []
    start = time.perf_counter()
    for _, row in test_df.iterrows():
        if row.get("cpu_user_pct", 0) >= thresholds["cpu_bound_user_pct"]:
            label = "CPU_Bound"
        elif (
            row.get("swap_out_kb_per_sec", 0) >= thresholds["memory_pressure_swap_out_kb_per_sec"]
            or row.get("mem_used_pct", 0) >= thresholds["memory_pressure_used_pct"]
        ):
            label = "Memory_Pressure"
        elif (
            row.get("cpu_iowait_pct", 0) >= thresholds["io_contention_iowait_pct"]
            or row.get("disk_write_kb_per_sec", 0) >= thresholds["io_contention_write_kb_per_sec"]
        ):
            label = "IO_Contention"
        else:
            label = "Normal"
        predictions.append(label)

    latency = (time.perf_counter() - start) * 1000 / len(test_df)
    y_pred = label_encoder.transform(predictions)
    return metrics_record(
        "Static_Rule_Baseline", "Heuristic Baseline", thresholds,
        test_df["label"].to_numpy(), y_pred, latency,
    )


class ConvNet1D(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(min(4, input_dim)),
        )
        pooled = min(4, input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(64 * pooled, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, values):
        values = self.features(values)
        return self.classifier(values.flatten(1))


def fit_cnn_epochs(X, y, input_dim, num_classes, epochs, batch_size, learning_rate, validation=None, patience=6):
    model = ConvNet1D(input_dim, num_classes)
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1), torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_state = None
    best_loss = float("inf")
    best_epoch = epochs
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        if validation is not None:
            X_val, y_val = validation
            model.eval()
            with torch.no_grad():
                val_output = model(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1))
                val_loss = criterion(val_output, torch.tensor(y_val, dtype=torch.long)).item()
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def train_group_validated_cnn(X_train, y_train, train_groups, X_test, y_test, num_classes):
    cfg = CONFIG["models"]["cnn_1d"]
    group_counts = pd.Series(train_groups).nunique()
    folds = min(5, group_counts)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    subtrain_idx, validation_idx = next(splitter.split(X_train, y_train, train_groups))

    validation_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    X_subtrain = validation_preprocessor.fit_transform(X_train.iloc[subtrain_idx])
    X_validation = validation_preprocessor.transform(X_train.iloc[validation_idx])

    set_seeds(SEED)
    _, best_epoch = fit_cnn_epochs(
        X_subtrain, y_train[subtrain_idx], X_train.shape[1], num_classes,
        int(cfg["max_epochs"]), int(cfg["batch_size"]), float(cfg["learning_rate"]),
        validation=(X_validation, y_train[validation_idx]),
        patience=int(cfg["early_stopping_patience"]),
    )

    final_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    X_train_final = final_preprocessor.fit_transform(X_train)
    X_test_final = final_preprocessor.transform(X_test)

    set_seeds(SEED)
    final_model, _ = fit_cnn_epochs(
        X_train_final, y_train, X_train.shape[1], num_classes,
        best_epoch, int(cfg["batch_size"]), float(cfg["learning_rate"]),
    )
    final_model.eval()
    test_tensor = torch.tensor(X_test_final, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        final_model(test_tensor[: min(32, len(test_tensor))])
        start = time.perf_counter()
        predictions = final_model(test_tensor).argmax(dim=1).cpu().numpy()
    latency = (time.perf_counter() - start) * 1000 / len(X_test_final)

    joblib.dump(final_preprocessor, PATHS["scaler_path"])
    torch.save({
        "state_dict": final_model.state_dict(),
        "input_dim": X_train.shape[1],
        "num_classes": num_classes,
        "epochs": best_epoch,
    }, os.path.join(PATHS["artifacts_dir"], "CNN_1D.pt"))

    return metrics_record(
        "1D_CNN_Net (DL)", "Deep Learning",
        {"epochs": best_epoch, "learning_rate": cfg["learning_rate"]},
        y_test, predictions, latency,
    )


def make_pipeline(estimator, scale: bool) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def run_training_pipeline() -> None:
    set_seeds(SEED)
    os.makedirs(PATHS["artifacts_dir"], exist_ok=True)
    train_df = pd.read_csv(PATHS["train_csv"])
    test_df = pd.read_csv(PATHS["test_csv"])
    feature_columns = joblib.load(PATHS["feature_names_path"])
    label_encoder = joblib.load(PATHS["label_encoder_path"])

    if GROUP_COLUMN not in train_df.columns or GROUP_COLUMN not in test_df.columns:
        raise ValueError("Processed data has no run_id. Run the fixed preprocess.py first.")
    overlap = set(train_df[GROUP_COLUMN].astype(str)) & set(test_df[GROUP_COLUMN].astype(str))
    if overlap:
        raise ValueError(f"Data leakage detected: {len(overlap)} run IDs cross train/test.")

    X_train = train_df[feature_columns]
    y_train = train_df["label"].to_numpy(dtype=int)
    train_groups = train_df[GROUP_COLUMN].astype(str).to_numpy()
    X_test = test_df[feature_columns]
    y_test = test_df["label"].to_numpy(dtype=int)
    num_classes = len(label_encoder.classes_)

    runs_by_class = train_df.groupby("label")[GROUP_COLUMN].nunique()
    cv_folds = min(int(CV_FOLDS), int(runs_by_class.min()))
    if cv_folds < 2:
        raise ValueError("At least two training runs per class are required for grouped CV.")
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)

    registry = {
        "Logistic_Regression": (
            LogisticRegression(random_state=SEED),
            CONFIG["models"]["logistic_regression"], True, "Classical ML"
        ),
        "Random_Forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            CONFIG["models"]["random_forest"], False, "Classical ML"
        ),
        "XGBoost": (
            XGBClassifier(random_state=SEED, eval_metric="mlogloss", n_jobs=1),
            CONFIG["models"]["xgboost"], False, "Classical ML"
        ),
        "LightGBM": (
            LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=1),
            CONFIG["models"]["lightgbm"], False, "Classical ML"
        ),
        "SVM_RBF": (
            SVC(kernel="rbf", probability=CONFIG["models"]["svm_rbf"].get("probability", True), random_state=SEED),
            CONFIG["models"]["svm_rbf"], True, "Classical ML"
        ),
        "MLP_Deep_Net": (
            MLPClassifier(random_state=SEED, max_iter=CONFIG["models"]["mlp_deep_net"]["max_iter"]),
            CONFIG["models"]["mlp_deep_net"], True, "Deep Learning"
        ),
    }

    print("=" * 92)
    print("LEAKAGE-SAFE OS TELEMETRY BENCHMARK")
    print(f"Train runs: {train_df[GROUP_COLUMN].nunique()} | Test runs: {test_df[GROUP_COLUMN].nunique()}")
    print(f"Features: {len(feature_columns)} | Grouped CV folds: {cv_folds}")
    print("=" * 92)

    results = [evaluate_static_heuristic(test_df, label_encoder)]
    for model_name, (estimator, cfg, requires_scaling, model_type) in registry.items():
        print(f"[*] Grouped optimization and training: {model_name}")
        pipeline = make_pipeline(estimator, requires_scaling)
        grid = GridSearchCV(
            pipeline,
            prefixed_grid(cfg["param_grid"]),
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
            error_score="raise",
        )
        grid.fit(X_train, y_train, groups=train_groups)
        best_model = grid.best_estimator_
        best_model.predict(X_test.iloc[: min(32, len(X_test))])
        start = time.perf_counter()
        predictions = best_model.predict(X_test)
        latency = (time.perf_counter() - start) * 1000 / len(X_test)
        readable_params = {
            key.replace("model__", ""): value for key, value in grid.best_params_.items()
        }
        result = metrics_record(
            model_name, model_type, readable_params, y_test, predictions,
            latency, grid.best_score_,
        )
        results.append(result)
        joblib.dump(best_model, os.path.join(PATHS["artifacts_dir"], f"{model_name}.joblib"))
        print(
            f"    CV Macro F1={result['CV_Macro_F1']:.4f} | "
            f"Test Macro F1={result['Macro_F1']:.4f} | Accuracy={result['Accuracy']:.4f}"
        )

    print("[*] Group-validated training: 1D_CNN_Net")
    cnn_result = train_group_validated_cnn(
        X_train, y_train, train_groups, X_test, y_test, num_classes
    )
    results.append(cnn_result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(PATHS["model_metrics_csv"], index=False)
    display_columns = [
        "Model", "Type", "CV_Macro_F1", "Accuracy", "Precision",
        "Recall", "Macro_F1", "Latency_ms",
    ]
    print("\n" + "=" * 100)
    print(results_df[display_columns].to_string(index=False))
    print("=" * 100)
    print(f"Metrics written to {PATHS['model_metrics_csv']}")


if __name__ == "__main__":
    run_training_pipeline()