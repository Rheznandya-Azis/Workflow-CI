#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
modelling.py
============
Melatih model klasifikasi kelayakan kredit menggunakan Scikit-Learn dan
mencatat proses pelatihan dengan MLflow autolog.

Variabel respon:
    Loan_Status
    - Y = 1: pengajuan disetujui/eligible
    - N = 0: pengajuan tidak disetujui/not eligible

Kolom yang tidak digunakan sebagai fitur:
    - Loan_ID: identifier
    - Loan_Amount_Term: konstan pada dataset preprocessing
    - Loan_Status: variabel respon

Dataset dianggap sudah bersih dan selesai melalui preprocessing. Oleh karena
itu, file ini tidak melakukan imputasi, encoding fitur, scaling, maupun
hyperparameter tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn

# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "credit-eligibility_preprocessing.xlsx"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_TRACKING_DATABASE = BASE_DIR / "mlflow.db"
DEFAULT_TRACKING_URI = f"sqlite:///{DEFAULT_TRACKING_DATABASE.as_posix()}"

EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "credit_eligibility_classification",
)
DEFAULT_RUN_NAME = "logistic_regression_fixed_parameters"

TARGET_COLUMN = "Loan_Status"
TARGET_MAPPING = {"N": 0, "Y": 1}
EXCLUDED_FEATURES = ["Loan_ID", "Loan_Amount_Term", TARGET_COLUMN]

RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20

# Parameter model ditetapkan langsung. Tidak terdapat GridSearchCV,
# RandomizedSearchCV, Optuna, atau metode hyperparameter tuning lainnya.
MODEL_PARAMS: dict[str, Any] = {
    "solver": "liblinear",
    "penalty": "l2",
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# Saat dijalankan melalui GitHub Actions/MLflow Project, tracking URI dibaca
# dari environment variable. Jika dijalankan langsung secara lokal, digunakan
# database SQLite di folder MLProject.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
mlflow.set_tracking_uri(TRACKING_URI)

# MLflow Project sudah membuat run sebelum modelling.py dieksekusi. Dalam
# kondisi tersebut MLFLOW_RUN_ID tersedia, sehingga eksperimen tidak perlu
# dibuat ulang oleh script. Saat dijalankan langsung, eksperimen diaktifkan di
# sini.
IS_MLFLOW_PROJECT_RUN = bool(os.getenv("MLFLOW_RUN_ID"))
if not IS_MLFLOW_PROJECT_RUN:
    mlflow.set_experiment(EXPERIMENT_NAME)

# Autolog diaktifkan sebelum fungsi evaluasi Scikit-Learn digunakan.
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    log_datasets=True,
    log_post_training_metrics=True,
)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def parse_arguments() -> argparse.Namespace:
    """Membaca parameter ketika modelling.py dijalankan."""
    parser = argparse.ArgumentParser(
        description="Melatih model kelayakan kredit dengan MLflow."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Lokasi file Excel hasil preprocessing.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Proporsi data pengujian, misalnya 0.20.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder penyimpanan model dan hasil evaluasi.",
    )
    return parser.parse_args()


def calculate_sha256(file_path: Path) -> str:
    """Menghitung checksum dataset untuk ketertelusuran eksperimen."""
    digest = hashlib.sha256()
    with file_path.open("rb") as file_object:
        for block in iter(lambda: file_object.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_input(df: pd.DataFrame, test_size: float) -> None:
    """
    Memvalidasi kesiapan data tanpa melakukan pembersihan atau transformasi.
    """
    required_columns = set(EXCLUDED_FEATURES)
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {sorted(missing_columns)}"
        )

    if not 0.0 < test_size < 1.0:
        raise ValueError("--test-size harus berada di antara 0 dan 1.")

    if df.empty:
        raise ValueError("Dataset kosong.")

    if df.isna().any().any():
        raise ValueError(
            "Dataset masih mengandung missing value. "
            "Gunakan file hasil preprocessing yang sudah bersih."
        )

    target_values = set(df[TARGET_COLUMN].unique())
    unexpected_targets = target_values.difference(TARGET_MAPPING)
    if unexpected_targets:
        raise ValueError(
            f"Nilai target tidak dikenali: {sorted(unexpected_targets)}. "
            "Nilai yang diharapkan hanya 'Y' dan 'N'."
        )

    feature_columns = [
        column for column in df.columns if column not in EXCLUDED_FEATURES
    ]
    if not feature_columns:
        raise ValueError("Tidak terdapat fitur yang dapat digunakan.")

    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric_features:
        raise TypeError(
            "Seluruh fitur harus sudah numerik setelah preprocessing. "
            f"Kolom nonnumerik: {non_numeric_features}"
        )


def prepare_model_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Memisahkan fitur dan variabel respon."""
    feature_columns = [
        column for column in df.columns if column not in EXCLUDED_FEATURES
    ]

    X = df.loc[:, feature_columns].copy()
    y = df[TARGET_COLUMN].map(TARGET_MAPPING).astype(int)
    return X, y


def create_confusion_matrix_artifact(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> plt.Figure:
    """Membuat visualisasi confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["N / Tidak Disetujui", "Y / Disetujui"],
    )

    figure, axis = plt.subplots(figsize=(7, 5))
    display.plot(ax=axis, values_format="d", colorbar=False)
    axis.set_title("Confusion Matrix – Data Pengujian")
    figure.tight_layout()
    return figure


def collect_execution_tags() -> dict[str, str]:
    """Mengambil metadata GitHub Actions jika script dijalankan melalui CI."""
    environment_to_tag = {
        "GITHUB_ACTIONS": "ci.github_actions",
        "GITHUB_REPOSITORY": "ci.repository",
        "GITHUB_REF_NAME": "ci.branch",
        "GITHUB_SHA": "ci.commit_sha",
        "GITHUB_RUN_ID": "ci.workflow_run_id",
        "GITHUB_RUN_NUMBER": "ci.workflow_run_number",
        "GITHUB_ACTOR": "ci.triggered_by",
    }

    tags: dict[str, str] = {}
    for environment_name, tag_name in environment_to_tag.items():
        value = os.getenv(environment_name)
        if value:
            tags[tag_name] = value
    return tags


def train_and_track(
    data_path: Path,
    test_size: float,
    output_dir: Path,
) -> None:
    """Melatih model dan menyimpan hasil eksperimen ke MLflow."""
    resolved_data_path = data_path.expanduser().resolve()
    resolved_output_dir = output_dir.expanduser().resolve()

    if not resolved_data_path.exists():
        raise FileNotFoundError(
            f"File data tidak ditemukan: {resolved_data_path}"
        )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset sudah selesai melalui preprocessing; tahap ini hanya membaca data.
    df = pd.read_excel(resolved_data_path, sheet_name="Sheet1")
    validate_input(df, test_size)

    X, y = prepare_model_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = LogisticRegression(**MODEL_PARAMS)

    # Jika dipanggil oleh "mlflow run", start_run() akan melanjutkan run yang
    # ID-nya diberikan melalui MLFLOW_RUN_ID. Jika dijalankan langsung, run baru
    # dibuat menggunakan DEFAULT_RUN_NAME.
    start_run_kwargs: dict[str, Any] = {}
    if not IS_MLFLOW_PROJECT_RUN:
        start_run_kwargs["run_name"] = DEFAULT_RUN_NAME

    with mlflow.start_run(**start_run_kwargs) as active_run:
        base_tags = {
            "task": "binary_classification",
            "target": TARGET_COLUMN,
            "positive_class": "Y",
            "negative_class": "N",
            "data_preprocessed": "true",
            "hyperparameter_tuning": "false",
            "model_family": "logistic_regression",
            "execution_mode": (
                "mlflow_project" if IS_MLFLOW_PROJECT_RUN else "direct_python"
            ),
        }
        base_tags.update(collect_execution_tags())
        mlflow.set_tags(base_tags)

        mlflow.log_params(
            {
                "target_column": TARGET_COLUMN,
                "test_size": test_size,
                "random_state": RANDOM_STATE,
                "dataset_filename": resolved_data_path.name,
                "dataset_sha256": calculate_sha256(resolved_data_path),
                "number_of_rows": len(df),
                "number_of_features": X.shape[1],
            }
        )

        # fit() memicu MLflow Scikit-Learn autolog.
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_probability = model.predict_proba(X_test)[:, 1]

        metrics = {
            "train_accuracy": float(model.score(X_train, y_train)),
            "test_accuracy": float(model.score(X_test, y_test)),
            "test_precision": float(
                precision_score(y_test, y_test_pred, zero_division=0)
            ),
            "test_recall": float(
                recall_score(y_test, y_test_pred, zero_division=0)
            ),
            "test_f1_score": float(
                f1_score(y_test, y_test_pred, zero_division=0)
            ),
            "test_roc_auc": float(
                roc_auc_score(y_test, y_test_probability)
            ),
        }
        mlflow.log_metrics(metrics)

        data_summary = {
            "source_file": resolved_data_path.name,
            "dataset_sha256": calculate_sha256(resolved_data_path),
            "total_rows": int(len(df)),
            "number_of_features": int(X.shape[1]),
            "feature_columns": X.columns.tolist(),
            "excluded_columns": EXCLUDED_FEATURES,
            "target_mapping": TARGET_MAPPING,
            "target_distribution_original": {
                str(key): int(value)
                for key, value in df[TARGET_COLUMN].value_counts().items()
            },
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_size": float(test_size),
            "random_state": RANDOM_STATE,
        }
        mlflow.log_dict(data_summary, "evaluation/data_summary.json")

        report = classification_report(
            y_test,
            y_test_pred,
            labels=[0, 1],
            target_names=["N", "Y"],
            output_dict=True,
            zero_division=0,
        )
        mlflow.log_dict(report, "evaluation/classification_report.json")

        confusion_figure = create_confusion_matrix_artifact(
            y_test,
            y_test_pred,
        )
        mlflow.log_figure(
            confusion_figure,
            "evaluation/confusion_matrix.png",
        )
        plt.close(confusion_figure)

        coefficient_table = pd.DataFrame(
            {
                "feature": X.columns,
                "coefficient": model.coef_[0],
            }
        )
        coefficient_table["absolute_coefficient"] = (
            coefficient_table["coefficient"].abs()
        )
        coefficient_table = coefficient_table.sort_values(
            "absolute_coefficient",
            ascending=False,
        )

        coefficient_path = resolved_output_dir / "model_coefficients.csv"
        coefficient_table.to_csv(coefficient_path, index=False)
        mlflow.log_artifact(
            str(coefficient_path),
            artifact_path="evaluation",
        )

        prediction_table = X_test.copy()
        prediction_table["actual_loan_status"] = y_test
        prediction_table["predicted_loan_status"] = y_test_pred
        prediction_table["probability_status_Y"] = y_test_probability

        prediction_path = resolved_output_dir / "test_predictions.csv"
        prediction_table.to_csv(prediction_path, index=True)
        mlflow.log_artifact(
            str(prediction_path),
            artifact_path="evaluation",
        )

        metrics_path = resolved_output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="evaluation",
        )

        # File model eksplisit disimpan agar dapat diunduh langsung sebagai
        # GitHub Actions artifact, selain model yang dicatat oleh autolog MLflow.
        model_path = resolved_output_dir / "credit_eligibility_model.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(
            str(model_path),
            artifact_path="exported_model",
        )

        feature_schema_path = resolved_output_dir / "feature_schema.json"
        feature_schema_path.write_text(
            json.dumps(
                {
                    "features": X.columns.tolist(),
                    "target": TARGET_COLUMN,
                    "target_mapping": TARGET_MAPPING,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        mlflow.log_artifact(
            str(feature_schema_path),
            artifact_path="exported_model",
        )

        print("\nPelatihan selesai.")
        print(f"Run ID             : {active_run.info.run_id}")
        print(f"Experiment         : {EXPERIMENT_NAME}")
        print(f"Tracking URI       : {mlflow.get_tracking_uri()}")
        print(f"Dataset            : {resolved_data_path}")
        print(f"Output folder      : {resolved_output_dir}")
        print(f"Jumlah fitur       : {X.shape[1]}")
        print(f"Train observations : {len(X_train)}")
        print(f"Test observations  : {len(X_test)}")
        print(json.dumps(metrics, indent=2))


def main() -> None:
    args = parse_arguments()
    train_and_track(
        data_path=args.data,
        test_size=args.test_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
