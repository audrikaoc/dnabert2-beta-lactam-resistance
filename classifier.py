#!/usr/bin/env python3
"""Train and evaluate a Random Forest classifier on DNABERT-2 embeddings."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit


DEFAULT_EMBEDDINGS_PATH = "embeddings.npy"
DEFAULT_LABELS_PATH = "labeled_sequences.csv"
DEFAULT_LABEL_COLUMN = "label"
DEFAULT_SEQUENCE_COLUMN = "sequence"
DEFAULT_OUTPUT_DIR = "classifier_plots"
DEFAULT_METRICS_PATH = "test_metrics.json"
DEFAULT_GROUP_NEAR_DUPLICATES = True
DEFAULT_KMER_SIZE = 15
DEFAULT_SIMILARITY_THRESHOLD = 0.85


def load_data(
    embeddings_path: str | Path = DEFAULT_EMBEDDINGS_PATH,
    labels_path: str | Path = DEFAULT_LABELS_PATH,
    label_column: str = DEFAULT_LABEL_COLUMN,
    sequence_column: str = DEFAULT_SEQUENCE_COLUMN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load embedding matrix and binary labels, validating row alignment."""
    embeddings = np.load(embeddings_path)
    labels_df = pd.read_csv(labels_path)

    if label_column not in labels_df.columns:
        raise KeyError(f"Labels file must contain a '{label_column}' column")

    labels = labels_df[label_column].to_numpy()
    sequences = (
        labels_df[sequence_column].astype(str).to_numpy()
        if sequence_column in labels_df.columns
        else None
    )
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            "Embedding rows and labels do not match: "
            f"{embeddings.shape[0]} embeddings vs {labels.shape[0]} labels"
        )

    return embeddings, labels, sequences


def build_near_duplicate_groups(
    sequences: np.ndarray,
    kmer_size: int = DEFAULT_KMER_SIZE,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[np.ndarray, dict]:
    """Cluster sequences whose k-mer Jaccard similarity meets the threshold."""
    if kmer_size < 1:
        raise ValueError("kmer_size must be at least 1")
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")

    sequence_list = [str(sequence) for sequence in sequences]
    n_sequences = len(sequence_list)
    parents = list(range(n_sequences))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    kmer_sets = [
        {sequence[i : i + kmer_size] for i in range(max(0, len(sequence) - kmer_size + 1))}
        for sequence in sequence_list
    ]

    near_duplicate_pairs = 0
    for left, right in combinations(range(n_sequences), 2):
        left_len = len(sequence_list[left])
        right_len = len(sequence_list[right])
        if left_len == 0 or right_len == 0:
            continue
        if min(left_len, right_len) / max(left_len, right_len) < similarity_threshold:
            continue

        intersection_size = len(kmer_sets[left] & kmer_sets[right])
        union_size = len(kmer_sets[left] | kmer_sets[right])
        similarity = intersection_size / union_size if union_size else 0.0
        if similarity >= similarity_threshold:
            union(left, right)
            near_duplicate_pairs += 1

    raw_groups = [find(index) for index in range(n_sequences)]
    compressed_group_ids = {
        group_id: compressed_id for compressed_id, group_id in enumerate(sorted(set(raw_groups)))
    }
    groups = np.array([compressed_group_ids[group_id] for group_id in raw_groups])
    group_sizes = Counter(groups)
    audit = {
        "kmer_size": int(kmer_size),
        "similarity_threshold": float(similarity_threshold),
        "n_groups": int(len(group_sizes)),
        "largest_group_size": int(max(group_sizes.values()) if group_sizes else 0),
        "near_duplicate_pairs": int(near_duplicate_pairs),
    }
    return groups, audit


def split_data(
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Split data, keeping near-duplicate groups together when provided."""
    if groups is None:
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
        return X_train, X_test, y_train, y_test, {"group_aware_split": False}

    splitter = GroupShuffleSplit(n_splits=100, test_size=test_size, random_state=random_state)
    best_split = None
    overall_positive_rate = labels.mean()

    for train_index, test_index in splitter.split(embeddings, labels, groups):
        if len(np.unique(labels[test_index])) < 2:
            continue
        split_score = abs((len(test_index) / len(labels)) - test_size)
        split_score += abs(labels[test_index].mean() - overall_positive_rate)
        if best_split is None or split_score < best_split[0]:
            best_split = (split_score, train_index, test_index)

    if best_split is None:
        raise ValueError("Could not create a grouped split containing both classes")

    train_index = best_split[1]
    test_index = best_split[2]
    split_audit = {
        "group_aware_split": True,
        "train_samples": int(len(train_index)),
        "test_samples": int(len(test_index)),
        "train_groups": int(len(set(groups[train_index]))),
        "test_groups": int(len(set(groups[test_index]))),
    }
    return (
        embeddings[train_index],
        embeddings[test_index],
        labels[train_index],
        labels[test_index],
        split_audit,
    )


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save a high-resolution confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Control (0)", "Resistant (1)"],
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Random Forest Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_roc_auc_plot(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: str | Path,
) -> float:
    """Save a high-resolution ROC-AUC curve and return the AUC score."""
    auc_score = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_true,
        y_score,
        name=f"Random Forest (AUC = {auc_score:.3f})",
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("Random Forest ROC-AUC Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return auc_score


def save_metrics_json(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    accuracy: float,
    roc_auc: float,
    output_path: str | Path,
    split_audit: dict | None = None,
    grouping_audit: dict | None = None,
) -> dict:
    """Save evaluation metrics to a structured JSON file."""
    binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "positive_class": {
            "label": 1,
            "name": "Resistant",
            "precision": float(binary_precision),
            "recall": float(binary_recall),
            "f1_score": float(binary_f1),
        },
        "macro_average": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1_score": float(macro_f1),
        },
        "weighted_average": {
            "precision": float(weighted_precision),
            "recall": float(weighted_recall),
            "f1_score": float(weighted_f1),
        },
        "test_set": {
            "n_samples": int(len(y_true)),
            "n_control": int(np.sum(y_true == 0)),
            "n_resistant": int(np.sum(y_true == 1)),
        },
    }
    if split_audit is not None:
        metrics["split"] = split_audit
    if grouping_audit is not None:
        metrics["near_duplicate_grouping"] = grouping_audit

    metrics_file = Path(output_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def train_and_evaluate(
    embeddings_path: str | Path = DEFAULT_EMBEDDINGS_PATH,
    labels_path: str | Path = DEFAULT_LABELS_PATH,
    label_column: str = DEFAULT_LABEL_COLUMN,
    sequence_column: str = DEFAULT_SEQUENCE_COLUMN,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
    n_jobs: int = 1,
    group_near_duplicates: bool = DEFAULT_GROUP_NEAR_DUPLICATES,
    kmer_size: int = DEFAULT_KMER_SIZE,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> RandomForestClassifier:
    """Train a RandomForestClassifier and save evaluation plots."""
    embeddings, labels, sequences = load_data(
        embeddings_path,
        labels_path,
        label_column,
        sequence_column,
    )

    groups = None
    grouping_audit = None
    if group_near_duplicates:
        if sequences is None:
            raise KeyError(
                f"Labels file must contain a '{sequence_column}' column for grouped splitting"
            )
        groups, grouping_audit = build_near_duplicate_groups(
            sequences,
            kmer_size=kmer_size,
            similarity_threshold=similarity_threshold,
        )
        mixed_label_groups = sum(
            1 for group_id in set(groups) if len(set(labels[groups == group_id])) > 1
        )
        grouping_audit["mixed_label_groups"] = int(mixed_label_groups)

    X_train, X_test, y_train, y_test, split_audit = split_data(
        embeddings,
        labels,
        groups,
        test_size,
        random_state,
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=n_jobs,
    )
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_score = classifier.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Control", "Resistant"]))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    confusion_matrix_path = output_path / "confusion_matrix.png"
    roc_auc_path = output_path / "roc_auc_curve.png"

    save_confusion_matrix_plot(y_test, y_pred, confusion_matrix_path)
    auc_score = save_roc_auc_plot(y_test, y_score, roc_auc_path)
    save_metrics_json(
        y_test,
        y_pred,
        accuracy,
        auc_score,
        metrics_path,
        split_audit=split_audit,
        grouping_audit=grouping_audit,
    )

    print(f"ROC-AUC: {auc_score:.4f}")
    print(f"Saved confusion matrix plot to {confusion_matrix_path}")
    print(f"Saved ROC-AUC curve plot to {roc_auc_path}")
    print(f"Saved test metrics to {metrics_path}")

    return classifier


def main() -> None:
    """Command-line entrypoint."""
    parser = argparse.ArgumentParser(
        description="Train a Random Forest classifier on DNABERT-2 embeddings."
    )
    parser.add_argument(
        "--embeddings",
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Path to the saved NumPy embedding matrix",
    )
    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS_PATH,
        help="CSV file containing binary labels aligned with the embeddings",
    )
    parser.add_argument(
        "--label-column",
        default=DEFAULT_LABEL_COLUMN,
        help="Name of the label column in the labels CSV",
    )
    parser.add_argument(
        "--sequence-column",
        default=DEFAULT_SEQUENCE_COLUMN,
        help="Name of the sequence column used for near-duplicate grouping",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where evaluation plots will be saved",
    )
    parser.add_argument(
        "--metrics-output",
        default=DEFAULT_METRICS_PATH,
        help="Path where structured test metrics JSON will be saved",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data held out for testing",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splitting and training",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees in the Random Forest",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for Random Forest training",
    )
    parser.add_argument(
        "--no-group-near-duplicates",
        action="store_true",
        help="Use a regular stratified split instead of grouping similar sequences",
    )
    parser.add_argument(
        "--kmer-size",
        type=int,
        default=DEFAULT_KMER_SIZE,
        help="k-mer size used to detect near-duplicate sequences",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="k-mer Jaccard similarity threshold for grouping near duplicates",
    )
    args = parser.parse_args()

    train_and_evaluate(
        embeddings_path=args.embeddings,
        labels_path=args.labels,
        label_column=args.label_column,
        sequence_column=args.sequence_column,
        output_dir=args.output_dir,
        metrics_path=args.metrics_output,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        group_near_duplicates=not args.no_group_near_duplicates,
        kmer_size=args.kmer_size,
        similarity_threshold=args.similarity_threshold,
    )


if __name__ == "__main__":
    main()
