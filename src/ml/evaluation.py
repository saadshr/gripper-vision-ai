"""
src/ml/evaluation.py
=====================

Fonctions d'évaluation et de visualisation pour le Machine Learning :
    - métriques standard (accuracy, precision, recall, F1)
    - matrice de confusion (figure + CSV)
    - importance des features (si le modèle le permet)
    - distribution des classes du dataset

Tous les graphiques sont enregistrés automatiquement dans
outputs/graphs/ (voir config.py), pour être consultés après coup ou inclus
dans un rapport / une soutenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import OUTPUTS_GRAPHS_DIR, OUTPUTS_REPORTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _get_pyplot():
    """Charge la bibliothèque graphique uniquement lors d'une demande de figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Matplotlib indisponible dans cet environnement. Le venv historique "
            "a un chemin Windows trop long ; prévoir un environnement neuf sur autorisation."
        ) from error
    return plt


@dataclass
class EvaluationResult:
    """Regroupe toutes les métriques calculées pour un modèle donné."""

    model_name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    cv_mean_accuracy: float
    cv_std_accuracy: float
    classification_report_text: str = field(repr=False)
    confusion_matrix_array: np.ndarray = field(repr=False)


def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    cv_scores: np.ndarray,
) -> EvaluationResult:
    """Calcule toutes les métriques standard pour un jeu de prédictions.

    Args:
        model_name: nom lisible du modèle (pour les logs et les rapports).
        y_true: vraies classes (encodées en entiers) du jeu de test.
        y_pred: classes prédites par le modèle sur le jeu de test.
        class_names: noms des classes, dans l'ordre de l'encodage.
        cv_scores: scores de validation croisée obtenus sur le jeu d'entraînement.

    Returns:
        Un EvaluationResult prêt à être comparé à ceux des autres modèles.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    logger.info(
        "[%s] accuracy=%.3f | precision=%.3f | recall=%.3f | f1=%.3f | cv=%.3f (+/-%.3f)",
        model_name, accuracy, precision, recall, f1, cv_scores.mean(), cv_scores.std(),
    )

    return EvaluationResult(
        model_name=model_name,
        accuracy=accuracy,
        precision_macro=precision,
        recall_macro=recall,
        f1_macro=f1,
        cv_mean_accuracy=float(cv_scores.mean()),
        cv_std_accuracy=float(cv_scores.std()),
        classification_report_text=report_text,
        confusion_matrix_array=cm,
    )


def save_confusion_matrix_plot(
    result: EvaluationResult, class_names: list[str], *, filename: str | None = None
) -> Path:
    """Sauvegarde la matrice de confusion d'un modèle sous forme d'image PNG."""
    plt = _get_pyplot()
    OUTPUTS_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    filename = filename or f"confusion_matrix_{result.model_name}.png"
    output_path = OUTPUTS_GRAPHS_DIR / filename

    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=result.confusion_matrix_array, display_labels=class_names
    )
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True)
    ax.set_title(f"Matrice de confusion — {result.model_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

    logger.info("Matrice de confusion sauvegardée : %s", output_path)
    return output_path


def save_model_comparison_plot(results: list[EvaluationResult]) -> Path:
    """Sauvegarde un graphique en barres comparant l'accuracy de chaque modèle."""
    plt = _get_pyplot()
    OUTPUTS_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_GRAPHS_DIR / "model_comparison.png"

    names = [r.model_name for r in results]
    test_acc = [r.accuracy for r in results]
    cv_acc = [r.cv_mean_accuracy for r in results]
    cv_std = [r.cv_std_accuracy for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, test_acc, width, label="Accuracy (test)", color="#2E6E9E")
    ax.bar(x + width / 2, cv_acc, width, yerr=cv_std, capsize=4,
           label="Accuracy moyenne (CV)", color="#E07A1F")

    ax.set_ylabel("Accuracy")
    ax.set_title("Comparaison des modèles")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

    logger.info("Comparaison des modèles sauvegardée : %s", output_path)
    return output_path


def save_feature_importance_plot(
    feature_names: list[str], importances: np.ndarray, model_name: str
) -> Path:
    """Sauvegarde l'importance des features (si le modèle la fournit)."""
    plt = _get_pyplot()
    OUTPUTS_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_GRAPHS_DIR / "feature_importance.png"

    series = pd.Series(importances, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(series) * 0.28)))
    series.plot.barh(ax=ax, color="#2F7A5C")
    ax.set_title(f"Importance des features — {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

    logger.info("Importance des features sauvegardée : %s", output_path)
    return output_path


def save_class_distribution_plot(labels: pd.Series) -> Path:
    """Sauvegarde la distribution du nombre d'images par classe."""
    plt = _get_pyplot()
    OUTPUTS_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_GRAPHS_DIR / "class_distribution.png"

    counts = labels.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts.plot.bar(ax=ax, color="#0E4459")
    ax.set_title("Distribution des classes dans le dataset")
    ax.set_ylabel("Nombre d'images")
    ax.set_xlabel("Gripper")
    for i, value in enumerate(counts):
        ax.text(i, value + 0.3, str(value), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

    logger.info("Distribution des classes sauvegardée : %s", output_path)
    return output_path


def save_feature_histograms(dataframe: pd.DataFrame, feature_names: list[str]) -> Path:
    """Sauvegarde un histogramme par feature géométrique clé, coloré par classe."""
    plt = _get_pyplot()
    OUTPUTS_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_GRAPHS_DIR / "feature_histograms.png"

    key_features = [f for f in
                     ["area", "circularity", "aspect_ratio", "solidity", "num_holes", "num_corners"]
                     if f in feature_names][:6]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, feature in zip(axes, key_features):
        for label, group in dataframe.groupby("label"):
            ax.hist(group[feature], bins=12, alpha=0.5, label=label)
        ax.set_title(feature)
        ax.legend(fontsize=7)

    fig.suptitle("Distribution des features clés par classe")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

    logger.info("Histogrammes de features sauvegardés : %s", output_path)
    return output_path


def write_text_report(results: list[EvaluationResult], best_model_name: str) -> Path:
    """Écrit un rapport texte lisible, résumant la comparaison des modèles."""
    OUTPUTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_REPORTS_DIR / "training_report.txt"

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("RAPPORT D'ENTRAINEMENT — Gripper Vision AI")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Meilleur modèle retenu : {best_model_name}")
    lines.append("")
    lines.append("Comparaison des modèles (triés par F1-macro décroissant) :")
    lines.append("-" * 70)

    sorted_results = sorted(results, key=lambda r: r.f1_macro, reverse=True)
    for r in sorted_results:
        lines.append(
            f"{r.model_name:20s} | accuracy={r.accuracy:.3f} | "
            f"precision={r.precision_macro:.3f} | recall={r.recall_macro:.3f} | "
            f"f1_macro={r.f1_macro:.3f} | cv={r.cv_mean_accuracy:.3f} (+/-{r.cv_std_accuracy:.3f})"
        )

    lines.append("")
    lines.append("=" * 70)
    lines.append(f"Rapport de classification détaillé — {best_model_name}")
    lines.append("=" * 70)
    best_result = next(r for r in results if r.model_name == best_model_name)
    lines.append(best_result.classification_report_text)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Rapport texte sauvegardé : %s", output_path)
    return output_path


def save_model_comparison_csv(results: list[EvaluationResult], best_model_name: str) -> Path:
    """Sauvegarde la comparaison des modèles dans un CSV structuré.

    Complète write_text_report() (lisible par un humain) avec un format
    exploitable directement par pandas — utilisé par le tableau de bord
    Streamlit pour afficher une vraie table interactive plutôt qu'un bloc
    de texte préformaté.
    """
    OUTPUTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_REPORTS_DIR / "model_comparison.csv"

    rows = [
        {
            "model_name": r.model_name,
            "accuracy": r.accuracy,
            "precision_macro": r.precision_macro,
            "recall_macro": r.recall_macro,
            "f1_macro": r.f1_macro,
            "cv_mean_accuracy": r.cv_mean_accuracy,
            "cv_std_accuracy": r.cv_std_accuracy,
            "is_best_model": r.model_name == best_model_name,
        }
        for r in results
    ]
    dataframe = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    dataframe.to_csv(output_path, index=False)
    logger.info("Comparaison des modèles (CSV) sauvegardée : %s", output_path)
    return output_path
