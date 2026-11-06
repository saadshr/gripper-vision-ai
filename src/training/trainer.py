"""
src/training/trainer.py
========================

Orchestre l'entraînement complet : chargement du CSV de features,
préparation des données, comparaison de plusieurs modèles, sélection
automatique du meilleur, puis sauvegarde du modèle final.

Ce module ne contient pas de code d'affichage (print) : il journalise tout
via le logger centralisé (src/utils/logger.py) et retourne des objets
structurés. Son lancement reste bloqué jusqu'à une nouvelle autorisation.
Les imports graphiques sont différés pour garder les fonctions de split légères.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    CV_FOLDS,
    DATASET_CSV_PATH,
    LABEL_ENCODER_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    NON_FEATURE_COLUMNS,
    RANDOM_SEED,
    TEST_SIZE,
)
if TYPE_CHECKING:
    from src.ml.evaluation import EvaluationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingOutcome:
    """Résumé complet d'une session d'entraînement, retourné à train.py."""

    best_model_name: str
    best_pipeline: Pipeline
    label_encoder: LabelEncoder
    all_results: list[EvaluationResult]
    feature_names: list[str]
    n_samples: int
    n_classes: int


def _load_dataset() -> pd.DataFrame:
    """Charge le CSV de features généré par extract_features.py."""
    if not DATASET_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATASET_CSV_PATH}. "
            "La préparation des caractéristiques nécessite une nouvelle autorisation."
        )
    dataframe = pd.read_csv(DATASET_CSV_PATH)
    logger.info("Dataset chargé : %d lignes, %d colonnes", *dataframe.shape)
    return dataframe


def _split_features_and_labels(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Sépare les colonnes de features de la colonne cible (label)."""
    feature_columns = [c for c in dataframe.columns if c not in NON_FEATURE_COLUMNS]
    X = dataframe[feature_columns]
    y = dataframe["label"]
    return X, y, feature_columns


def _group_aware_split(
    dataframe: pd.DataFrame, y: np.ndarray, *, test_size: float, random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sépare les indices en train/test en gardant TOUTES les variantes
    d'une même photo source (vraie photo + ses variantes augmentées) du
    même côté du split.

    Pourquoi c'est nécessaire : si des variantes augmentées de la même
    photo source se retrouvaient de part et d'autre du split, le modèle
    "reconnaîtrait" en partie une image du jeu de test déjà vue sous une
    autre forme à l'entraînement — le score de test serait alors
    artificiellement gonflé (fuite de données). Le jeu de test est en
    plus restreint aux VRAIES photos uniquement (jamais de variante
    augmentée en test), pour que l'évaluation reste représentative de
    photos réelles jamais vues.

    Si le CSV ne contient pas de colonne 'source_image'/'is_augmented'
    (dataset généré par une ancienne version du pipeline), chaque ligne
    est traitée comme sa propre source — comportement identique à un
    split stratifié classique.

    Returns:
        (indices_train, indices_test) — des tableaux d'entiers positionnels,
        utilisables avec .iloc[].
    """
    if "source_image" not in dataframe.columns or "is_augmented" not in dataframe.columns:
        indices = np.arange(len(dataframe))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=y,
        )
        return train_idx, test_idx

    is_augmented = dataframe["is_augmented"].to_numpy()
    source_image = dataframe["source_image"].to_numpy()

    # Un (source_image -> label) unique par vraie photo, pour stratifier le
    # split sur les PHOTOS SOURCES plutôt que sur les lignes individuelles.
    real_mask = ~is_augmented
    sources_df = (
        pd.DataFrame({"source_image": source_image[real_mask], "label": y[real_mask]})
        .drop_duplicates(subset="source_image")
        .reset_index(drop=True)
    )

    train_sources, test_sources = train_test_split(
        sources_df["source_image"], test_size=test_size,
        random_state=random_state, stratify=sources_df["label"],
    )
    train_sources_set = set(train_sources)
    test_sources_set = set(test_sources)

    train_mask = np.isin(source_image, list(train_sources_set))
    # Jeu de test : uniquement de vraies photos (pas de variante augmentée),
    # même si (par construction actuelle) aucune variante n'existe pour les
    # photos de test — cette double condition protège aussi contre une
    # évolution future du pipeline d'augmentation.
    test_mask = np.isin(source_image, list(test_sources_set)) & real_mask

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, test_idx


def run_training(*, generate_plots: bool = True) -> TrainingOutcome:
    """Exécute le pipeline d'entraînement complet, de bout en bout.

    Étapes :
        1. Chargement du CSV de features.
        2. Encodage des labels + split train/test REGROUPÉ PAR PHOTO SOURCE
           (si le dataset inclut des variantes augmentées, toutes les
           variantes d'une même photo restent du même côté du split, et le
           jeu de test ne contient jamais de variante augmentée — voir
           _group_aware_split).
        3. Pour chaque modèle candidat : validation croisée sur le train,
           puis entraînement complet et évaluation sur le test.
        4. Sélection du modèle avec la meilleure accuracy moyenne en
           validation croisée (choix qui n'utilise JAMAIS le jeu de test,
           pour éviter toute fuite de données dans la sélection de modèle).
        5. Ré-entraînement du modèle gagnant sur l'intégralité des données
           disponibles (train + test) pour le modèle final sauvegardé —
           pratique standard en production, une fois l'architecture choisie.
        6. Sauvegarde du modèle, de l'encodeur de labels, des graphiques et
           du rapport texte.

    Returns:
        Un TrainingOutcome regroupant tous les résultats.
    """
    from src.safety import refuser_action
    refuser_action('run_training')
    from src.ml.models import get_candidate_models
    from src.ml.evaluation import (
        evaluate_predictions,
        save_class_distribution_plot,
        save_confusion_matrix_plot,
        save_feature_histograms,
        save_feature_importance_plot,
        save_model_comparison_csv,
        save_model_comparison_plot,
        write_text_report,
    )
    dataframe = _load_dataset()
    X, y_raw, feature_names = _split_features_and_labels(dataframe)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)

    X_train_idx, X_test_idx = _group_aware_split(
        dataframe, y, test_size=TEST_SIZE, random_state=RANDOM_SEED,
    )
    X_train, X_test = X.iloc[X_train_idx], X.iloc[X_test_idx]
    y_train, y_test = y[X_train_idx], y[X_test_idx]
    logger.info(
        "Split train/test : %d échantillons train, %d échantillons test "
        "(regroupés par photo source — aucune variante augmentée d'une photo "
        "de test ne peut apparaître à l'entraînement)",
        len(X_train), len(X_test),
    )

    candidate_models = get_candidate_models(random_state=RANDOM_SEED)
    results: list[EvaluationResult] = []
    fitted_pipelines: dict[str, Pipeline] = {}

    # Groupes (photo source) pour le TRAIN uniquement : la validation
    # croisée doit elle aussi éviter qu'une variante augmentée d'une photo
    # se retrouve dans le fold de validation pendant qu'une autre variante
    # de la MÊME photo est dans le fold d'entraînement (même risque de
    # fuite que pour le split train/test, mais à l'intérieur de la CV).
    groups_train = dataframe.iloc[X_train_idx]["source_image"].to_numpy()
    n_cv_folds = min(CV_FOLDS, _min_group_count_per_class(y_train, groups_train))
    cv_splitter = StratifiedGroupKFold(n_splits=n_cv_folds, shuffle=True, random_state=RANDOM_SEED)

    for model_name, model in candidate_models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])

        # Validation croisée UNIQUEMENT sur le train, regroupée par photo
        # source (StratifiedGroupKFold) : sert à la sélection du modèle.
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv_splitter, groups=groups_train,
            scoring="accuracy",
        )

        # Entraînement complet + évaluation sur le jeu de test, pour le rapport final.
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        result = evaluate_predictions(model_name, y_test, y_pred, class_names, cv_scores)
        results.append(result)
        fitted_pipelines[model_name] = pipeline

        if generate_plots:
            save_confusion_matrix_plot(result, class_names)

    # --- Sélection du meilleur modèle sur la base de la validation croisée ---
    best_result = max(results, key=lambda r: r.cv_mean_accuracy)
    best_model_name = best_result.model_name
    logger.info(
        "Meilleur modèle retenu : %s (cv_accuracy=%.3f)",
        best_model_name, best_result.cv_mean_accuracy,
    )

    # --- Ré-entraînement final sur TOUTES les données disponibles ---
    final_model = get_candidate_models(random_state=RANDOM_SEED)[best_model_name]
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", final_model),
    ])
    final_pipeline.fit(X, y)

    if generate_plots:
        save_model_comparison_plot(results)
        save_class_distribution_plot(dataframe["label"])
        save_feature_histograms(dataframe, feature_names)

        trained_model = final_pipeline.named_steps["model"]
        if hasattr(trained_model, "feature_importances_"):
            save_feature_importance_plot(
                feature_names, trained_model.feature_importances_, best_model_name
            )
        else:
            logger.info(
                "Le modèle '%s' ne fournit pas d'importance de features native (ignoré).",
                best_model_name,
            )

    write_text_report(results, best_model_name)
    save_model_comparison_csv(results, best_model_name)
    _save_model_artifacts(final_pipeline, label_encoder, best_model_name, results, feature_names)

    return TrainingOutcome(
        best_model_name=best_model_name,
        best_pipeline=final_pipeline,
        label_encoder=label_encoder,
        all_results=results,
        feature_names=feature_names,
        n_samples=len(dataframe),
        n_classes=len(class_names),
    )


def _min_group_count_per_class(y: np.ndarray, groups: np.ndarray) -> int:
    """Retourne, pour la classe la moins bien représentée, le nombre de
    PHOTOS SOURCES DISTINCTES qui la composent (pas le nombre de lignes).

    StratifiedGroupKFold a besoin d'au moins `n_splits` groupes distincts
    par classe — avec l'augmentation de données, une même photo source
    donne plusieurs lignes (1 originale + N variantes) qui doivent être
    comptées comme UN SEUL groupe, sinon le nombre de folds serait
    surestimé et scikit-learn lèverait une erreur ou, pire, se rabattrait
    silencieusement sur un split incorrect.
    """
    unique_pairs = {(g, label) for g, label in zip(groups, y)}
    _, counts = np.unique([label for _, label in unique_pairs], return_counts=True)
    return max(2, int(counts.min()))


def _save_model_artifacts(
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    best_model_name: str,
    results: list[EvaluationResult],
    feature_names: list[str],
) -> None:
    """Sauvegarde le pipeline entraîné, l'encodeur de labels et les métadonnées."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    logger.info("Modèle sauvegardé : %s", MODEL_PATH)
    logger.info("Encodeur de labels sauvegardé : %s", LABEL_ENCODER_PATH)

    best_result = next(r for r in results if r.model_name == best_model_name)
    metadata = {
        "best_model_name": best_model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": feature_names,
        "classes": list(label_encoder.classes_),
        "test_accuracy": best_result.accuracy,
        "test_f1_macro": best_result.f1_macro,
        "cv_mean_accuracy": best_result.cv_mean_accuracy,
        "all_models_compared": [r.model_name for r in results],
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Métadonnées du modèle sauvegardées : %s", MODEL_METADATA_PATH)
