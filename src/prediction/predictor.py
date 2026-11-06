"""
src/prediction/predictor.py
============================

Pipeline de prédiction bout-en-bout : prend le chemin d'une image et
retourne le gripper recommandé, avec son score de confiance et le détail
des probabilités par classe.

Ce module réutilise EXACTEMENT le même pipeline de segmentation et
d'extraction de features que celui utilisé pour construire le dataset
d'entraînement (src/cv/segmentation.py et src/cv/features.py) — c'est une
règle d'or en Machine Learning appliqué aux images (voir le guide de
cours, chapitre 10) : ce qui a été fait à l'entraînement doit être refait
à l'identique en prédiction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import InconsistentVersionWarning

from src.config import CONFIDENCE_WARNING_THRESHOLD, LABEL_ENCODER_PATH, MODEL_PATH
from src.cv.features import extract_all_features
from src.cv.segmentation import SegmentationResult, segment_piece, segment_piece_from_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelNotTrainedError(RuntimeError):
    """Levée quand on tente de prédire alors qu'aucun modèle n'a été entraîné."""


@dataclass
class PredictionResult:
    """Regroupe tout ce dont l'interface graphique a besoin pour afficher
    le résultat d'une prédiction : classe prédite, confiance, détail des
    probabilités, features calculées, et le résultat de segmentation (pour
    dessiner le contour / la boîte englobante / l'enveloppe convexe).
    """

    predicted_class: str
    predicted_class_display: str
    confidence: float
    class_probabilities: dict[str, float]
    features: dict[str, float]
    segmentation: SegmentationResult
    is_low_confidence: bool


class GripperPredictor:
    """Encapsule le modèle entraîné et expose une méthode `predict()` simple.

    Utiliser une classe plutôt que des fonctions isolées permet de charger
    le modèle UNE SEULE FOIS (opération relativement coûteuse) puis de
    l'utiliser pour de nombreuses prédictions successives — typiquement
    l'usage qu'en fera l'interface graphique.
    """

    def __init__(self, model_path: Path = MODEL_PATH, encoder_path: Path = LABEL_ENCODER_PATH) -> None:
        self._model_path = model_path
        self._encoder_path = encoder_path
        self._pipeline: Pipeline | None = None
        self._label_encoder: LabelEncoder | None = None
        self.compatibility_warnings: list[str] = []

    @property
    def is_loaded(self) -> bool:
        """Indique si un modèle est actuellement chargé en mémoire."""
        return self._pipeline is not None and self._label_encoder is not None

    def load(self) -> None:
        """Charge le modèle et l'encodeur de labels depuis le disque.

        Raises:
            ModelNotTrainedError: si les fichiers du modèle n'existent pas
                encore. Restaurer les artefacts historiques, sans lancer de fit.
        """
        if not self._model_path.exists() or not self._encoder_path.exists():
            raise ModelNotTrainedError(
                "Aucun modèle entraîné n'a été trouvé. "
                "Les modèles historiques doivent être présents dans models/. "
                "Ne pas lancer un entraînement sans autorisation."
            )
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", InconsistentVersionWarning)
            self._pipeline = joblib.load(self._model_path)
            self._label_encoder = joblib.load(self._encoder_path)
        self.compatibility_warnings = list(dict.fromkeys(
            f"Modèle sérialisé avec scikit-learn {item.message.original_sklearn_version} ; "
            f"version installée {item.message.current_sklearn_version}. "
            "La compatibilité et les résultats numériques ne sont pas garantis."
            for item in recorded if isinstance(item.message, InconsistentVersionWarning)
        ))
        # Les avertissements de la bibliothèque restent visibles dans les tests/console.
        for item in recorded:
            warnings.warn(item.message, item.category, stacklevel=2)
        logger.info("Modèle chargé depuis %s", self._model_path)

    def predict_from_path(self, image_path: str | Path) -> PredictionResult:
        """Exécute le pipeline complet : image -> segmentation -> features -> prédiction.

        Args:
            image_path: chemin vers l'image à analyser.

        Returns:
            Un PredictionResult prêt à être affiché par l'interface graphique.
        """
        return self._predict_segmentation(segment_piece_from_path(image_path), Path(image_path).name)

    def predict_from_array(self, image: np.ndarray) -> PredictionResult:
        """Analyse une image BGR OpenCV en mémoire, sans créer d'upload sur disque."""
        return self._predict_segmentation(segment_piece(image), "image importée")

    def _predict_segmentation(self, segmentation: SegmentationResult, image_name: str) -> PredictionResult:
        """Réutilise à l'identique les features et le modèle historiques."""
        if not self.is_loaded:
            self.load()
        assert self._pipeline is not None and self._label_encoder is not None
        features = extract_all_features(segmentation)

        # Le pipeline a été entraîné sur un DataFrame avec des colonnes
        # précises, dans un ordre précis : on reconstruit exactement la
        # même structure pour une seule ligne.
        feature_row = pd.DataFrame([features])[self._pipeline.feature_names_in_]

        predicted_index = int(self._pipeline.predict(feature_row)[0])
        probabilities = self._pipeline.predict_proba(feature_row)[0]

        class_names = list(self._label_encoder.classes_)
        class_probabilities = {
            class_name: float(proba) for class_name, proba in zip(class_names, probabilities)
        }

        predicted_class = self._label_encoder.inverse_transform([predicted_index])[0]
        confidence = float(max(probabilities))

        from src.config import CLASS_DISPLAY_NAMES  # import local : évite un cycle avec config au chargement du module

        result = PredictionResult(
            predicted_class=predicted_class,
            predicted_class_display=CLASS_DISPLAY_NAMES.get(predicted_class, predicted_class),
            confidence=confidence,
            class_probabilities=class_probabilities,
            features=features,
            segmentation=segmentation,
            is_low_confidence=confidence < CONFIDENCE_WARNING_THRESHOLD,
        )

        logger.info(
            "Prédiction : %s (confiance=%.1f%%) pour l'image %s",
            result.predicted_class_display, confidence * 100, image_name,
        )
        if result.is_low_confidence:
            logger.warning(
                "Confiance faible (%.1f%%) pour l'image %s — vérification humaine recommandée.",
                confidence * 100, image_name,
            )

        return result
