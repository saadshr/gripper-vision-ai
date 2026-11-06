"""Tests réels d'inférence et sécurité, sans entraînement de modèle."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.config import DATASET_DIR, MODEL_PATH, CLASSES
from src.prediction.predictor import GripperPredictor, ModelNotTrainedError
from src.training.trainer import run_training
from src.dataset import build_feature_dataset

class TestPrediction(unittest.TestCase):
    def test_path_and_memory_predictions_agree(self):
        """Le passage en mémoire ne change ni features ni prédiction historique."""
        predictor = GripperPredictor()
        for path in sorted(DATASET_DIR.glob("*/*"))[:2]:
            with self.subTest(photo=path.name):
                image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                a = predictor.predict_from_path(path)
                b = predictor.predict_from_array(image)
                self.assertEqual(a.predicted_class, b.predicted_class)
                self.assertEqual(a.features, b.features)
                self.assertAlmostEqual(a.confidence, b.confidence)
                self.assertEqual(len(a.features), 26)
                self.assertEqual(set(a.class_probabilities), set(CLASSES))
                self.assertAlmostEqual(sum(a.class_probabilities.values()), 1.0)
                self.assertIsInstance(predictor.compatibility_warnings, list)

    def test_training_is_blocked_before_loading_data(self):
        with patch("src.training.trainer._load_dataset", side_effect=AssertionError("lecture interdite")):
            with self.assertRaisesRegex(RuntimeError, "bloquée"):
                run_training()

    def test_feature_export_is_blocked(self):
        with self.assertRaisesRegex(RuntimeError, "bloquée"):
            build_feature_dataset()

    def test_missing_model_has_explicit_error(self):
        predictor = GripperPredictor(model_path=MODEL_PATH.parent / "modele_inexistant.joblib")
        with self.assertRaises(ModelNotTrainedError):
            predictor.load()

if __name__ == "__main__":
    unittest.main()
