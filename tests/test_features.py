"""
tests/test_features.py
=======================

Tests unitaires du pipeline de vision par ordinateur : segmentation et
extraction de features (src/cv/segmentation.py, src/cv/features.py).

Plutôt que de tester sur des images générées aléatoirement (shapes.py),
ces tests dessinent des formes géométriques SIMPLES ET CONNUES (un cercle
parfait, un carré parfait, un anneau) pour lesquelles on peut prédire à
l'avance la valeur attendue de chaque feature. C'est ce qui permet de
détecter une régression si le code de calcul change par erreur.

Utilisation :

    python -m unittest discover -s tests
    # ou, si pytest est installé :
    pytest tests/
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.cv.features import extract_all_features
from src.cv.segmentation import SegmentationError, resize_with_letterbox, segment_piece

_SIZE = 300
_BACKGROUND_COLOR = (220, 220, 220)
_PIECE_COLOR = (60, 60, 60)


def _make_image(draw_fn) -> np.ndarray:
    """Crée une image de test : fond clair + forme sombre dessinée par draw_fn."""
    image = np.full((_SIZE, _SIZE, 3), _BACKGROUND_COLOR, dtype=np.uint8)
    draw_fn(image)
    return image


class TestSegmentation(unittest.TestCase):
    """Vérifie que le pipeline de segmentation détecte correctement une pièce."""

    def test_detects_a_simple_circle(self) -> None:
        image = _make_image(lambda img: cv2.circle(img, (150, 150), 80, _PIECE_COLOR, -1))
        result = segment_piece(image, target_size=_SIZE)
        self.assertGreater(cv2.contourArea(result.contour), 0)

    def test_raises_on_empty_image(self) -> None:
        # Une image entièrement uniforme ne contient aucune pièce détectable.
        blank_image = np.full((_SIZE, _SIZE, 3), _BACKGROUND_COLOR, dtype=np.uint8)
        with self.assertRaises(SegmentationError):
            segment_piece(blank_image, target_size=_SIZE)

    def test_letterbox_preserves_aspect_ratio_of_rectangular_photos(self) -> None:
        # Régression : une photo réelle NON carrée (ex: 300x600, ratio 1:2)
        # ne doit pas être déformée par le redimensionnement. Un carré
        # dessiné dans une image rectangulaire doit rester un carré
        # (aspect_ratio proche de 1) après resize_with_letterbox, alors
        # qu'un cv2.resize() carré forcé l'aurait étiré en rectangle.
        rectangular_image = np.full((300, 600, 3), _BACKGROUND_COLOR, dtype=np.uint8)
        cv2.rectangle(rectangular_image, (220, 20), (380, 180), _PIECE_COLOR, -1)  # carré 160x160

        result = resize_with_letterbox(rectangular_image, _SIZE)
        self.assertEqual(result.shape[:2], (_SIZE, _SIZE))

        segmented = segment_piece(rectangular_image, target_size=_SIZE)
        x, y, w, h = cv2.boundingRect(segmented.contour)
        aspect_ratio = w / h
        self.assertAlmostEqual(aspect_ratio, 1.0, delta=0.15)


class TestFeatureExtraction(unittest.TestCase):
    """Vérifie que chaque feature calculée correspond à l'intuition géométrique."""

    def test_circle_has_high_circularity(self) -> None:
        image = _make_image(lambda img: cv2.circle(img, (150, 150), 80, _PIECE_COLOR, -1))
        segmentation = segment_piece(image, target_size=_SIZE)
        features = extract_all_features(segmentation)

        self.assertGreater(features["circularity"], 0.85, "Un cercle doit avoir une circularité proche de 1")
        self.assertGreater(features["solidity"], 0.95, "Un cercle est parfaitement convexe")
        self.assertEqual(features["num_holes"], 0, "Un disque plein n'a aucun trou")

    def test_square_has_high_extent_and_four_corners(self) -> None:
        def draw_square(img: np.ndarray) -> None:
            cv2.rectangle(img, (80, 80), (220, 220), _PIECE_COLOR, -1)

        image = _make_image(draw_square)
        segmentation = segment_piece(image, target_size=_SIZE)
        features = extract_all_features(segmentation)

        self.assertGreater(features["extent"], 0.9, "Un carré doit presque remplir sa boîte englobante")
        self.assertGreaterEqual(features["num_corners"], 4)
        self.assertLessEqual(features["num_corners"], 6)  # tolérance : l'arrondi des coins peut ajouter 1-2 sommets
        # Un carré axis-aligned a un aspect ratio proche de 1.
        self.assertAlmostEqual(features["aspect_ratio"], 1.0, delta=0.15)

    def test_ring_has_exactly_one_hole(self) -> None:
        def draw_ring(img: np.ndarray) -> None:
            cv2.circle(img, (150, 150), 90, _PIECE_COLOR, -1)
            cv2.circle(img, (150, 150), 35, _BACKGROUND_COLOR, -1)  # trou central

        image = _make_image(draw_ring)
        segmentation = segment_piece(image, target_size=_SIZE)
        features = extract_all_features(segmentation)

        self.assertEqual(features["num_holes"], 1, "Un anneau doit avoir exactement un trou détecté")

    def test_l_shape_has_lower_solidity_than_a_square(self) -> None:
        def draw_square(img: np.ndarray) -> None:
            cv2.rectangle(img, (80, 80), (220, 220), _PIECE_COLOR, -1)

        def draw_l_shape(img: np.ndarray) -> None:
            points = np.array([
                [80, 80], [140, 80], [140, 160], [220, 160], [220, 220], [80, 220],
            ], dtype=np.int32)
            cv2.fillPoly(img, [points], _PIECE_COLOR)

        square_features = extract_all_features(segment_piece(_make_image(draw_square), target_size=_SIZE))
        l_shape_features = extract_all_features(segment_piece(_make_image(draw_l_shape), target_size=_SIZE))

        self.assertLess(
            l_shape_features["solidity"], square_features["solidity"],
            "Une forme en L (non convexe) doit avoir une solidité plus faible qu'un carré plein",
        )

    def test_feature_dict_contains_all_expected_keys(self) -> None:
        image = _make_image(lambda img: cv2.circle(img, (150, 150), 80, _PIECE_COLOR, -1))
        segmentation = segment_piece(image, target_size=_SIZE)
        features = extract_all_features(segmentation)

        expected_keys = {
            "area", "perimeter", "bbox_width", "bbox_height", "aspect_ratio",
            "length", "width", "orientation_deg", "circularity", "compactness",
            "extent", "equivalent_diameter", "hull_area", "solidity", "convexity",
            "num_holes", "num_corners", "centroid_x", "centroid_y",
            "hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6", "hu_7",
        }
        self.assertTrue(expected_keys.issubset(features.keys()))


if __name__ == "__main__":
    unittest.main()
