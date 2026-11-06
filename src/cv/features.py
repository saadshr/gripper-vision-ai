"""
src/cv/features.py
===================

Calcule l'ensemble des caractéristiques géométriques (features) d'une pièce
à partir du résultat de segmentation (src/cv/segmentation.py).

Chaque fonction `_compute_xxx` calcule UNE feature et est documentée
indépendamment, pour que le lien entre le code et la définition théorique
(voir le guide de cours, chapitre 7) soit direct et vérifiable.

Toutes les features sont finalement regroupées dans un seul dictionnaire
par `extract_all_features()`, prêt à devenir une ligne de DataFrame pandas.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.cv.segmentation import SegmentationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Petite valeur pour éviter les divisions par zéro sans fausser les résultats.
_EPSILON = 1e-6


def _compute_area(contour: np.ndarray) -> float:
    """Aire (Surface) de la pièce, en pixels carrés."""
    return float(cv2.contourArea(contour))


def _compute_perimeter(contour: np.ndarray) -> float:
    """Périmètre du contour externe de la pièce."""
    return float(cv2.arcLength(contour, closed=True))


def _compute_bounding_box(contour: np.ndarray) -> tuple[int, int, int, int]:
    """Boîte englobante alignée sur les axes : (x, y, largeur, hauteur)."""
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h


def _compute_oriented_dimensions(contour: np.ndarray) -> tuple[float, float, float]:
    """Longueur, largeur et orientation réelles de la pièce (boîte orientée).

    Contrairement à la boîte englobante alignée sur les axes, la boîte
    orientée (cv2.minAreaRect) suit l'angle réel de la pièce. On en déduit :
        - la longueur : la plus grande des deux dimensions
        - la largeur  : la plus petite des deux dimensions
        - l'orientation : l'angle du rectangle, en degrés
    """
    (_, _), (dim_a, dim_b), angle = cv2.minAreaRect(contour)
    length = float(max(dim_a, dim_b))
    width = float(min(dim_a, dim_b))
    return length, width, float(angle)


def _compute_circularity(area: float, perimeter: float) -> float:
    """Circularité : 1.0 pour un cercle parfait, tend vers 0 pour une forme allongée."""
    if perimeter < _EPSILON:
        return 0.0
    return float((4 * math.pi * area) / (perimeter ** 2))


def _compute_extent(area: float, bbox_w: int, bbox_h: int) -> float:
    """Rectangularité (extent) : proportion de la boîte englobante remplie par la pièce."""
    bbox_area = bbox_w * bbox_h
    if bbox_area < _EPSILON:
        return 0.0
    return float(area / bbox_area)


def _compute_convex_hull_metrics(contour: np.ndarray, area: float, perimeter: float) -> dict[str, float]:
    """Calcule les métriques liées à l'enveloppe convexe : solidité et convexité.

    - Solidité (solidity) : aire de la pièce / aire de son enveloppe convexe.
      Proche de 1 pour une forme sans concavité.
    - Convexité (convexity) : périmètre de l'enveloppe convexe / périmètre réel.
      Proche de 1 pour une forme déjà quasi-convexe.
    """
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    hull_perimeter = float(cv2.arcLength(hull, closed=True))

    solidity = area / hull_area if hull_area > _EPSILON else 0.0
    convexity = hull_perimeter / perimeter if perimeter > _EPSILON else 0.0

    return {
        "hull_area": hull_area,
        "solidity": float(solidity),
        "convexity": float(convexity),
    }


def _compute_equivalent_diameter(area: float) -> float:
    """Diamètre du cercle qui aurait exactement la même aire que la pièce."""
    return float(math.sqrt((4 * area) / math.pi)) if area > 0 else 0.0


def _compute_compactness(perimeter: float, area: float) -> float:
    """Compacité : mesure normalisée de la complexité du contour par rapport à l'aire."""
    if area < _EPSILON:
        return 0.0
    return float((perimeter ** 2) / area)


def _compute_centroid(contour: np.ndarray) -> tuple[float, float]:
    """Centre de gravité (centroïde), calculé à partir des moments du contour."""
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < _EPSILON:
        # Repli si l'aire des moments est nulle (forme dégénérée).
        x, y, w, h = cv2.boundingRect(contour)
        return x + w / 2, y + h / 2
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return float(cx), float(cy)


def _compute_hu_moments(contour: np.ndarray) -> dict[str, float]:
    """Sept moments de Hu : descripteurs de forme invariants (translation,
    rotation, échelle). On applique une transformation logarithmique, une
    pratique standard, pour ramener ces valeurs (souvent très petites ou
    très grandes) à une échelle numérique exploitable par un modèle ML.
    """
    moments = cv2.moments(contour)
    hu_raw = cv2.HuMoments(moments).flatten()

    hu_log: dict[str, float] = {}
    for i, value in enumerate(hu_raw, start=1):
        if value == 0:
            hu_log[f"hu_{i}"] = 0.0
        else:
            hu_log[f"hu_{i}"] = float(-np.sign(value) * np.log10(np.abs(value)))
    return hu_log


def _compute_num_holes(hierarchy: np.ndarray) -> int:
    """Compte le nombre de contours internes (trous) via la hiérarchie OpenCV.

    Dans la hiérarchie retournée par cv2.findContours (mode RETR_CCOMP),
    la 4e colonne de chaque ligne est l'indice du contour "parent". Un
    contour dont le parent n'est pas -1 est un trou interne à une pièce.
    """
    if hierarchy is None or len(hierarchy) == 0:
        return 0
    return int(sum(1 for entry in hierarchy[0] if entry[3] != -1))


def _compute_num_corners(contour: np.ndarray, perimeter: float) -> int:
    """Approxime le contour par un polygone et compte ses sommets (coins).

    epsilon contrôle la tolérance de simplification : 2% du périmètre est
    une valeur standard qui simplifie le bruit sans perdre les vrais coins.
    """
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    return int(len(approx))


def extract_all_features(segmentation: SegmentationResult) -> dict[str, float]:
    """Calcule toutes les features géométriques d'une pièce segmentée.

    Args:
        segmentation: résultat du pipeline de segmentation (segmentation.py).

    Returns:
        Un dictionnaire {nom_feature: valeur}, directement convertible en
        une ligne de DataFrame pandas avec `pd.DataFrame([features])`.
    """
    contour = segmentation.contour

    area = _compute_area(contour)
    perimeter = _compute_perimeter(contour)
    x, y, bbox_w, bbox_h = _compute_bounding_box(contour)
    length, width, orientation = _compute_oriented_dimensions(contour)
    centroid_x, centroid_y = _compute_centroid(contour)
    hull_metrics = _compute_convex_hull_metrics(contour, area, perimeter)
    hu_moments = _compute_hu_moments(contour)

    features: dict[str, float] = {
        # --- Mesures de base ---
        "area": area,                                              # Area / Surface
        "perimeter": perimeter,                                    # Perimeter
        "bbox_width": float(bbox_w),                                # Bounding Box (largeur)
        "bbox_height": float(bbox_h),                               # Bounding Box (hauteur)
        "aspect_ratio": float(bbox_w) / float(bbox_h) if bbox_h else 0.0,  # Aspect Ratio
        "length": length,                                           # Longueur (orientée)
        "width": width,                                             # Largeur (orientée)
        "orientation_deg": orientation,                             # Orientation

        # --- Descripteurs de forme ---
        "circularity": _compute_circularity(area, perimeter),       # Circularity
        "compactness": _compute_compactness(perimeter, area),       # Compactness
        "extent": _compute_extent(area, bbox_w, bbox_h),            # Extent
        "equivalent_diameter": _compute_equivalent_diameter(area),  # Equivalent Diameter
        **hull_metrics,                                             # Convex Hull, Convexity, Solidity

        # --- Topologie ---
        "num_holes": float(_compute_num_holes(segmentation.all_contours_hierarchy)),  # Nombre de trous
        "num_corners": float(_compute_num_corners(contour, perimeter)),               # Nombre de coins

        # --- Position ---
        "centroid_x": centroid_x,   # Centre de gravité (x)
        "centroid_y": centroid_y,   # Centre de gravité (y)

        # --- Moments de Hu (invariants de forme) ---
        **hu_moments,
    }

    return features
