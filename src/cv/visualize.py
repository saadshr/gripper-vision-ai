"""
src/cv/visualize.py
====================

Fonctions de dessin (overlays) utilisées à deux endroits du projet :
    - extract_features.py : sauvegarde une image annotée par pièce dans
      features/ (bonus "visualisations").
    - l'interface graphique (src/gui/app_gui.py) : affiche le contour, la
      boîte englobante et l'enveloppe convexe détectés sur l'image importée
      par l'utilisateur.

Toutes les fonctions travaillent sur une COPIE de l'image d'entrée : elles
ne modifient jamais l'image originale (bonne pratique pour éviter les effets
de bord difficiles à déboguer).
"""

from __future__ import annotations

import cv2
import numpy as np

from src.cv.segmentation import SegmentationResult

# Palette de couleurs BGR utilisée pour toutes les annotations du projet.
COLOR_CONTOUR = (0, 200, 0)       # vert : contour détecté
COLOR_BBOX = (0, 140, 255)        # orange : boîte englobante
COLOR_HULL = (255, 90, 0)         # bleu : enveloppe convexe
COLOR_CENTROID = (0, 0, 255)      # rouge : centre de gravité


def draw_contour_overlay(segmentation: SegmentationResult) -> np.ndarray:
    """Dessine le contour détecté sur une copie de l'image originale."""
    image = segmentation.original_image.copy()
    cv2.drawContours(image, [segmentation.contour], -1, COLOR_CONTOUR, 2)
    return image


def draw_full_overlay(segmentation: SegmentationResult) -> np.ndarray:
    """Dessine, sur une seule image, le contour, la boîte englobante,
    l'enveloppe convexe et le centre de gravité — pratique pour l'onglet
    "Voir le contour détecté" de l'interface graphique.
    """
    image = segmentation.original_image.copy()
    contour = segmentation.contour

    # Contour réel de la pièce
    cv2.drawContours(image, [contour], -1, COLOR_CONTOUR, 2)

    # Boîte englobante (axis-aligned)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), COLOR_BBOX, 2)

    # Enveloppe convexe
    hull = cv2.convexHull(contour)
    cv2.drawContours(image, [hull], -1, COLOR_HULL, 2)

    # Centre de gravité
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-6:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        cv2.circle(image, (cx, cy), 5, COLOR_CENTROID, thickness=-1)

    return image


def draw_binary_mask_preview(segmentation: SegmentationResult) -> np.ndarray:
    """Retourne le masque binaire converti en image BGR (pour l'affichage)."""
    return cv2.cvtColor(segmentation.binary_mask, cv2.COLOR_GRAY2BGR)
