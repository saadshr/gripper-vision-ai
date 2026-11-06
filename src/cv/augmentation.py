"""
src/cv/augmentation.py
=======================

Augmentation de données pour le dataset RÉEL.

Pourquoi l'augmentation de données ?
-------------------------------------
Avec seulement ~20 photos par classe, le modèle a très peu d'exemples pour
apprendre à généraliser. L'augmentation de données génère automatiquement
des variantes plausibles de chaque photo réelle (légère rotation, miroir,
luminosité/contraste différents, léger bruit de capteur) : le modèle voit
ainsi plusieurs "versions" de la même pièce, ce qui l'aide à apprendre les
caractéristiques géométriques réellement invariantes plutôt que des
particularités d'une seule photo (angle exact, éclairage exact...).

Ce que l'augmentation ne fait PAS
-----------------------------------
Elle n'invente pas de nouvelles pièces ni de nouvelles formes : chaque
variante reste dérivée d'une vraie photo. Elle ne remplace donc pas le
besoin de vraies photos supplémentaires et diversifiées — voir la mise en
garde dans README.md.

Point méthodologique important (fuite de données)
-----------------------------------------------------
Les variantes augmentées d'une même photo se ressemblent beaucoup. Si une
variante finit dans le jeu d'entraînement et une autre variante de LA MÊME
PHOTO finit dans le jeu de test, le modèle "reconnaîtrait" en partie une
image qu'il a déjà vue sous une autre forme — le score de test serait
alors artificiellement gonflé (fuite de données). C'est pourquoi
`src/training/trainer.py` regroupe explicitement toutes les variantes
d'une même photo source du même côté du split train/test (voir la colonne
`source_image` du CSV de features).
"""

from __future__ import annotations

import cv2
import numpy as np


def _estimate_border_color(image: np.ndarray) -> tuple[int, int, int]:
    """Estime la couleur de fond à partir des 4 coins de l'image (même
    technique que resize_with_letterbox dans segmentation.py), pour remplir
    les zones vides créées par une rotation sans introduire de couleur
    arbitraire qui perturberait le seuillage Otsu."""
    corner_pixels = np.concatenate(
        [
            image[0:10, 0:10].reshape(-1, 3),
            image[0:10, -10:].reshape(-1, 3),
            image[-10:, 0:10].reshape(-1, 3),
            image[-10:, -10:].reshape(-1, 3),
        ],
        axis=0,
    )
    return tuple(int(v) for v in np.median(corner_pixels, axis=0))


def _random_rotation(image: np.ndarray, rng: np.random.Generator, max_angle: float = 25.0) -> np.ndarray:
    """Applique une rotation d'angle aléatoire autour du centre de l'image.

    L'angle est volontairement limité (+/- max_angle degrés) : une rotation
    trop importante changerait radicalement le cadrage de la pièce et
    risquerait de la faire sortir du champ. On simule ainsi une variation
    réaliste d'angle de prise de vue, pas une rotation arbitraire à 360°.
    """
    h, w = image.shape[:2]
    angle = rng.uniform(-max_angle, max_angle)
    center = (w / 2, h / 2)
    border_color = _estimate_border_color(image)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    return cv2.warpAffine(
        image, rotation_matrix, (w, h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=border_color,
    )


def _random_horizontal_flip(image: np.ndarray, rng: np.random.Generator, probability: float = 0.5) -> np.ndarray:
    """Retourne l'image horizontalement une fois sur deux en moyenne.

    Pertinent ici car la plupart des pièces plastiques n'ont pas de
    "sens" fonctionnel imposé sur la photo — leur image miroir reste une
    vue plausible de la même pièce.
    """
    if rng.random() < probability:
        return cv2.flip(image, 1)
    return image


def _random_brightness_contrast(
    image: np.ndarray, rng: np.random.Generator,
    brightness_range: float = 25.0, contrast_range: float = 0.25,
) -> np.ndarray:
    """Ajuste légèrement la luminosité et le contraste, pour simuler un
    éclairage différent d'une prise de vue à l'autre."""
    brightness = rng.uniform(-brightness_range, brightness_range)
    contrast = 1.0 + rng.uniform(-contrast_range, contrast_range)
    adjusted = image.astype(np.float32) * contrast + brightness
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _random_gaussian_noise(image: np.ndarray, rng: np.random.Generator, sigma: float = 6.0) -> np.ndarray:
    """Ajoute un léger bruit gaussien, pour simuler le bruit de capteur
    d'un appareil photo réel (utile car nos photos sources sont souvent
    des images produit très "propres")."""
    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _random_scale(image: np.ndarray, rng: np.random.Generator, scale_range: float = 0.12) -> np.ndarray:
    """Zoom léger avant/arrière autour du centre, pour simuler une distance
    de prise de vue légèrement différente. La taille de l'image est
    conservée (recadrage ou complément par la couleur de fond estimée)."""
    h, w = image.shape[:2]
    scale = 1.0 + rng.uniform(-scale_range, scale_range)
    border_color = _estimate_border_color(image)

    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), border_color, dtype=np.uint8)

    if scale >= 1.0:
        # Image agrandie : on recadre le centre à la taille d'origine.
        rh, rw = resized.shape[:2]
        y0, x0 = (rh - h) // 2, (rw - w) // 2
        canvas = resized[y0:y0 + h, x0:x0 + w]
    else:
        # Image réduite : on la colle centrée sur un fond de la couleur estimée.
        rh, rw = resized.shape[:2]
        y0, x0 = (h - rh) // 2, (w - rw) // 2
        canvas[y0:y0 + rh, x0:x0 + rw] = resized

    return canvas


def augment_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Applique une combinaison aléatoire de transformations à une image.

    Chaque appel avec un `rng` différent produit une variante différente
    et reproductible (même seed -> mêmes variantes).

    Args:
        image: image couleur BGR d'origine (telle que retournée par cv2.imread).
        rng: générateur aléatoire NumPy (pour la reproductibilité).

    Returns:
        Une nouvelle image, de mêmes dimensions que l'originale.
    """
    result = image.copy()
    result = _random_rotation(result, rng)
    result = _random_horizontal_flip(result, rng)
    result = _random_scale(result, rng)
    result = _random_brightness_contrast(result, rng)
    result = _random_gaussian_noise(result, rng)
    return result


def generate_augmented_variants(
    image: np.ndarray, n_variants: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Génère `n_variants` variantes augmentées distinctes d'une image.

    Args:
        image: image couleur BGR d'origine.
        n_variants: nombre de variantes à générer.
        rng: générateur aléatoire NumPy partagé (chaque variante utilise
            un tirage différent du même générateur, donc reproductible).

    Returns:
        Une liste de `n_variants` images augmentées.
    """
    return [augment_image(image, rng) for _ in range(n_variants)]
