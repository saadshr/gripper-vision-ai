"""
src/cv/segmentation.py
=======================

Pipeline de segmentation OpenCV, utilisé à la fois :
    - lors de la construction du dataset (extract_features.py)
    - lors d'une prédiction sur une nouvelle image (predict.py / GUI)

C'est une règle d'or en Machine Learning appliqué aux images : la
photo d'entraînement et la photo à prédire DOIVENT traverser exactement
le même traitement. En centralisant ce pipeline ici, on garantit qu'il
n'existe qu'une seule version du code de segmentation dans tout le projet.

Pipeline (voir aussi le guide de cours, chapitre 6) :

    Chargement -> Resize -> Grayscale -> Blur -> Threshold (Otsu)
                -> Morphology -> Contours -> plus grand contour retenu
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import BLUR_KERNEL_SIZE, IMAGE_SIZE, MORPH_KERNEL_SIZE
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SegmentationError(RuntimeError):
    """Levée quand aucune pièce n'a pu être détectée dans une image."""


@dataclass
class SegmentationResult:
    """Regroupe tous les objets intermédiaires produits par la segmentation.

    Conserver ces objets intermédiaires (et pas seulement le contour final)
    est très utile pour le débogage et pour les visualisations du chapitre
    "visualisations" (contour dessiné, masque, boîte englobante...).
    """

    original_image: np.ndarray      # image couleur BGR redimensionnée
    gray_image: np.ndarray          # niveaux de gris
    binary_mask: np.ndarray         # masque binaire final (0/255)
    contour: np.ndarray             # contour externe de la pièce (le plus grand)
    all_contours_hierarchy: np.ndarray  # hiérarchie complète (pour compter les trous)
    all_contours: list[np.ndarray]      # tous les contours (externe + trous)


def load_image(image_path: str | Path) -> np.ndarray:
    """Charge une image depuis le disque et vérifie qu'elle est valide.

    Args:
        image_path: chemin vers le fichier image (jpg, png...).

    Raises:
        FileNotFoundError: si le fichier n'existe pas ou n'est pas une image
            lisible par OpenCV (cv2.imread renvoie None silencieusement,
            c'est un piège classique qu'on intercepte explicitement ici).
    """
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(
            f"Impossible de lire l'image : {image_path}. "
            "Vérifie le chemin et le format du fichier."
        )
    return image


def resize_with_letterbox(
    image: np.ndarray, target_size: int
) -> np.ndarray:
    """Redimensionne une image à `target_size x target_size` SANS déformer la pièce.

    Pourquoi cette fonction est nécessaire
    ----------------------------------------
    Notre dataset synthétique est généré directement en 400x400 : un simple
    `cv2.resize(image, (target_size, target_size))` ne posait donc aucun
    problème (l'image était déjà carrée). Mais une vraie photo (téléphone,
    catalogue fournisseur...) est presque toujours RECTANGULAIRE. Forcer un
    redimensionnement carré étire ou compresse la pièce elle-même :
    un rond photographié dans un cadre 4:3 devient une ellipse après un
    resize carré brutal, ce qui fausse silencieusement `aspect_ratio`,
    `circularity`, `orientation_deg`, etc.

    La solution standard en vision par ordinateur est le "letterboxing" :
    1. On redimensionne l'image en conservant ses proportions d'origine,
       de sorte que sa plus grande dimension atteigne `target_size`.
    2. On colle cette image (centrée) sur un canevas carré `target_size x
       target_size`, complété par une couleur de bordure neutre.

    La couleur de bordure est estimée à partir des 4 coins de l'image
    d'origine (là où se trouve presque toujours le fond, pas la pièce),
    plutôt qu'une couleur arbitraire (blanc/noir) qui pourrait elle-même
    perturber le seuillage Otsu si elle tranche avec le vrai fond de la
    photo.

    Args:
        image: image couleur BGR d'origine, de dimensions quelconques.
        target_size: taille du canevas carré final.

    Returns:
        Image carrée `target_size x target_size`, pièce non déformée.
    """
    h, w = image.shape[:2]

    # Cas déjà carré (notre dataset synthétique) : on garde l'ancien
    # comportement, plus rapide et strictement identique à avant.
    if h == w:
        return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # 1. Redimensionnement proportionnel : la plus grande dimension atteint target_size.
    scale = target_size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2. Couleur de bordure = médiane des pixels des 4 coins (estimation du fond réel).
    corner_pixels = np.concatenate(
        [
            image[0:10, 0:10].reshape(-1, 3),
            image[0:10, -10:].reshape(-1, 3),
            image[-10:, 0:10].reshape(-1, 3),
            image[-10:, -10:].reshape(-1, 3),
        ],
        axis=0,
    )
    border_color = tuple(int(v) for v in np.median(corner_pixels, axis=0))

    # 3. Canevas carré rempli de cette couleur de fond, image collée au centre.
    canvas = np.full((target_size, target_size, 3), border_color, dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


def segment_piece(image: np.ndarray, *, target_size: int = IMAGE_SIZE) -> SegmentationResult:
    """Exécute le pipeline complet de segmentation sur une image déjà chargée.

    Args:
        image: image couleur BGR (telle que retournée par cv2.imread).
        target_size: taille à laquelle l'image est redimensionnée avant
            traitement, pour que toutes les images du pipeline (entraînement
            et prédiction) soient traitées à la même échelle.

    Returns:
        Un SegmentationResult contenant tous les objets intermédiaires.

    Raises:
        SegmentationError: si aucun contour exploitable n'a été détecté
            (image vide, entièrement uniforme, etc.).
    """
    # --- Resize : uniformise la résolution de traitement SANS déformer la pièce ---
    resized = resize_with_letterbox(image, target_size)

    # --- Grayscale : simplifie la détection de forme (voir guide, ch.5) ---
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # --- Blur : réduit le bruit avant seuillage ---
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)

    # --- Threshold (Otsu) : sépare automatiquement la pièce du fond ---
    # THRESH_BINARY_INV car nos pièces sont plus sombres que le fond
    # (voir rendering.py) -> on veut que la pièce devienne blanche (255).
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --- Morphology : nettoie le bruit résiduel ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # --- Contours (avec hiérarchie, pour pouvoir compter les trous) ---
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise SegmentationError(
            "Aucun contour détecté : vérifie le contraste entre la pièce et le fond."
        )

    # On retient comme "la pièce" le contour de plus grande aire.
    main_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(main_contour) < 50:
        raise SegmentationError(
            "Le plus grand contour détecté est trop petit pour être une pièce valide."
        )

    return SegmentationResult(
        original_image=resized,
        gray_image=gray,
        binary_mask=binary,
        contour=main_contour,
        all_contours_hierarchy=hierarchy,
        all_contours=list(contours),
    )


def segment_piece_from_path(image_path: str | Path, *, target_size: int = IMAGE_SIZE) -> SegmentationResult:
    """Raccourci pratique : charge une image depuis un chemin puis la segmente."""
    image = load_image(image_path)
    return segment_piece(image, target_size=target_size)
