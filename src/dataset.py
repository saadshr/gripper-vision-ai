"""
extract_features.py
====================

Script d'entrée : parcourt dataset/<classe>/ (formats jpg/jpeg/png/webp),
applique le pipeline de segmentation (src/cv/segmentation.py) puis
d'extraction de features (src/cv/features.py) à chaque image, et sauvegarde
le résultat dans csv/dataset_features.csv.

Génère aussi, pour un sous-échantillon d'images, une visualisation
annotée (contour + boîte englobante + enveloppe convexe) dans le dossier
features/ — bonus "visualisations" demandé dans le cahier des charges.

Utilisation :

    python -m src.dataset
    python -m src.dataset --no-visualizations   # plus rapide
    python -m src.dataset --augmented            # inclut dataset_augmented/
                                                        # (lancer augment_dataset.py avant)
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2
import pandas as pd


from src.config import (  # noqa: E402
    AUGMENTED_DATASET_DIR,
    CLASSES,
    DATASET_CSV_PATH,
    DATASET_DIR,
    FEATURES_DIR,
    ensure_project_dirs,
)
from src.cv.features import extract_all_features  # noqa: E402
from src.cv.segmentation import SegmentationError, segment_piece_from_path  # noqa: E402
from src.cv.visualize import draw_full_overlay  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# Nombre maximum d'images annotées sauvegardées PAR CLASSE dans features/
# (pas besoin d'en générer 120, quelques exemples suffisent à vérifier
# visuellement que la segmentation fonctionne bien).
MAX_VISUALIZATIONS_PER_CLASS = 5

# Formats d'image acceptés dans dataset/. Le dataset synthétique ne générait
# que du .png, mais les vraies photos Eurostyle arrivent dans des formats
# variés (jpg, jpeg, webp) — les ignorer silencieusement ferait perdre la
# quasi-totalité du vrai dataset.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

# Reconnaît le suffixe "_aug3" ajouté par augment_dataset.py, pour retrouver
# le nom de la photo réelle d'origine à partir du nom d'une variante augmentée.
_AUGMENTED_SUFFIX_PATTERN = re.compile(r"_aug\d+$")


def _source_image_stem(stem: str) -> str:
    """Retrouve le nom de la photo source à partir du nom d'un fichier,
    qu'il s'agisse d'une vraie photo (renvoyée telle quelle) ou d'une
    variante augmentée (ex: 'piece_003_aug2' -> 'piece_003')."""
    return _AUGMENTED_SUFFIX_PATTERN.sub("", stem)


def _process_class_directory(
    class_dir: Path, class_name: str, *, is_augmented: bool,
    save_visualizations: bool, n_visualized_so_far: int,
) -> tuple[list[dict], int, int]:
    """Traite toutes les images d'un dossier de classe (réel ou augmenté).

    Returns:
        (lignes de features extraites, nombre d'erreurs, nombre de
        visualisations sauvegardées lors de cet appel).
    """
    rows: list[dict] = []
    n_errors = 0
    n_visualized = n_visualized_so_far

    image_paths = sorted(
        p for ext in IMAGE_EXTENSIONS for p in class_dir.glob(f"*{ext}")
    )

    for image_path in image_paths:
        try:
            segmentation = segment_piece_from_path(image_path)
            features = extract_all_features(segmentation)
            features["filename"] = image_path.name
            features["label"] = class_name
            features["source_image"] = _source_image_stem(image_path.stem)
            features["is_augmented"] = is_augmented
            rows.append(features)

            if save_visualizations and not is_augmented and n_visualized < MAX_VISUALIZATIONS_PER_CLASS:
                overlay = draw_full_overlay(segmentation)
                out_path = FEATURES_DIR / f"{class_name}_{image_path.stem}_overlay.png"
                cv2.imwrite(str(out_path), overlay)
                n_visualized += 1

        except SegmentationError as error:
            n_errors += 1
            logger.error("Segmentation impossible pour %s : %s", image_path.name, error)
        except Exception as error:  # noqa: BLE001 - on ne veut jamais arrêter toute la boucle
            n_errors += 1
            logger.error("Erreur inattendue pour %s : %s", image_path.name, error, exc_info=True)

    return rows, n_errors, n_visualized


def build_feature_dataset(
    *, save_visualizations: bool = True, include_augmented: bool = False
) -> pd.DataFrame:
    """Construit le dataset de features à partir de toutes les images.

    Args:
        save_visualizations: si True, sauvegarde quelques images annotées
            par classe dans features/, pour un contrôle visuel rapide.
            Ne s'applique qu'aux vraies photos (jamais aux variantes
            augmentées, qui ne présentent pas d'intérêt visuel propre).
        include_augmented: si True, inclut aussi les variantes de
            dataset_augmented/ (voir augment_dataset.py) en plus des vraies
            photos de dataset/. Chaque ligne du CSV reste traçable via les
            colonnes `source_image` (photo réelle d'origine) et
            `is_augmented`, utilisées par trainer.py pour éviter toute
            fuite de données entre variantes d'une même photo lors du
            split train/test.

    Returns:
        Le DataFrame pandas complet (aussi sauvegardé en CSV sur disque).
    """
    from src.safety import refuser_action
    refuser_action('build_feature_dataset')
    ensure_project_dirs()

    all_rows: list[dict] = []
    total_errors = 0
    start_time = time.time()

    for class_name in CLASSES:
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            logger.warning("Dossier de classe manquant : %s (ignoré)", class_dir)
            continue

        real_rows, real_errors, n_visualized = _process_class_directory(
            class_dir, class_name, is_augmented=False,
            save_visualizations=save_visualizations, n_visualized_so_far=0,
        )
        all_rows.extend(real_rows)
        total_errors += real_errors
        log_msg = f"Traitement de la classe '{class_name}' : {len(real_rows)} vraies photos"

        if include_augmented:
            augmented_class_dir = AUGMENTED_DATASET_DIR / class_name
            if augmented_class_dir.exists():
                aug_rows, aug_errors, _ = _process_class_directory(
                    augmented_class_dir, class_name, is_augmented=True,
                    save_visualizations=save_visualizations, n_visualized_so_far=n_visualized,
                )
                all_rows.extend(aug_rows)
                total_errors += aug_errors
                log_msg += f" + {len(aug_rows)} variantes augmentées"
            else:
                logger.warning(
                    "include_augmented=True mais %s n'existe pas — lance "
                    "'python augment_dataset.py' d'abord. Classe traitée sans augmentation.",
                    augmented_class_dir,
                )

        logger.info(log_msg)

    if not all_rows:
        raise RuntimeError(
            "Aucune feature n'a pu être extraite. Le dossier dataset/ contient-il "
            "bien des photos, organisées en sous-dossiers dataset/<Classe>/ ?"
        )

    dataframe = pd.DataFrame(all_rows)

    # Colonnes d'identification/traçabilité en premier, pour la lisibilité du CSV.
    id_columns = ["filename", "label", "source_image", "is_augmented"]
    ordered_columns = id_columns + [c for c in dataframe.columns if c not in id_columns]
    dataframe = dataframe[ordered_columns]

    DATASET_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(DATASET_CSV_PATH, index=False)

    elapsed = time.time() - start_time
    n_real = int((~dataframe["is_augmented"]).sum())
    n_aug = int(dataframe["is_augmented"].sum())
    logger.info(
        "Extraction terminée : %d lignes (%d réelles + %d augmentées), %d colonnes de "
        "features, %d erreurs, en %.1fs.",
        len(dataframe), n_real, n_aug, len(dataframe.columns) - len(id_columns), total_errors, elapsed,
    )
    logger.info("CSV sauvegardé : %s", DATASET_CSV_PATH)

    return dataframe


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrait les features de toutes les images du dataset.")
    parser.add_argument(
        "--no-visualizations", action="store_true",
        help="Désactive la sauvegarde des images annotées dans features/ (plus rapide).",
    )
    parser.add_argument(
        "--augmented", action="store_true",
        help="Inclut aussi les variantes de dataset_augmented/ (lancer augment_dataset.py avant).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_feature_dataset(
        save_visualizations=not args.no_visualizations,
        include_augmented=args.augmented,
    )


if __name__ == "__main__":
    main()
