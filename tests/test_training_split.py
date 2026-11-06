"""
tests/test_training_split.py
=============================

Tests de régression pour le split train/test et la validation croisée
"conscients des groupes" (src/training/trainer.py).

Pourquoi ces tests existent
----------------------------
Avec l'augmentation de données, plusieurs lignes du CSV peuvent provenir
de la MÊME photo réelle (1 originale + N variantes). Un split train/test
ou une validation croisée qui l'ignorerait pourrait placer deux variantes
d'une même photo de part et d'autre d'une frontière train/test, ce qui
gonflerait artificiellement les scores (fuite de données). Ces tests
vérifient explicitement qu'aucune fuite de ce type n'est possible, avec
un jeu de données synthétique où la réponse attendue est connue à l'avance.

Utilisation :

    python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.training.trainer import _group_aware_split, _min_group_count_per_class


def _build_fake_augmented_dataframe(n_sources_per_class: int = 10, n_variants: int = 4) -> pd.DataFrame:
    """Construit un faux dataset avec la même structure qu'un vrai CSV de
    features augmenté : plusieurs lignes (1 originale + N variantes)
    partagent le même `source_image`."""
    rows = []
    for class_name in ["A", "B", "C"]:
        for source_index in range(n_sources_per_class):
            source_name = f"{class_name}_{source_index}"
            # La photo réelle elle-même (is_augmented=False)
            rows.append({"source_image": source_name, "label": class_name, "is_augmented": False})
            # Ses variantes augmentées (is_augmented=True)
            for _ in range(n_variants):
                rows.append({"source_image": source_name, "label": class_name, "is_augmented": True})
    return pd.DataFrame(rows)


class TestGroupAwareSplit(unittest.TestCase):
    def test_no_source_image_overlap_between_train_and_test(self) -> None:
        dataframe = _build_fake_augmented_dataframe()
        y = dataframe["label"].to_numpy()

        train_idx, test_idx = _group_aware_split(dataframe, y, test_size=0.2, random_state=42)

        train_sources = set(dataframe.iloc[train_idx]["source_image"])
        test_sources = set(dataframe.iloc[test_idx]["source_image"])

        self.assertEqual(
            train_sources & test_sources, set(),
            "Une même photo source apparaît des deux côtés du split : risque de fuite de données.",
        )

    def test_test_set_contains_no_augmented_rows(self) -> None:
        dataframe = _build_fake_augmented_dataframe()
        y = dataframe["label"].to_numpy()

        _, test_idx = _group_aware_split(dataframe, y, test_size=0.2, random_state=42)

        self.assertFalse(
            dataframe.iloc[test_idx]["is_augmented"].any(),
            "Le jeu de test contient une variante augmentée — il ne devrait "
            "contenir que de vraies photos jamais vues.",
        )

    def test_split_is_reproducible_with_same_seed(self) -> None:
        dataframe = _build_fake_augmented_dataframe()
        y = dataframe["label"].to_numpy()

        train_idx_1, test_idx_1 = _group_aware_split(dataframe, y, test_size=0.2, random_state=7)
        train_idx_2, test_idx_2 = _group_aware_split(dataframe, y, test_size=0.2, random_state=7)

        np.testing.assert_array_equal(train_idx_1, train_idx_2)
        np.testing.assert_array_equal(test_idx_1, test_idx_2)


class TestGroupAwareCrossValidation(unittest.TestCase):
    def test_stratified_group_kfold_never_splits_a_source_across_folds(self) -> None:
        dataframe = _build_fake_augmented_dataframe()
        y = dataframe["label"].to_numpy()
        groups = dataframe["source_image"].to_numpy()

        n_folds = min(5, _min_group_count_per_class(y, groups))
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

        for train_fold_idx, val_fold_idx in cv.split(np.zeros(len(y)), y, groups):
            train_fold_sources = set(groups[train_fold_idx])
            val_fold_sources = set(groups[val_fold_idx])
            self.assertEqual(
                train_fold_sources & val_fold_sources, set(),
                "Une même photo source apparaît dans le train ET la validation "
                "d'un même fold de validation croisée : risque de fuite de données.",
            )


if __name__ == "__main__":
    unittest.main()
