"""
src/ml/models.py
=================

Registre centralisé des modèles de Machine Learning à comparer.

Pourquoi un registre plutôt que d'instancier les modèles directement dans
train.py ? Pour pouvoir ajouter ou retirer un modèle à comparer en modifiant
UNE seule fonction, sans toucher à la logique d'entraînement elle-même
(voir src/training/trainer.py).
"""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.utils.logger import get_logger

logger = get_logger(__name__)

# XGBoost est une dépendance optionnelle : le projet doit rester
# fonctionnel même si elle n'est pas installée (voir requirements.txt).
try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - dépend de l'environnement d'exécution
    _XGBOOST_AVAILABLE = False


def get_candidate_models(random_state: int = 42) -> dict[str, ClassifierMixin]:
    """Retourne le dictionnaire {nom_du_modèle: instance non entraînée}.

    Tous les modèles sont volontairement laissés avec des hyperparamètres
    raisonnables par défaut : l'objectif de cette étape est de COMPARER des
    familles d'algorithmes, pas encore de les régler finement (cela viendra
    dans une étape ultérieure de recherche d'hyperparamètres si besoin).
    """
    models: dict[str, ClassifierMixin] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=random_state),
    }

    if _XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            random_state=random_state,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    else:
        logger.warning(
            "XGBoost n'est pas installé : ce modèle est exclu de la comparaison. "
            "Installe-le avec 'pip install xgboost' pour l'inclure."
        )

    return models
