"""Chemins et constantes du projet ; aucun dossier créé à l'import."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "data" / "photos"
AUGMENTED_DATASET_DIR = ROOT_DIR / "data" / "augmented"
MODELS_DIR = ROOT_DIR / "models"
HISTORY_DIR = ROOT_DIR / "results" / "history"
# Ces sorties sont réservées à une prochaine phase autorisée.
PREPARATION_DIR = ROOT_DIR / "results" / "preparation"
FEATURES_DIR = PREPARATION_DIR / "overlays"
DATASET_CSV_PATH = PREPARATION_DIR / "dataset_features.csv"
OUTPUTS_GRAPHS_DIR = PREPARATION_DIR / "graphs"
OUTPUTS_REPORTS_DIR = PREPARATION_DIR / "reports"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

CLASSES = ("Vacuum", "Parallel", "Angular", "ThreeFinger", "Soft", "Internal")
CLASS_DISPLAY_NAMES = {
    "Vacuum": "Ventouse", "Parallel": "Pince parallèle",
    "Angular": "Pince angulaire", "ThreeFinger": "Pince à trois doigts",
    "Soft": "Préhenseur souple", "Internal": "Préhenseur interne",
}
APP_TITLE = "Gripper Vision AI — Eurostyle Systems Tanger"
IMAGE_SIZE = 400
BLUR_KERNEL_SIZE = (5, 5)
MORPH_KERNEL_SIZE = (5, 5)
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
NON_FEATURE_COLUMNS = ("filename", "label", "source_image", "is_augmented")
CONFIDENCE_WARNING_THRESHOLD = 0.6
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def ensure_project_dirs() -> None:
    """Crée uniquement des sorties dérivées après autorisation de préparation."""
    from src.safety import refuser_action
    refuser_action("création de sorties de préparation")
