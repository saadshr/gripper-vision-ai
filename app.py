"""Seule interface utilisateur : Streamlit, sans génération ni entraînement automatique."""
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    APP_TITLE, CLASS_DISPLAY_NAMES, CLASSES, DATASET_DIR, HISTORY_DIR,
    IMAGE_EXTENSIONS, MODEL_METADATA_PATH,
)
from src.cv.segmentation import SegmentationError
from src.cv.visualize import draw_full_overlay
from src.prediction.predictor import GripperPredictor, ModelNotTrainedError

@st.cache_resource(show_spinner=False)
def load_predictor() -> GripperPredictor:
    """Charge une fois le modèle historique ; ne fait jamais de fit."""
    predictor = GripperPredictor()
    predictor.load()
    return predictor

def photo_paths() -> list[Path]:
    """Retourne uniquement les photos classées, sans les images incertaines."""
    return sorted(p for p in DATASET_DIR.glob("*/*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

def show_prediction(image: np.ndarray) -> None:
    """Affiche la segmentation, la classe, les scores et les caractéristiques."""
    with st.spinner("Analyse de la photo…"):
        predictor = load_predictor()
        result = predictor.predict_from_array(image)
    for message in predictor.compatibility_warnings:
        st.warning(message)
    left, right = st.columns(2)
    left.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Photo analysée", use_column_width=True)
    overlay = draw_full_overlay(result.segmentation)
    right.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Segmentation à vérifier visuellement", use_column_width=True)
    st.subheader(result.predicted_class_display)
    st.metric("Score du modèle pour cette classe", f"{result.confidence:.0%}")
    st.caption("Ce score n'est pas une probabilité validée de réussite mécanique. Une validation humaine reste nécessaire.")
    if result.is_low_confidence:
        st.warning("Score inférieur au seuil indicatif de 60 % : recommandation incertaine.")
    scores = pd.DataFrame([
        {"Préhenseur": CLASS_DISPLAY_NAMES.get(name, name), "Score": value}
        for name, value in result.class_probabilities.items()
    ]).set_index("Préhenseur")
    st.bar_chart(scores)
    with st.expander("Les 26 caractéristiques extraites"):
        st.dataframe(pd.DataFrame(result.features.items(), columns=["Caractéristique", "Valeur"]), hide_index=True)

def prediction_page() -> None:
    """Choisit une photo importée ou historique ; aucune écriture d'upload."""
    source = st.radio("Source de la photo", ["Importer une photo", "Essayer une photo du dataset"])
    payload = None
    if source == "Importer une photo":
        uploaded = st.file_uploader("Une pièce, sur un fond contrasté", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            if uploaded.size > 10 * 1024 * 1024:
                st.error("La photo dépasse la limite de 10 Mo.")
                return
            payload = uploaded.getvalue()
    else:
        st.info("Ces photos appartiennent aux données historiques déjà vues par le modèle. Cet essai n'est pas un test indépendant.")
        paths = photo_paths()
        if not paths:
            st.error("Aucune photo disponible dans data/photos/.")
            return
        selected = st.selectbox("Photo du dataset", paths, format_func=lambda p: p.relative_to(DATASET_DIR).as_posix())
        payload = selected.read_bytes()
    if st.button("Analyser la photo", disabled=payload is None, type="primary"):
        try:
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Image illisible ou format non reconnu.")
            show_prediction(image)
        except (SegmentationError, ModelNotTrainedError, ValueError, OSError) as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Analyse impossible ({type(error).__name__}) : {error}. Vérifier les modèles et dépendances indiqués dans README.md.")

def history_page() -> None:
    """Consulte des résultats enregistrés, sans recalculer de métrique."""
    rows = [{"Classe": name, "Photos": sum(p.parent.name == name for p in photo_paths())} for name in CLASSES]
    st.subheader("Dataset photographique")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("125 photos actives ; 8 images incertaines à part ; aucune image synthétique utilisée ici.")
    st.subheader("Expérience historique — résultats non recalculés")
    if MODEL_METADATA_PATH.exists():
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        st.write(f"Modèle sélectionné : {metadata['best_model_name']}")
        a, b = st.columns(2)
        a.metric("Accuracy test historique", f"{metadata['test_accuracy']:.0%}")
        b.metric("Accuracy CV historique", f"{metadata['cv_mean_accuracy']:.0%}")
    comparison = HISTORY_DIR / "model_comparison.csv"
    if comparison.exists():
        st.dataframe(pd.read_csv(comparison), hide_index=True)
    st.warning("Ces scores concernent un ancien test de 25 photos et ne valident pas un usage industriel. Le modèle sauvegardé a ensuite été ajusté sur toutes les données. Le meilleur score test du tableau n'est pas nécessairement celui du modèle sélectionné par CV.")

def main() -> None:
    st.set_page_config(page_title="Gripper Vision AI", page_icon="🔍", layout="wide")
    st.title(APP_TITLE)
    st.caption("Prototype pédagogique d'aide au choix d'un préhenseur pour pièces plastiques.")
    page = st.sidebar.radio("Navigation", ["Prédiction", "Données et résultats", "À propos"])
    st.sidebar.info("Prédiction uniquement. Aucun entraînement automatique.")
    if page == "Prédiction":
        prediction_page()
    elif page == "Données et résultats":
        history_page()
    else:
        st.write("Photo → segmentation OpenCV → 26 caractéristiques géométriques → modèle RandomForest historique → proposition à vérifier.")
        st.write("Les images, labels et modèles gardent leur provenance. La masse, la matière et les contraintes du poste ne sont pas encore intégrées.")
        st.warning("La silhouette ne suffit pas à garantir une prise. Ce prototype ne commande aucun robot.")

if __name__ == "__main__":
    main()
