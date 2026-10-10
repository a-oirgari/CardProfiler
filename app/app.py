from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_pipeline.joblib"
FEATURES = ["BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
            "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS"]


@st.cache_resource            # charge le modèle une seule fois
def load_pipeline():
    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()    # le pipeline COMPLET (scaler + modèle) : pas de prétraitement recodé ici

st.title("Prédiction du segment client")
tab_manuel, tab_csv = st.tabs(["Saisie manuelle", "Charger un CSV"])

with tab_manuel:
    valeurs = {f: st.number_input(f, min_value=0.0, value=0.0, step=10.0) for f in FEATURES}
    if st.button("Prédire"):
        client = pd.DataFrame([valeurs])                 # 1 ligne, colonnes brutes
        segment = pipeline.predict(client)[0]
        st.success(f"Segment prédit : **{segment}**")
        if hasattr(pipeline, "predict_proba"):
            probas = pd.Series(pipeline.predict_proba(client)[0], index=pipeline.classes_)
            st.bar_chart(probas)

with tab_csv:
    fichier = st.file_uploader("CSV avec les colonnes : " + ", ".join(FEATURES), type="csv")
    if fichier is not None:
        df_new = pd.read_csv(fichier)
        df_new["segment_predit"] = pipeline.predict(df_new[FEATURES])
        st.dataframe(df_new)