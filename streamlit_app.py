"""
streamlit_app.py
-----------------
Interface Streamlit pour l'analyse de sentiment.
Appelle l'API FastAPI (locale en dev, Render en production).

Lancement local :
    streamlit run streamlit_app.py
"""

import os
import requests
import streamlit as st
import pandas as pd

# Configuration de la page (doit être la première commande Streamlit)
st.set_page_config(page_title="Analyse de Sentiment", page_icon="💬", layout="centered")

# En local : http://localhost:8000
# En production : URL Render, à définir dans les secrets Streamlit Cloud
# (Settings -> Secrets -> API_URL = "https://votre-api.onrender.com")
API_URL = os.environ.get("API_URL", "http://localhost:8000")
try:
    API_URL = st.secrets.get("API_URL", API_URL)
except (FileNotFoundError, KeyError):
    pass

st.title("💬 Analyse de Sentiment")
st.caption(f"API connectée : `{API_URL}`")


@st.cache_data(ttl=30, show_spinner=False)
def check_health(api_url: str):
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "unreachable", "model_loaded": False, "error": str(e)}


health = check_health(API_URL)
if health.get("status") == "ok":
    st.success("✅ API opérationnelle")
elif health.get("status") == "degraded":
    st.warning("⚠️ API en ligne mais modèle non chargé")
else:
    st.error(f"❌ API injoignable : {health.get('error', 'inconnu')}")

st.divider()

text_input = st.text_area(
    "Entrez un texte à analyser",
    placeholder="Ex : This product exceeded all my expectations!",
    height=120,
)

col1, col2 = st.columns(2)
with col1:
    predict_clicked = st.button("🔍 Prédire le sentiment", use_container_width=True)
with col2:
    explain_clicked = st.button("🧠 Expliquer la prédiction", use_container_width=True)

if predict_clicked or explain_clicked:
    if not text_input.strip():
        st.warning("Merci d'entrer un texte.")
    else:
        endpoint = "/explain" if explain_clicked else "/predict"
        payload = {"text": text_input}
        if explain_clicked:
            payload["top_n"] = 5

        try:
            with st.spinner("Analyse en cours..."):
                response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            sentiment = result["sentiment"]
            confidence = result["confidence"]

            if sentiment == "positive":
                st.success(f"Sentiment : **Positif** 😊 (confiance : {confidence:.1%})")
            else:
                st.error(f"Sentiment : **Négatif** 😞 (confiance : {confidence:.1%})")

            if explain_clicked and "top_words" in result:
                st.subheader("Mots les plus influents")
                df = pd.DataFrame(result["top_words"])
                if not df.empty:
                    df = df.set_index("word")
                    st.bar_chart(df["weight"])
                    st.dataframe(df, use_container_width=True)

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")

st.divider()
st.caption("Projet d'analyse de sentiment — API FastAPI (Render) + Interface Streamlit (Streamlit Cloud)")
