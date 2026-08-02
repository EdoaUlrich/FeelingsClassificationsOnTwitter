"""
train_model.py
--------------
Version script Python (.py) du notebook TrainingModel_corrected.ipynb.

Reprend fidèlement le pipeline du notebook :
  1. Connexion MLflow (tracking server)
  2. Chargement + exploration du dataset Sentiment140
  3. Visualisations de la distribution des sentiments
  4. Échantillonnage stratifié (2%)
  5. Prétraitement NLP (emojis, spaCy FR/EN, langdetect, stopwords)
  6. Split train/test stratifié
  7. Vectorisation TF-IDF
  8. Comparaison de plusieurs modèles (evaluate_and_log_model)
  9. Optimisation par GridSearchCV (runs MLflow imbriqués)
 10. Enregistrement dans le Model Registry MLflow (alias Staging/Production)
 11. Interprétabilité (coefficients + LIME)
 12. Synthèse de toutes les expériences MLflow
 13. Export des artifacts pour l'API (joblib) — ÉTAPE CRITIQUE pour le déploiement
 14. Benchmark MLflow vs Joblib

Prérequis (voir requirements.txt) :
    pip install mlflow python-dotenv seaborn emoji langdetect lime spacy
                nltk scikit-learn pandas numpy joblib wget xgboost
    python -m spacy download fr_core_news_sm
    python -m spacy download en_core_web_sm

Variables d'environnement (fichier .env, voir .env.example) :
    MLFLOW_TRACKING_URI=https://votre-serveur-mlflow
    EXPERIMENT_NAME=EHU_sentiment_analysis

Usage :
    python train_model.py
"""

import os
import re
import json
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # backend non-interactif : indispensable en script (pas de notebook display)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

import spacy
import emoji
from langdetect import detect, LangDetectException
from nltk.stem.snowball import SnowballStemmer
from dotenv import load_dotenv

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve, get_scorer,
)
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

# ============================================================
# CONFIGURATION GLOBALE
# ============================================================
DATASET_URL = "https://github.com/archiducarmel/ESIEA_MLOPS/releases/download/datas/training.1600000.processed.noemoticon.csv"
SAMPLING_RATIO = 0.02
RANDOM_STATE = 42
API_ARTIFACTS_DIR = "api_artifacts"
MODEL_NAME = "meilleur_modele"


# ============================================================
# 5. PRÉTRAITEMENT NLP (fonctions réutilisables, importées par l'API)
# ============================================================
def load_preprocessing_resources(emojis_path="dict_emojis.json", stopwords_path="stopwords.txt"):
    """Charge le dictionnaire d'emojis et la liste de stopwords depuis disque."""
    with open(emojis_path, "r", encoding="utf-8") as fichier:
        dict_emojis = json.load(fichier)

    stopwords = set()
    try:
        with open(stopwords_path, "r", encoding="utf-8") as fichier:
            stopwords = set(ligne.strip() for ligne in fichier if ligne.strip())
        print(f"{len(stopwords)} stopwords chargés avec succès !")
    except FileNotFoundError:
        print("Erreur : le fichier 'stopwords.txt' est introuvable.")

    return dict_emojis, stopwords


def load_nlp_models():
    """Charge les pipelines spaCy FR/EN et les stemmers Snowball."""
    nlp_models = {}
    try:
        nlp_models = {
            "fr": spacy.load("fr_core_news_sm"),
            "en": spacy.load("en_core_web_sm"),
        }
    except OSError:
        print("Erreur : modèle SpaCy manquant. Lancez : python -m spacy download fr_core_news_sm")

    stemmers = {
        "fr": SnowballStemmer("french"),
        "en": SnowballStemmer("english"),
    }
    return nlp_models, stemmers


def preprocess_text(tweet, dict_emojis, stopwords, nlp_models, stemmers, stem_or_lem="lem"):
    """
    Prétraite un tweet : détection de langue, nettoyage (URLs/mentions/hashtags),
    normalisation des emojis, tokenisation + lemmatisation/stemming via spaCy.

    Returns:
        list[str] : liste de tokens prétraités.
    """
    if isinstance(tweet, list):
        tweet = " ".join(str(element) for element in tweet)

    tweet = str(tweet)
    try:
        langue_detectee = detect(tweet)
    except LangDetectException:
        langue_detectee = "en"  # valeur par défaut si le texte est vide ou indétectable

    if langue_detectee not in ["fr", "en"]:
        langue_detectee = "en"

    texte = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    texte = re.sub(r"@\w+", "", texte)
    texte = re.sub(r"#\w+", "", texte)
    texte = emoji.demojize(texte, language=langue_detectee)
    for emoticon, valeur_texte in dict_emojis[langue_detectee].items():
        texte = texte.replace(emoticon, f" {valeur_texte} ")
    texte = re.sub(r"\s+", " ", texte).strip()

    doc = nlp_models[langue_detectee](texte)
    mots = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        mot_original = token.text.lower()
        if stem_or_lem == "lem":
            mot_transforme = token.lemma_.lower()
        elif stem_or_lem == "stem":
            mot_transforme = stemmers[langue_detectee].stem(mot_original)
        else:
            mot_transforme = mot_original

        if mot_original not in stopwords and mot_transforme not in stopwords:
            mots.append(mot_transforme)
    return mots


# ============================================================
# 8. COMPARAISON DE MODÈLES
# ============================================================
def evaluate_and_log_model(model, model_name, X_train, X_test, y_train, y_test, **model_params):
    """Entraîne, évalue un modèle et logge tout dans MLflow (params, métriques, figures, modèle)."""
    with mlflow.start_run(run_name=model_name) as run:
        if model_params:
            model.set_params(**model_params)

        try:
            for k, v in model.get_params().items():
                mlflow.log_param(k, v)
        except Exception:
            pass

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        predictions = model.predict(X_test)
        # LinearRegression renvoie des valeurs continues : on binarise pour les métriques
        # de classification (accuracy, f1, matrice de confusion)
        pred_labels = np.round(np.clip(predictions, 0, 1)).astype(int)

        # AUC calculée sur des scores/probabilités continus, pas sur les labels 0/1
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            y_score = predictions

        accuracy = accuracy_score(y_test, pred_labels)
        f1 = f1_score(y_test, pred_labels, average="weighted")
        auc = roc_auc_score(y_test, y_score)

        metrics = {
            "Temps_entrainement_secondes": training_time,
            "Accuracy": accuracy,
            "f1_score": f1,
            "AUC_ROC_Score": auc,
        }
        mlflow.log_metrics(metrics)

        cm = confusion_matrix(y_test, pred_labels)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Négatif", "Positif"], yticklabels=["Négatif", "Positif"], ax=ax_cm)
        ax_cm.set_xlabel("Prédiction")
        ax_cm.set_ylabel("Réalité")
        ax_cm.set_title(f"Matrice de confusion - {model_name}")
        mlflow.log_figure(fig_cm, f"confusion_matrix_{model_name}.png")
        plt.close(fig_cm)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey")
        ax_roc.set_xlabel("Taux de faux positifs")
        ax_roc.set_ylabel("Taux de vrais positifs")
        ax_roc.set_title(f"Courbe ROC - {model_name}")
        ax_roc.legend()
        mlflow.log_figure(fig_roc, f"roc_curve_{model_name}.png")
        plt.close(fig_roc)

        model_info = mlflow.sklearn.log_model(model, name=model_name)
        run_id = run.info.run_id

    return {**metrics, "model_uri": model_info.model_uri, "run_id": run_id}


# ============================================================
# 9. GRIDSEARCH + TRACKING MLFLOW (runs imbriqués)
# ============================================================
def log_gridsearch_to_mlflow(estimator, param_grid, X_train, y_train, X_test, y_test, cv=5, scoring="accuracy"):
    """Exécute GridSearchCV en loggant chaque configuration testée (run parent + runs enfants)."""
    with mlflow.start_run(run_name="Optimisation") as parent_run:
        grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring=scoring)
        grid.fit(X_train, y_train)

        cv_results = grid.cv_results_
        nombre_configs = len(cv_results["params"])

        for i in range(nombre_configs):
            with mlflow.start_run(run_name=f"Optimisation_fils{i+1}", nested=True):
                mlflow.log_params(cv_results["params"][i])
                mlflow.log_metric(f"cv_mean_{scoring}", cv_results["mean_test_score"][i])
                mlflow.log_metric(f"cv_std_{scoring}", cv_results["std_test_score"][i])
                mlflow.log_metric("fit_time_seconds", cv_results["mean_fit_time"][i])

        best_params = grid.best_params_
        best_model = grid.best_estimator_

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        model_info = mlflow.sklearn.log_model(best_model, name="meilleur_modele")

        scorer = get_scorer(scoring)
        test_score = scorer(best_model, X_test, y_test)
        mlflow.log_metric(f"test_{scoring}", test_score)

        print("\n✅ Optimisation terminée !")
        print(f"Meilleurs paramètres : {best_params}")
        print(f"Score de validation croisée ({scoring}) : {grid.best_score_:.4f}")
        print(f"Score sur le jeu de test ({scoring}) : {test_score:.4f}")

        return best_model, parent_run.info.run_id, model_info.model_uri


# ============================================================
# FONCTIONS DE CHARGEMENT POUR LE BENCHMARK MLFLOW VS JOBLIB
# ============================================================
def load_model_from_mlflow(model_name=MODEL_NAME):
    # NB : on a assigné l'alias "Production" (pas le stage déprécié), donc la syntaxe
    # d'URI utilise "@Production" et non "/Production"
    return mlflow.pyfunc.load_model(f"models:/{model_name}@Production")


def load_model_from_joblib(artifacts_dir=API_ARTIFACTS_DIR):
    m = joblib.load(os.path.join(artifacts_dir, "sentiment_model.joblib"))
    v = joblib.load(os.path.join(artifacts_dir, "tfidf_vectorizer.joblib"))
    return m, v


def benchmark_approaches(tfidf, model_name=MODEL_NAME):
    test_texts = ["I love this product!", "This is terrible.", "Average, nothing special."]

    mlflow_model = load_model_from_mlflow(model_name)
    start = time.time()
    for t in test_texts:
        mlflow_model.predict(tfidf.transform([t]))
    mlflow_time = (time.time() - start) / len(test_texts)

    model, vectorizer = load_model_from_joblib()
    start = time.time()
    for t in test_texts:
        model.predict(vectorizer.transform([t]))
    joblib_time = (time.time() - start) / len(test_texts)

    print(f"MLflow : {mlflow_time*1000:.1f} ms/prédiction")
    print(f"Joblib : {joblib_time*1000:.1f} ms/prédiction")
    print(f"🚀 Joblib est {mlflow_time/joblib_time:.1f}x plus rapide")


# ============================================================
# PIPELINE PRINCIPAL (exécution séquentielle, reprend l'ordre du notebook)
# ============================================================
def main():
    # ---- 1. Connexion MLflow ----
    load_dotenv(override=True)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = os.getenv("EXPERIMENT_NAME")
    print(f"MLFLOW_TRACKING_URI: {tracking_uri}")
    print(f"EXPERIMENT_NAME: {experiment_name}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_metric("Connexion", 1)
        print("✓ Connexion au serveur MLflow réussie !")

    # ---- 2. Chargement + exploration du dataset ----
    dataset = pd.read_csv(
        DATASET_URL, names=["target", "ids", "date", "flag", "user", "text"], encoding="latin-1"
    )

    with mlflow.start_run(run_name="data_exploration"):
        dataset = dataset.loc[:, ["target", "text"]]
        dataset["target"] = dataset["target"].replace(4, 1)
        print(dataset.info())

        class_counts = dataset["target"].value_counts()
        mlflow.log_metric("total_rows", len(dataset))
        mlflow.log_metric("negative_count", int(class_counts.get(0, 0)))
        mlflow.log_metric("positive_count", int(class_counts.get(1, 0)))
        mlflow.log_metric("missing_values", int(dataset.isna().sum().sum()))

    # ---- 3. Visualisations ----
    repartition = dataset["target"].value_counts()
    repartition_pct = dataset["target"].value_counts(normalize=True) * 100

    with mlflow.start_run(run_name="data_visualisation"):
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.pie(repartition, labels=repartition.index, autopct="%1.1f%%", startangle=140,
                colors=["#4e79a7", "#f28e2b"])
        ax1.set_title("Repartition des sentiments", fontsize=14, weight="bold")
        mlflow.log_figure(fig1, "pie_chart_repartition.jpg")
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        bars = ax2.bar(repartition_pct.index, repartition_pct.values, color=["#4e79a7", "#f28e2b"])
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                         fontsize=10, weight="bold")
        ax2.set_title("Répartition en pourcentage des émotions", fontsize=14, weight="bold")
        ax2.set_xlabel("Emotions")
        ax2.set_ylabel("Repartition")
        mlflow.log_figure(fig2, "bar_chart_pourcentage.jpg")
        plt.close(fig2)

    with mlflow.start_run(run_name="data_statistics"):
        mlflow.log_param("sentiments_0", int(repartition[0]))
        mlflow.log_param("sentiments_1", int(repartition[1]))
        mlflow.log_param("sentiments_0_pct", f"{repartition_pct[0]:.2f}%")
        mlflow.log_param("sentiments_1_pct", f"{repartition_pct[1]:.2f}%")
        mlflow.log_param("total_samples", len(dataset))
        print("✓ Statistics logged successfully!")

    # ---- 4. Échantillonnage stratifié ----
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["text"], dataset["target"], train_size=SAMPLING_RATIO,
        stratify=dataset["target"], random_state=RANDOM_STATE,
    )

    with mlflow.start_run(run_name="data_sampling"):
        mlflow.log_param("RANDOM_STATE", RANDOM_STATE)
        mlflow.log_param("SAMPLING_RATIO", SAMPLING_RATIO)

    # ---- 5. Prétraitement NLP ----
    dict_emojis, stopwords = load_preprocessing_resources()
    nlp_models, stemmers = load_nlp_models()

    with mlflow.start_run(run_name="text_preprocessing"):
        example_text = "This is a sample tweet with @user and http://example.com :) !!!"
        processed = preprocess_text(example_text, dict_emojis, stopwords, nlp_models, stemmers)
        print(f"Original: {example_text}")
        print(f"Processed: {processed}")

    # ---- 6. Split train/test stratifié ----
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        X_train, y_train, test_size=0.15, random_state=90, stratify=y_train
    )
    with mlflow.start_run(run_name="data_splitting"):
        repartition_pct1 = y_train1.value_counts(normalize=True) * 100
        mlflow.log_param("TEST SIZE", 0.15)
        mlflow.log_param("RANDOM STATE", 90)
        mlflow.log_metric("train_size", len(X_train1))
        mlflow.log_metric("test_size_n", len(X_test1))
        mlflow.log_metric("train_negative_pct", repartition_pct1.get(0, 0))
        mlflow.log_metric("train_positive_pct", repartition_pct1.get(1, 0))

    X_train1 = X_train1.to_frame(name="text")
    X_test1 = X_test1.to_frame(name="text")

    texte_traitee_train = [
        preprocess_text(ligne, dict_emojis, stopwords, nlp_models, stemmers) for ligne in X_train1["text"]
    ]
    texte_traitee_test = [
        preprocess_text(ligne, dict_emojis, stopwords, nlp_models, stemmers) for ligne in X_test1["text"]
    ]

    textes_finaux_train = [" ".join(lst) for lst in texte_traitee_train]
    textes_finaux_test = [" ".join(lst) for lst in texte_traitee_test]
    X_train1["final_processed_text"] = textes_finaux_train
    X_test1["final_processed_text"] = textes_finaux_test

    print(f"✅ {len(X_train1)} textes train / {len(X_test1)} textes test prétraités")

    # ---- 7. Vectorisation TF-IDF ----
    with mlflow.start_run(run_name="tfidf_vectorization"):
        min_df, max_df, stopword = 5, 0.8, "english"
        tfidf = TfidfVectorizer(
            min_df=min_df, max_df=max_df, stop_words=stopword, ngram_range=(1, 2), max_features=5000,
        )
        tfidf.fit(X_train1["final_processed_text"])
        X_train_tfidf = tfidf.transform(X_train1["final_processed_text"])
        X_test_tfidf = tfidf.transform(X_test1["final_processed_text"])

        mlflow.log_params({
            "min_df": min_df, "max_df": max_df, "stop_words": stopword,
            "ngram_range": "(1, 2)", "max_features": 5000,
        })
        mlflow.log_metric("Taille_vocabulaire", len(tfidf.vocabulary_))

    print(X_train_tfidf.shape, X_test_tfidf.shape)

    # ---- 8. Comparaison de modèles ----
    models_to_compare = {
        "Regression_Logistic": LogisticRegression(max_iter=1000),
        "Regression_Lineaire": LinearRegression(),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(),
    }
    try:
        from xgboost import XGBClassifier
        models_to_compare["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=42)
    except ImportError:
        print("⚠️ xgboost n'est pas installé (pip install xgboost) - modèle ignoré.")

    results = {
        name: evaluate_and_log_model(model, name, X_train_tfidf, X_test_tfidf, y_train1, y_test1)
        for name, model in models_to_compare.items()
    }
    results_df = pd.DataFrame(results).T.sort_values("Accuracy", ascending=False)
    print("\n🏆 Comparaison des modèles :")
    print(results_df[["Accuracy", "AUC_ROC_Score", "f1_score", "Temps_entrainement_secondes"]])

    # ---- 9. GridSearchCV ----
    param_grid = {"C": [0.1, 1.0, 10.0, 15.0, 20.0], "max_iter": [1000, 2000, 3000, 4000, 5000]}
    model_best, id_best_model, best_model_uri = log_gridsearch_to_mlflow(
        models_to_compare["Regression_Logistic"], param_grid,
        X_train_tfidf, y_train1, X_test_tfidf, y_test1, cv=5, scoring="accuracy",
    )

    # ---- 10. Model Registry ----
    client = MlflowClient()
    model_version_info = mlflow.register_model(model_uri=best_model_uri, name=MODEL_NAME)
    version_str = str(model_version_info.version)
    print(f"Modèle enregistré sous le nom '{MODEL_NAME}', version : {version_str}")

    client.update_registered_model(
        name=MODEL_NAME, description="Modèle de classification optimisé pour prédire la catégorie cible."
    )
    client.set_registered_model_tag(name=MODEL_NAME, key="tache", value="classification_binaire")
    client.set_model_version_tag(
        name=MODEL_NAME, version=version_str, key="statut_validation", value="approuve_par_data_scientist"
    )
    client.set_registered_model_alias(name=MODEL_NAME, alias="Staging", version=version_str)
    client.set_registered_model_alias(name=MODEL_NAME, alias="Production", version=version_str)
    print(f"✅ Alias 'Staging' et 'Production' assignés à la version {version_str}.")

    # ---- 11. Interprétabilité (coefficients + LIME) ----
    with mlflow.start_run(run_name="model_interpretability"):
        if hasattr(model_best, "coef_"):
            feature_names = tfidf.get_feature_names_out()
            coefs = model_best.coef_[0]
            top_positive_indices = np.argsort(coefs)[-10:]
            top_negative_indices = np.argsort(coefs)[:10]
            top_indices = np.concatenate([top_negative_indices, top_positive_indices])

            fig_coef, ax = plt.subplots(figsize=(10, 6))
            colors = ["red" if c < 0 else "green" for c in coefs[top_indices]]
            ax.barh(np.arange(20), coefs[top_indices], color=colors)
            ax.set_yticks(np.arange(20))
            ax.set_yticklabels(feature_names[top_indices])
            plt.title("Importance Globale : Top Mots Positifs et Négatifs")
            plt.xlabel("Valeur du coefficient")
            plt.tight_layout()
            mlflow.log_figure(fig_coef, "interpretability/global_coefficients.png")
            plt.close(fig_coef)
            print("✅ Graphique des coefficients loggé.")

        pipeline = make_pipeline(tfidf, model_best)
        explainer = LimeTextExplainer(class_names=["Classe_Negative", "Classe_Positive"])
        texte_exemple = "C'est un produit absolument fantastique mais la livraison a été un peu lente."

        if hasattr(model_best, "predict_proba"):
            explanation = explainer.explain_instance(texte_exemple, pipeline.predict_proba, num_features=6)
            fig_lime = explanation.as_pyplot_figure()
            plt.title("Explication LIME (Prédiction locale)")
            plt.tight_layout()
            mlflow.log_figure(fig_lime, "interpretability/lime_local_explanation.png")
            plt.close(fig_lime)

            html_path = "lime_report.html"
            explanation.save_to_file(html_path)
            mlflow.log_artifact(html_path, "interpretability")
            print(f"✅ Explications LIME loggées pour l'exemple : '{texte_exemple}'")

    # ---- 12. Synthèse des expériences MLflow ----
    experiment = client.get_experiment_by_name(experiment_name)
    all_runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.Accuracy DESC"])

    records = [{
        "run_name": run.data.tags.get("mlflow.runName", ""),
        "run_id": run.info.run_id,
        "accuracy": run.data.metrics.get("Accuracy"),
        "auc": run.data.metrics.get("AUC_ROC_Score"),
        "f1_score": run.data.metrics.get("f1_score"),
        "train_time_s": run.data.metrics.get("Temps_entrainement_secondes"),
    } for run in all_runs]

    runs_df = pd.DataFrame(records)
    runs_with_accuracy = runs_df.dropna(subset=["accuracy"]).sort_values("accuracy", ascending=False)
    print(f"📊 {len(runs_df)} runs trouvés dans l'expérience '{experiment_name}'")
    print(runs_with_accuracy.head(10))

    report_path = "mlflow_experiments_report.csv"
    runs_with_accuracy.to_csv(report_path, index=False)
    with mlflow.start_run(run_name="experiments_summary"):
        mlflow.log_artifact(report_path)
        mlflow.log_metric("total_runs_analyzed", len(runs_df))
        if not runs_with_accuracy.empty:
            mlflow.log_metric("best_accuracy_found", runs_with_accuracy.iloc[0]["accuracy"])

    # ---- 13. Export des artifacts pour l'API (ÉTAPE CRITIQUE) ----
    os.makedirs(API_ARTIFACTS_DIR, exist_ok=True)
    with mlflow.start_run(run_name="deployment_artifacts_export"):
        model_path = os.path.join(API_ARTIFACTS_DIR, "sentiment_model.joblib")
        vectorizer_path = os.path.join(API_ARTIFACTS_DIR, "tfidf_vectorizer.joblib")
        preprocessing_path = os.path.join(API_ARTIFACTS_DIR, "preprocessing_objects.joblib")

        joblib.dump(model_best, model_path)
        joblib.dump(tfidf, vectorizer_path)
        # Les modèles spaCy ne sont PAS sérialisés (trop volumineux/fragiles en joblib) :
        # l'API doit faire spacy.load(...) elle-même. On exporte seulement les ressources légères.
        joblib.dump({"dict_emojis": dict_emojis, "stopwords": list(stopwords)}, preprocessing_path)

        metadata = {
            "model_name": MODEL_NAME,
            "algorithm": "LogisticRegression (optimisé GridSearchCV)",
            "training_date": datetime.now().isoformat(),
            "dataset": "Sentiment140",
            "sampling_ratio": SAMPLING_RATIO,
            "vocabulary_size": len(tfidf.vocabulary_),
            "labels": {"0": "negative", "1": "positive"},
        }
        metadata_path = os.path.join(API_ARTIFACTS_DIR, "model_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        mlflow.log_param("deployment_strategy", "hybrid_mlflow_joblib")
        mlflow.log_artifacts(API_ARTIFACTS_DIR, artifact_path="api_artifacts")

        reloaded_model = joblib.load(model_path)
        reloaded_vectorizer = joblib.load(vectorizer_path)
        test_vect = reloaded_vectorizer.transform(["i really love this new feature"])
        test_pred = reloaded_model.predict(test_vect)[0]
        print(f"🧪 Test de rechargement -> sentiment prédit : {'positive' if test_pred == 1 else 'negative'}")

    print(f"\n✅ Dossier '{API_ARTIFACTS_DIR}' créé avec : sentiment_model.joblib, "
          f"tfidf_vectorizer.joblib, preprocessing_objects.joblib, model_metadata.json")

    # ---- 14. Benchmark MLflow vs Joblib ----
    benchmark_approaches(tfidf, MODEL_NAME)


if __name__ == "__main__":
    main()
