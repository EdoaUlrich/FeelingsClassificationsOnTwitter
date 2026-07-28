"""
train_model.py
--------------
Entraîne un modèle de classification de sentiment (positif / négatif)
à partir d'un jeu de données texte, puis sauvegarde :
  - artifacts/tfidf_vectorizer.joblib
  - artifacts/sentiment_model.joblib

Usage:
    python train_model.py
"""

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = Path("data/reviews.csv")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    return df


def main():
    print("📥 Chargement des données...")
    df = load_data(DATA_PATH)
    print(f"   -> {len(df)} exemples chargés")
    print(df["label"].value_counts())

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n🔤 Vectorisation TF-IDF...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("🧠 Entraînement du modèle (Régression Logistique)...")
    model = LogisticRegression(max_iter=1000, C=10)
    model.fit(X_train_vec, y_train)

    print("\n📊 Évaluation sur le jeu de test :")
    y_pred = model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    print("💾 Sauvegarde des artifacts...")
    joblib.dump(model, ARTIFACTS_DIR / "sentiment_model.joblib")
    joblib.dump(vectorizer, ARTIFACTS_DIR / "tfidf_vectorizer.joblib")
    print(f"   -> {ARTIFACTS_DIR / 'sentiment_model.joblib'}")
    print(f"   -> {ARTIFACTS_DIR / 'tfidf_vectorizer.joblib'}")
    print("\n✅ Entraînement terminé.")


if __name__ == "__main__":
    main()
