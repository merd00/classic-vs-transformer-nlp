# ============================================
# Klasik NLP vs Transformer — Türkçe Duygu Analizi
# ============================================
import time, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import pipeline

# ---------- 1. Veri ----------
url = "https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset/resolve/main/test.csv"
df = pd.read_csv(url)
df = df[df["label"] != "Notr"]
df["etiket"] = df["label"].map({"Positive": 1, "Negative": 0})

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["etiket"], test_size=0.2, random_state=42, stratify=df["etiket"]
)

# ---------- 2. Klasik model (TF-IDF + Linear SVM) ----------
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
klasik = LinearSVC(class_weight="balanced")
klasik.fit(X_train_tfidf, y_train)

# ---------- 3. Transformer (hazır, önceden eğitilmiş) ----------
transformer = pipeline("sentiment-analysis",
                       model="savasy/bert-base-turkish-sentiment-cased")

def trans_etiket(s):
    return 1 if s["label"].lower().startswith("pos") else 0

# ---------- 4. Sayısal karşılaştırma (300 örnek) ----------
ornek_X = X_test.iloc[:300].tolist()
ornek_y = y_test.iloc[:300].tolist()

klasik_pred = klasik.predict(vectorizer.transform(ornek_X))
trans_pred = [trans_etiket(s) for s in transformer(ornek_X, truncation=True)]

print("===== KLASİK =====")
print(classification_report(ornek_y, klasik_pred, target_names=["Negative", "Positive"]))
print("===== TRANSFORMER =====")
print(classification_report(ornek_y, trans_pred, target_names=["Negative", "Positive"]))

# ---------- 5. Görselleştirme ----------
os.makedirs("images", exist_ok=True)
etiketler = ["Accuracy", "Negative F1", "Macro F1"]
klasik_d = [accuracy_score(ornek_y, klasik_pred),
            f1_score(ornek_y, klasik_pred, pos_label=0),
            f1_score(ornek_y, klasik_pred, average="macro")]
trans_d = [accuracy_score(ornek_y, trans_pred),
           f1_score(ornek_y, trans_pred, pos_label=0),
           f1_score(ornek_y, trans_pred, average="macro")]

x = np.arange(len(etiketler)); w = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, klasik_d, w, label="Klasik (TF-IDF + SVM)", color="#0f6e56")
ax.bar(x + w/2, trans_d, w, label="Transformer (hazır)", color="#185fa5")
ax.set_ylabel("Skor"); ax.set_ylim(0, 1.05)
ax.set_title("Klasik Model vs Transformer — Test Seti (300 örnek)")
ax.set_xticks(x); ax.set_xticklabels(etiketler); ax.legend()
for i, v in enumerate(klasik_d): ax.text(i - w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
for i, v in enumerate(trans_d): ax.text(i + w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("images/karsilastirma.png", dpi=120, bbox_inches="tight")
plt.close()
print("Grafik kaydedildi.")