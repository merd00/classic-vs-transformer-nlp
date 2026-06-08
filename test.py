import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from transformers import pipeline

# ---------- 1. KLASİK MODELİ HAZIRLA ----------
print("Klasik model eğitiliyor...")
url = "https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset/resolve/main/test.csv"
df = pd.read_csv(url)
df = df[df["label"] != "Notr"]
df["etiket"] = df["label"].map({"Positive": 1, "Negative": 0})

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["etiket"], test_size=0.2, random_state=42, stratify=df["etiket"]
)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
klasik = LinearSVC(class_weight="balanced")
klasik.fit(X_train_tfidf, y_train)
print("Klasik model hazır.")

# ---------- 2. TRANSFORMER'I YÜKLE ----------
print("Transformer yükleniyor...")
transformer = pipeline("sentiment-analysis",
                       model="savasy/bert-base-turkish-sentiment-cased")
print("Transformer hazır.\n")

# ---------- 3. ZOR CÜMLELERDE KARŞILAŞTIR ----------
zor_cumleler = [
    "bu ürün gerçekten harika çok memnun kaldım",
    "berbat bir deneyim asla tavsiye etmem",
    "çok yavaş kargo",
    "kutudan kırık çıktı",
    "ürün güzel ama kargo berbat asla almayın",
]

def klasik_tahmin(cumle):
    v = vectorizer.transform([cumle])
    return "Olumlu" if klasik.predict(v)[0] == 1 else "Olumsuz"

def transformer_tahmin(cumle):
    s = transformer(cumle)[0]
    etiket = "Olumlu" if s["label"].lower().startswith("pos") else "Olumsuz"
    return f"{etiket} (%{s['score']*100:.0f})"

print(f"{'CÜMLE':45} {'KLASİK':10} {'TRANSFORMER':18}")
print("-" * 75)
for c in zor_cumleler:
    print(f"{c[:43]:45} {klasik_tahmin(c):10} {transformer_tahmin(c):18}")
    
    
from sklearn.metrics import classification_report
import time

# Test setinden örnek al (CPU'da hız için 300 yorum)
ornek_X = X_test.iloc[:300].tolist()
ornek_y = y_test.iloc[:300].tolist()

# Klasik tahminler
klasik_pred = klasik.predict(vectorizer.transform(ornek_X))

# Transformer tahminler (CPU'da ~1 dk sürebilir)
print("\nTransformer örnek üzerinde çalışıyor (biraz sürebilir)...")
t0 = time.time()
trans_sonuc = transformer(ornek_X, truncation=True)
trans_pred = [1 if s["label"].lower().startswith("pos") else 0 for s in trans_sonuc]
print(f"Transformer süresi: {time.time()-t0:.1f} saniye")

print("\n===== KLASİK MODEL =====")
print(classification_report(ornek_y, klasik_pred, target_names=["Negative", "Positive"]))
print("===== TRANSFORMER =====")
print(classification_report(ornek_y, trans_pred, target_names=["Negative", "Positive"]))


print("\n--- Transformer'ın yanıldığı (gerçek: olumlu, tahmin: olumsuz) ---")
sayac = 0
for metin, gercek, tahmin in zip(ornek_X, ornek_y, trans_pred):
    if gercek == 1 and tahmin == 0:
        print(f"• {metin[:80]}")
        sayac += 1
        if sayac == 8:
            break
        
        
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score

os.makedirs("images", exist_ok=True)

# Metrikleri otomatik hesapla (tahminlerden)
etiketler = ["Accuracy", "Negative F1", "Macro F1"]
klasik_degerler = [
    accuracy_score(ornek_y, klasik_pred),
    f1_score(ornek_y, klasik_pred, pos_label=0),
    f1_score(ornek_y, klasik_pred, average="macro"),
]
trans_degerler = [
    accuracy_score(ornek_y, trans_pred),
    f1_score(ornek_y, trans_pred, pos_label=0),
    f1_score(ornek_y, trans_pred, average="macro"),
]

x = np.arange(len(etiketler))
w = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, klasik_degerler, w, label="Klasik (TF-IDF + SVM)", color="#0f6e56")
ax.bar(x + w/2, trans_degerler, w, label="Transformer (hazır)", color="#185fa5")

ax.set_ylabel("Skor")
ax.set_title("Klasik Model vs Transformer — Test Seti (300 örnek)")
ax.set_xticks(x)
ax.set_xticklabels(etiketler)
ax.set_ylim(0, 1.05)
ax.legend()

# Çubukların üstüne değerleri yaz
for i, v in enumerate(klasik_degerler):
    ax.text(i - w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
for i, v in enumerate(trans_degerler):
    ax.text(i + w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("LLM/images/karsilastirma.png", dpi=120, bbox_inches="tight")
plt.close()
print("Grafik kaydedildi: images/karsilastirma.png")