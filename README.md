# Klasik NLP vs Transformer — Türkçe Duygu Analizi Karşılaştırması

Türkçe yorumlarda iki yaklaşımın karşılaştırılması: klasik makine öğrenmesi
(TF-IDF + Linear SVM) ile önceden eğitilmiş bir transformer modeli. Sayısal
karşılaştırma ve derinlemesine hata analizi içerir.

## Kullanılan Araçlar
- Python, pandas, scikit-learn, matplotlib
- HuggingFace Transformers (savasy/bert-base-turkish-sentiment-cased)

## Yöntem
1. Klasik model bu veriyle eğitildi (TF-IDF + Linear SVM, class_weight="balanced")
2. Transformer hazır (out-of-the-box) olarak kullanıldı
3. Hem seçilmiş zor cümlelerde hem de temsili test örneğinde (300) karşılaştırıldı

## Sonuçlar (300 örneklik test)
| Metrik | Klasik | Transformer |
|---|---|---|
| Accuracy | %92 | %68 |
| Negative F1 | 0.80 | 0.54 |
| Macro F1 | 0.87 | 0.65 |

![Karşılaştırma](images/karsilastirma.png)

## Beklenmedik Bulgu ve Hata Analizi
Seçilmiş zor cümlelerde ("çok yavaş kargo" gibi) transformer üstün göründü. Ancak
temsili test setinde kendi verisiyle eğitilmiş klasik model belirgin şekilde öne geçti.

Hatalar incelendiğinde, transformer'ın yanıldığı vakaların büyük kısmının aslında:
- Karışık yorumlar (hem övgü hem şikayet),
- Nötr/bilgilendirici cümleler,
- Tartışmalı şekilde etiketlenmiş örnekler
olduğu görüldü. Yani fark kısmen domain uyumsuzluğundan, kısmen de veri
etiketlerinin gürültüsünden kaynaklanıyor.

## Çıkarımlar
- Önceden eğitilmiş model otomatik olarak daha iyi değildir; veriye uyum belirleyicidir.
- Gürültülü etikete karşı ölçülen accuracy, gürültüyü taklit etmeyi ödüllendirir.
- Adil karşılaştırma için temiz etiketler veya transformer'ın bu veriyle
  fine-tune edilmesi gerekir.
- Seçilmiş örnekler yanıltıcıdır; temsili örneklemde ölçmek şarttır.

## Çalıştırmak için
```bash
pip install -r requirements.txt
python karsilastirma.py
```

---

# Classic NLP vs Transformer — Turkish Sentiment Analysis Comparison

Comparison of two approaches on Turkish reviews: classic machine learning
(TF-IDF + Linear SVM) vs a pre-trained transformer. Includes quantitative
comparison and in-depth error analysis.

## Tools
- Python, pandas, scikit-learn, matplotlib
- HuggingFace Transformers (savasy/bert-base-turkish-sentiment-cased)

## Method
1. Classic model trained on this data (TF-IDF + Linear SVM, class_weight="balanced")
2. Transformer used out-of-the-box (zero-shot on this distribution)
3. Compared on both curated hard sentences and a representative test sample (300)

## Results (300-sample test)
| Metric | Classic | Transformer |
|---|---|---|
| Accuracy | 92% | 68% |
| Negative F1 | 0.80 | 0.54 |
| Macro F1 | 0.87 | 0.65 |

![Comparison](images/karsilastirma.png)

## Surprising Finding & Error Analysis
On curated hard sentences (e.g., "çok yavaş kargo"/"slow shipping"), the transformer
appeared superior. However, on a representative test set, the classic model trained
on this data clearly won.

Inspecting the errors revealed that most of the transformer's misclassifications were
actually mixed reviews (both praise and criticism), neutral/informational sentences,
or arguably mislabeled examples. The gap stems partly from domain mismatch and partly
from noisy data labels.

## Takeaways
- A pre-trained model is not automatically better; alignment with the data matters.
- Accuracy measured against noisy labels rewards mimicking the noise.
- A fair comparison requires clean labels or fine-tuning the transformer on this data.
- Cherry-picked examples are misleading; evaluate on a representative sample.

## How to run
```bash
pip install -r requirements.txt
python karsilastirma.py
```