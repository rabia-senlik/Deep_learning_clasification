"""
ADIM 2 - KEŞİFÇİ VERİ ANALİZİ (EDA)
===================================
Model kurmadan ÖNCE veriyi gözümüzle ve istatistikle tanırız. Burada bulduğumuz
şeyler sonraki adımlardaki kararları belirler (ör. dengesizlik var mı? sınıflar
parlaklık olarak ayrışıyor mu? ne kadar augmentation gerekir?).

Üretilen grafikler: reports/figures/ klasörüne kaydedilir.

Çalıştırma:  python src/02_eda.py
"""
import cv2
import matplotlib

matplotlib.use("Agg")  # Ekran açmadan dosyaya çizim yapmak için
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as cfg

sns.set_theme(style="whitegrid")
meta = pd.read_csv(cfg.ISLENMIS_VERI_DIZINI / "metadata.csv")


def goruntu_yukle(goreli_yol):
    """Kaydedilmiş PNG'yi gri tonlamalı olarak okur."""
    return cv2.imread(str(cfg.PROJE_KOKU / goreli_yol), cv2.IMREAD_GRAYSCALE)


# --------------------------------------------------------------------------
# 2.1) SINIF DAĞILIMI - Veri dengeli mi?
# --------------------------------------------------------------------------
# Dengesiz veride model çoğunluk sınıfa yönelir ve "accuracy" yanıltıcı olur.
# Dengeliyse accuracy güvenle kullanılabilir ve class_weight gerekmez.
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.countplot(data=meta, x="sinif", hue="bolum", order=cfg.SINIF_ADLARI, ax=ax)
ax.set_title("Sınıf Dağılımı (Train / Test)")
ax.set_xlabel("Kusur Sınıfı")
ax.set_ylabel("Görüntü Sayısı")
plt.xticks(rotation=20)
plt.tight_layout()
fig.savefig(cfg.SEKIL_DIZINI / "01_sinif_dagilimi.png", dpi=130)
plt.close(fig)

oran = meta[meta.bolum == "train"]["sinif"].value_counts()
print("Train sınıf sayıları:\n", oran.to_string())
print(f"En büyük/en küçük sınıf oranı: {oran.max() / oran.min():.3f}  (1.0 = mükemmel dengeli)")

# --------------------------------------------------------------------------
# 2.2) HER SINIFTAN ÖRNEK GÖRÜNTÜLER - Gözle tanıma
# --------------------------------------------------------------------------
# İstatistik gözümüzün yerini tutmaz. Kusurlar insan gözüyle ne kadar ayırt
# edilebiliyor? Sınıflar birbirine benziyor mu? Bunu burada hissederiz.
ORNEK_SAYISI = 8
fig, axes = plt.subplots(cfg.SINIF_SAYISI, ORNEK_SAYISI, figsize=(ORNEK_SAYISI * 1.6, cfg.SINIF_SAYISI * 1.7))
rng = np.random.default_rng(cfg.RASTGELE_TOHUM)
for i, sinif in enumerate(cfg.SINIF_ADLARI):
    alt = meta[(meta.bolum == "train") & (meta.sinif == sinif)]
    secilen = alt.iloc[rng.choice(len(alt), ORNEK_SAYISI, replace=False)]
    for j, (_, satir) in enumerate(secilen.iterrows()):
        axes[i, j].imshow(goruntu_yukle(satir["dosya_yolu"]), cmap="gray", vmin=0, vmax=255)
        axes[i, j].axis("off")
    axes[i, 0].set_title(sinif, loc="left", fontsize=10, fontweight="bold")
plt.tight_layout()
fig.savefig(cfg.SEKIL_DIZINI / "02_sinif_ornekleri.png", dpi=110)
plt.close(fig)

# --------------------------------------------------------------------------
# 2.3) PARLAKLIK / KONTRAST / NETLİK - Sınıflar istatistiksel olarak ayrışıyor mu?
# --------------------------------------------------------------------------
# Eğer bir özellik (ör. ortalama parlaklık) sınıfları net ayırıyorsa, basit
# kural tabanlı bir yöntem bile işe yarayabilir. Ayrışmıyorsa CNN'in dokuyu
# öğrenmesi gerekir - Keras kullanmak için iyi bir gerekçedir.
ozellikler = {
    "ortalama_parlaklik": "Ortalama Parlaklık (0-255)",
    "std_kontrast": "Kontrast (Std. Sapma)",
    "netlik_laplace": "Netlik / Doku (Laplace Varyansı)",
}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, (kolon, baslik) in zip(axes, ozellikler.items()):
    sns.boxplot(data=meta, x="sinif", y=kolon, order=cfg.SINIF_ADLARI, ax=ax, hue="sinif", legend=False)
    ax.set_title(baslik)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
plt.tight_layout()
fig.savefig(cfg.SEKIL_DIZINI / "03_sinif_istatistikleri.png", dpi=130)
plt.close(fig)

ozet = meta.groupby("sinif")[list(ozellikler)].agg(["mean", "std"]).round(2)
ozet.to_csv(cfg.RAPOR_DIZINI / "eda_sinif_ozet.csv", encoding="utf-8")
print("\nSınıf bazında istatistik özeti:\n", ozet.to_string())

# --------------------------------------------------------------------------
# 2.4) ORTALAMA GÖRÜNTÜ - Her sınıfın "ortalama yüzü"
# --------------------------------------------------------------------------
# Bir sınıftaki tüm görüntülerin piksel ortalaması alınır. Kusur her görüntüde
# aynı yerdeyse (ör. hep ortada) ortalamada net görünür; rastgele yerlerdeyse
# bulanık bir leke çıkar. Bu bize konum (translation) augmentation kararını verir.
fig, axes = plt.subplots(1, cfg.SINIF_SAYISI, figsize=(cfg.SINIF_SAYISI * 2.4, 2.8))
for ax, sinif in zip(axes, cfg.SINIF_ADLARI):
    alt = meta[(meta.bolum == "train") & (meta.sinif == sinif)]
    yigin = np.stack([goruntu_yukle(y) for y in alt["dosya_yolu"]]).astype(np.float32)
    ax.imshow(yigin.mean(axis=0), cmap="gray", vmin=0, vmax=255)
    ax.set_title(sinif, fontsize=9)
    ax.axis("off")
plt.suptitle("Sınıf Ortalama Görüntüleri", y=1.02)
plt.tight_layout()
fig.savefig(cfg.SEKIL_DIZINI / "04_ortalama_goruntuler.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------
# 2.5) PİKSEL HİSTOGRAMI - Normalizasyon kararı için
# --------------------------------------------------------------------------
# Piksel değerleri 0-255 aralığında. Sinir ağları küçük sayılarla (0-1) daha
# kararlı öğrenir; bu yüzden 255'e böleceğiz (Adım 3'te Rescaling katmanı).
tum_pikseller = np.concatenate(
    [goruntu_yukle(y).ravel() for y in meta[meta.bolum == "train"].sample(300, random_state=cfg.RASTGELE_TOHUM)["dosya_yolu"]]
)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(tum_pikseller, bins=64, color="#4C72B0")
ax.set_title("Piksel Değeri Dağılımı (300 rastgele görüntü)")
ax.set_xlabel("Piksel değeri (0=siyah, 255=beyaz)")
ax.set_ylabel("Piksel sayısı")
plt.tight_layout()
fig.savefig(cfg.SEKIL_DIZINI / "05_piksel_histogrami.png", dpi=130)
plt.close(fig)
print(f"\nPiksel ortalaması: {tum_pikseller.mean():.1f}  |  std: {tum_pikseller.std():.1f}  |  "
      f"min: {tum_pikseller.min()}  |  max: {tum_pikseller.max()}")

# --------------------------------------------------------------------------
# 2.6) BASİT BİR "NAIVE" ALT SINIR - Sadece 3 sayıyla ne kadar ayırabiliriz?
# --------------------------------------------------------------------------
# CNN'in gerçekten değer katıp katmadığını görmek için önce çok basit bir
# yöntem deneriz: parlaklık + kontrast + netlik (3 sayı) ile Random Forest.
# CNN bunu belirgin şekilde geçemiyorsa, CNN'e gerek yok demektir.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

egitim = meta[meta.bolum == "train"]
test = meta[meta.bolum == "test"]
rf = RandomForestClassifier(n_estimators=200, random_state=cfg.RASTGELE_TOHUM)
rf.fit(egitim[list(ozellikler)], egitim["etiket"])
rf_dogruluk = accuracy_score(test["etiket"], rf.predict(test[list(ozellikler)]))
print(f"\n[Baz çizgi] 3 basit istatistik + Random Forest test doğruluğu: {rf_dogruluk:.3f}")
print(f"[Baz çizgi] Rastgele tahmin doğruluğu: {1 / cfg.SINIF_SAYISI:.3f}")
pd.Series({"rf_3_istatistik_test_dogruluk": rf_dogruluk}).to_json(cfg.RAPOR_DIZINI / "baz_cizgi.json")

print(f"\nTüm grafikler kaydedildi: {cfg.SEKIL_DIZINI}")
