"""
ADIM 4 - DEĞERLENDİRME VE AÇIKLANABİLİRLİK (GRAD-CAM)
=====================================================
Eğitilmiş modeli, daha önce HİÇ görmediği TEST setinde bir kez değerlendirir:
  4.1  Doğruluk + sınıf bazlı precision / recall / f1
  4.2  Karışıklık matrisi (hangi sınıflar birbirine karışıyor?)
  4.3  En hatalı tahminler (model nerede yanılıyor, neden?)
  4.4  Grad-CAM: model karar verirken görüntünün NERESİNE bakıyor?
  4.5  Baz çizgi (3 istatistik + Random Forest) ile karşılaştırma

Çalıştırma:  python src/04_degerlendir.py
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import config as cfg

# --------------------------------------------------------------------------
# 4.0) MODELİ VE TEST VERİSİNİ YÜKLE
# --------------------------------------------------------------------------
model = keras.models.load_model(cfg.MODEL_DOSYASI)
sinif_adlari = json.loads(cfg.SINIF_ADLARI_DOSYASI.read_text(encoding="utf-8"))

test_ds = keras.utils.image_dataset_from_directory(
    str(cfg.ISLENMIS_VERI_DIZINI / "test"),
    labels="inferred",
    label_mode="int",
    class_names=sinif_adlari,
    color_mode="grayscale",
    image_size=(cfg.ORIJINAL_BOYUT, cfg.ORIJINAL_BOYUT),
    batch_size=cfg.BATCH_BOYUTU,
    shuffle=False,  # Sıra sabit kalmalı ki dosya yolları ile tahminleri eşleştirebilelim
)
dosya_yollari = np.array(test_ds.file_paths)
X_test = np.concatenate([x.numpy() for x, _ in test_ds])  # (N, 200, 200, 1), 0-255 aralığında
y_test = np.concatenate([y.numpy() for _, y in test_ds])

# --------------------------------------------------------------------------
# 4.1) GENEL VE SINIF BAZLI METRİKLER
# --------------------------------------------------------------------------
olasiliklar = model.predict(X_test, verbose=0)
y_tahmin = olasiliklar.argmax(axis=1)
dogruluk = accuracy_score(y_test, y_tahmin)
print(f"TEST DOĞRULUĞU: {dogruluk:.4f}  ({(y_tahmin == y_test).sum()}/{len(y_test)})")

rapor_metni = classification_report(y_test, y_tahmin, target_names=sinif_adlari, digits=3)
print("\n" + rapor_metni)
rapor_dict = classification_report(y_test, y_tahmin, target_names=sinif_adlari, output_dict=True)
pd.DataFrame(rapor_dict).T.round(4).to_csv(cfg.RAPOR_DIZINI / "test_siniflandirma_raporu.csv", encoding="utf-8")

# --------------------------------------------------------------------------
# 4.2) KARIŞIKLIK MATRİSİ
# --------------------------------------------------------------------------
# Satır = gerçek sınıf, sütun = modelin tahmini. Köşegen dışı her sayı bir hatadır.
cm = confusion_matrix(y_test, y_tahmin)
fig, eksenler = plt.subplots(1, 2, figsize=(14, 5.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sinif_adlari, yticklabels=sinif_adlari, ax=eksenler[0])
eksenler[0].set_title("Karışıklık Matrisi (Adet)")
cm_oran = cm / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_oran, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
            xticklabels=sinif_adlari, yticklabels=sinif_adlari, ax=eksenler[1])
eksenler[1].set_title("Karışıklık Matrisi (Satır Bazlı Oran = Recall)")
for ax in eksenler:
    ax.set_xlabel("Modelin Tahmini")
    ax.set_ylabel("Gerçek Sınıf")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
fig.savefig(cfg.SEKIL_DIZINI / "07_karisiklik_matrisi.png", dpi=130)
plt.close(fig)

# --------------------------------------------------------------------------
# 4.3) HATALI TAHMİNLER
# --------------------------------------------------------------------------
hata_idx = np.where(y_tahmin != y_test)[0]
print(f"Toplam hatalı tahmin: {len(hata_idx)}")
if len(hata_idx):
    # Modelin en emin olduğu hataları öne al (güvenle yanılmak en tehlikelisi)
    guven = olasiliklar[hata_idx, y_tahmin[hata_idx]]
    sirali = hata_idx[np.argsort(-guven)][:12]
    sutun = 6
    satir = int(np.ceil(len(sirali) / sutun))
    fig, eksenler = plt.subplots(satir, sutun, figsize=(sutun * 2.4, satir * 2.8), squeeze=False)
    for ax in eksenler.ravel():
        ax.axis("off")
    for ax, i in zip(eksenler.ravel(), sirali):
        ax.imshow(X_test[i, ..., 0], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Gerçek: {sinif_adlari[y_test[i]]}\nTahmin: {sinif_adlari[y_tahmin[i]]}\n"
                     f"Güven: {olasiliklar[i, y_tahmin[i]]:.2f}", fontsize=8, color="crimson")
    plt.suptitle("Modelin En Emin Olduğu Hatalı Tahminler", y=1.0)
    plt.tight_layout()
    fig.savefig(cfg.SEKIL_DIZINI / "08_hatali_tahminler.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------------------------
# 4.4) GRAD-CAM - MODEL NEREYE BAKIYOR?
# --------------------------------------------------------------------------
# Sektörde bir modele güvenmek için sadece doğruluk yetmez: "Doğru nedenle mi
# doğru söylüyor?" sorusunu cevaplamak gerekir. Grad-CAM, son konvolüsyon
# katmanının hangi bölgelerinin karara en çok katkı verdiğini ısı haritası olarak
# gösterir. Model kusura bakıyorsa güvenebiliriz; arka plana bakıyorsa tehlike var.
def gradcam_hesapla(girdi_goruntu, hedef_sinif=None):
    """Tek bir görüntü (200,200,1) için Grad-CAM ısı haritası (0-1) ve tahmin döndürür."""
    # Modeldeki SON Conv2D katmanını bul (özellik haritası en zengin olduğu yer).
    son_conv = [k for k in model.layers if isinstance(k, keras.layers.Conv2D)][-1]
    # İki çıktılı yardımcı model: (son conv çıktısı, sınıf olasılıkları)
    yardimci = keras.Model(model.inputs, [son_conv.output, model.output])

    x = tf.convert_to_tensor(girdi_goruntu[None, ...], dtype=tf.float32)
    with tf.GradientTape() as bant:
        conv_cikti, tahmin = yardimci(x, training=False)
        if hedef_sinif is None:
            hedef_sinif = int(tf.argmax(tahmin[0]))
        skor = tahmin[:, hedef_sinif]
    # Hedef sınıf skorunun, son konvolüsyon haritalarına göre türevi
    turev = bant.gradient(skor, conv_cikti)
    # Her özellik haritasının önemi = türevin uzamsal ortalaması
    agirliklar = tf.reduce_mean(turev, axis=(1, 2))
    isi = tf.reduce_sum(conv_cikti * agirliklar[:, None, None, :], axis=-1)[0]
    isi = tf.nn.relu(isi).numpy()  # Sadece olumlu katkılar
    isi = isi / (isi.max() + 1e-8)
    isi = cv2.resize(isi, (cfg.ORIJINAL_BOYUT, cfg.ORIJINAL_BOYUT))
    return isi, hedef_sinif


# Her sınıftan DOĞRU bilinen 2 örnek seç
rng = np.random.default_rng(cfg.RASTGELE_TOHUM)
fig, eksenler = plt.subplots(cfg.SINIF_SAYISI, 4, figsize=(9, cfg.SINIF_SAYISI * 2.2))
for s in range(cfg.SINIF_SAYISI):
    dogru_idx = np.where((y_test == s) & (y_tahmin == s))[0]
    secilen = rng.choice(dogru_idx, size=min(2, len(dogru_idx)), replace=False)
    for k, i in enumerate(secilen):
        isi, _ = gradcam_hesapla(X_test[i])
        orijinal = X_test[i, ..., 0].astype(np.uint8)
        renkli_isi = cv2.applyColorMap(np.uint8(255 * isi), cv2.COLORMAP_JET)[..., ::-1]  # BGR -> RGB
        bindirme = (0.55 * np.stack([orijinal] * 3, -1) + 0.45 * renkli_isi).astype(np.uint8)
        eksenler[s, 2 * k].imshow(orijinal, cmap="gray", vmin=0, vmax=255)
        eksenler[s, 2 * k + 1].imshow(bindirme)
        eksenler[s, 2 * k].set_ylabel(sinif_adlari[s], fontsize=8, rotation=0, labelpad=45, va="center")
    for ax in eksenler[s]:
        ax.set_xticks([])
        ax.set_yticks([])
for j, baslik in enumerate(["Orijinal", "Grad-CAM", "Orijinal", "Grad-CAM"]):
    eksenler[0, j].set_title(baslik, fontsize=9)
plt.suptitle("Grad-CAM: Model Karar Verirken Nereye Bakıyor? (kırmızı = çok önemli)", y=1.0, fontsize=10)
plt.tight_layout()
fig.savefig(cfg.SEKIL_DIZINI / "09_gradcam.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------
# 4.5) BAZ ÇİZGİ İLE KARŞILAŞTIRMA
# --------------------------------------------------------------------------
baz_yolu = cfg.RAPOR_DIZINI / "baz_cizgi.json"
baz = json.loads(baz_yolu.read_text())["rf_3_istatistik_test_dogruluk"] if baz_yolu.exists() else float("nan")
print(f"\nKARŞILAŞTIRMA -> Rastgele: {1 / cfg.SINIF_SAYISI:.3f} | "
      f"3 istatistik + Random Forest: {baz:.3f} | CNN (Keras): {dogruluk:.3f}")

ozet = {
    "test_dogruluk": float(dogruluk),
    "hatali_tahmin_sayisi": int(len(hata_idx)),
    "test_ornek_sayisi": int(len(y_test)),
    "baz_cizgi_rf_dogruluk": float(baz),
    "model_dosyasi": str(cfg.MODEL_DOSYASI.relative_to(cfg.PROJE_KOKU)),
}
(cfg.RAPOR_DIZINI / "test_ozeti.json").write_text(json.dumps(ozet, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Grafikler: {cfg.SEKIL_DIZINI}")
