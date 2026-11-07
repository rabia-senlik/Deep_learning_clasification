"""
ADIM 3 - VERİ HATTI (PIPELINE), CNN MODELİ VE EĞİTİM
====================================================
Bu betik şunları yapar:
  3.1  Eğitim verisini train / validation olarak böler (test'e DOKUNMAZ)
  3.2  tf.data hattı kurar (önbellek + önceden yükleme = hızlı eğitim)
  3.3  Keras ile CNN kurar (ön işleme + augmentation modelin İÇİNDE)
  3.4  Callback'lerle eğitir (EarlyStopping, ReduceLROnPlateau, Checkpoint)
  3.5  Eğitim eğrilerini çizer ve en iyi modeli models/ klasörüne kaydeder

Çalıştırma:  python src/03_model_egit.py
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # TensorFlow'un gürültülü loglarını sustur

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers

import config as cfg

# Aynı kodu tekrar çalıştırınca aynı sonucu almak için tüm rastgelelikleri sabitle.
keras.utils.set_random_seed(cfg.RASTGELE_TOHUM)

# --------------------------------------------------------------------------
# 3.1) VERİ YÜKLEME VE TRAIN / VALIDATION BÖLME
# --------------------------------------------------------------------------
# Veri 3 parçaya ayrılır ve her parçanın görevi farklıdır:
#   TRAIN      -> modelin öğrendiği veri
#   VALIDATION -> eğitim sırasında "iyi gidiyor muyuz?" kontrolü (EarlyStopping buna bakar)
#   TEST       -> SADECE EN SONDA, bir kez, gerçek başarıyı ölçmek için (Adım 4)
# Test'e burada dokunmayız; aksi halde model seçimi test'e göre yapılmış olur
# ve raporladığımız başarı gerçeği yansıtmaz (veri sızıntısı).
egitim_dizini = cfg.ISLENMIS_VERI_DIZINI / "train"
ortak_ayarlar = dict(
    directory=str(egitim_dizini),
    labels="inferred",              # Etiketleri klasör adlarından çıkar
    label_mode="int",               # Etiketler 0..5 tamsayı (sparse_categorical_crossentropy için)
    class_names=cfg.SINIF_ADLARI,   # Sınıf sırasını sabitle (alfabetik karışıklığı önler)
    color_mode="grayscale",         # Tek kanal
    image_size=(cfg.ORIJINAL_BOYUT, cfg.ORIJINAL_BOYUT),
    batch_size=None,                # Batch'leme işini aşağıda biz yapıyoruz (örnek sayısı bilinsin diye)
    validation_split=cfg.DOGRULAMA_ORANI,
    seed=cfg.RASTGELE_TOHUM,        # Train ve val aynı tohumla bölünmeli, yoksa kesişirler!
    # ÖNEMLİ: shuffle=True OLMAK ZORUNDA. shuffle=False iken Keras dosyaları alfabetik
    # sırayla keser: doğrulama seti sadece son sınıflardan (ör. scratches), eğitim seti
    # ise geri kalanlardan oluşur. Bu durumda model bazı sınıfları hiç görmeden
    # doğrulanır ve val_accuracy ~0 çıkar. (Bu hatayı ilk denemede yaşadık.)
    shuffle=True,
)
egitim_ds = keras.utils.image_dataset_from_directory(subset="training", **ortak_ayarlar)
dogrulama_ds = keras.utils.image_dataset_from_directory(subset="validation", **ortak_ayarlar)

# Bölmenin sağlıklı olduğunu DOĞRULA: her sınıf hem train'de hem val'de bulunmalı.
n_egitim = int(egitim_ds.cardinality())
n_dogrulama = int(dogrulama_ds.cardinality())
egitim_etiketleri = np.array([y for _, y in egitim_ds.as_numpy_iterator()])
dogrulama_etiketleri = np.array([y for _, y in dogrulama_ds.as_numpy_iterator()])
egitim_dagilim = np.bincount(egitim_etiketleri, minlength=cfg.SINIF_SAYISI)
dogrulama_dagilim = np.bincount(dogrulama_etiketleri, minlength=cfg.SINIF_SAYISI)
print(f"Eğitim örnek sayısı: {n_egitim}  | sınıf dağılımı: {egitim_dagilim.tolist()}")
print(f"Doğrulama örnek sayısı: {n_dogrulama}  | sınıf dağılımı: {dogrulama_dagilim.tolist()}")
assert (egitim_dagilim > 0).all() and (dogrulama_dagilim > 0).all(), "Bir sınıf train veya val'de eksik!"

# --------------------------------------------------------------------------
# 3.2) tf.data PERFORMANS AYARI
# --------------------------------------------------------------------------
# cache()    : Görüntüler ilk epoch'ta diskten okunur, sonra RAM'den gelir.
# shuffle()  : Her epoch'ta örnekler yeniden karıştırılır.
# batch()    : Örnekler batch'lere gruplanır.
# prefetch() : GPU/CPU bir batch'i işlerken bir sonraki batch hazırlanır.
#
# SIRA ÖNEMLİ: cache() -> shuffle() -> batch(). Karıştırılmış batch'leri cache'lersek
# her epoch'ta AYNI batch'ler aynı sırada gelir ve karıştırmanın faydası kaybolur.
AUTOTUNE = -1  # tf.data.AUTOTUNE
egitim_ds = (
    egitim_ds.cache()
    .shuffle(n_egitim, seed=cfg.RASTGELE_TOHUM, reshuffle_each_iteration=True)
    .batch(cfg.BATCH_BOYUTU)
    .prefetch(AUTOTUNE)
)
dogrulama_ds = dogrulama_ds.cache().batch(cfg.BATCH_BOYUTU).prefetch(AUTOTUNE)  # Doğrulamada karıştırma gerekmez


# --------------------------------------------------------------------------
# 3.3) CNN MODELİ (Keras Functional API)
# --------------------------------------------------------------------------
def konvolusyon_blogu(x, filtre_sayisi):
    """
    Conv -> ReLU -> Conv -> ReLU -> MaxPool -> Dropout
    
    Küçük veri setlerinde (1000-2000 görüntü) çok sayıda BatchNormalization
    katmanı moving_mean/variance istatistiklerinde uyumsuzluğa yol açıp
    val_loss'u patlatabilir. Bu yüzden klasik, kararlı ve hızlı VGG tarzı
    blok mimarisi tercih edilir.
    """
    x = layers.Conv2D(filtre_sayisi, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filtre_sayisi, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)
    return x


def model_kur():
    giris = keras.Input(shape=(cfg.ORIJINAL_BOYUT, cfg.ORIJINAL_BOYUT, cfg.KANAL_SAYISI), name="ham_goruntu")

    # --- Ön işleme (modelin parçası: ileride ham görüntü doğrudan verilebilir) ---
    x = layers.Resizing(cfg.GORUNTU_BOYUTU, cfg.GORUNTU_BOYUTU, name="yeniden_boyutlandir")(giris)
    x = layers.Rescaling(1.0 / 255, name="normalizasyon")(x)  # 0-255 -> 0-1

    # --- Veri artırma (YALNIZCA eğitimde aktif; tahminde otomatik kapanır) ---
    # Kusurun konumu görüntüden görüntüye değiştiği için (EDA 2.4'te gördük)
    # çevirme ve küçük kaydırma güvenlidir. Büyük döndürme KULLANMIYORUZ: çizik
    # (scratch) gibi sınıflarda yön bilgisi anlamlı olabilir.
    x = layers.RandomFlip("horizontal_and_vertical", name="aug_cevirme")(x)
    x = layers.RandomTranslation(0.08, 0.08, fill_mode="reflect", name="aug_kaydirma")(x)
    x = layers.RandomContrast(0.15, name="aug_kontrast")(x)

    # --- Özellik çıkarıcı: 3 konvolüsyon bloğu (filtre sayısı 32 -> 64 -> 128) ---
    for filtre in (32, 64, 128):
        x = konvolusyon_blogu(x, filtre)

    # --- Sınıflandırıcı kafa ---
    # GlobalAveragePooling: Flatten'a göre çok daha az parametre, daha az ezber.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    cikis = layers.Dense(cfg.SINIF_SAYISI, activation="softmax", name="sinif_olasiliklari")(x)

    return keras.Model(giris, cikis, name="neu_cnn_baseline")


model = model_kur()
model.summary()

# --------------------------------------------------------------------------
# 3.4) DERLEME VE CALLBACK'LER
# --------------------------------------------------------------------------
model.compile(
    optimizer=keras.optimizers.Adam(cfg.OGRENME_ORANI),
    loss="sparse_categorical_crossentropy",  # Etiketler tamsayı (0-5) olduğu için "sparse"
    metrics=["accuracy"],
)

callbackler = [
    # val_loss 8 epoch iyileşmezse eğitimi durdur ve EN İYİ ağırlıklara geri dön.
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    # val_loss 4 epoch iyileşmezse öğrenme oranını yarıya indir (ince ayar).
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
    # Her iyileşmede modeli diske yaz (eğitim yarıda kesilse bile en iyisi elimizde kalır).
    keras.callbacks.ModelCheckpoint(str(cfg.MODEL_DOSYASI), monitor="val_loss", save_best_only=True, verbose=0),
]

# --------------------------------------------------------------------------
# 3.5) EĞİTİM
# --------------------------------------------------------------------------
gecmis = model.fit(
    egitim_ds,
    validation_data=dogrulama_ds,
    epochs=cfg.MAKSIMUM_EPOCH,
    callbacks=callbackler,
    shuffle=False,  # Karıştırmayı tf.data hattında yaptık; burada tekrar gerekmez
    verbose=2,
)

# Sınıf adlarını modelin yanına kaydet: ileride model yüklenince 0..5 -> isim çevirisi için.
cfg.SINIF_ADLARI_DOSYASI.write_text(json.dumps(cfg.SINIF_ADLARI, ensure_ascii=False, indent=2), encoding="utf-8")

# Eğitim geçmişini CSV olarak sakla
gecmis_df = pd.DataFrame(gecmis.history)
gecmis_df.index.name = "epoch"
gecmis_df.to_csv(cfg.RAPOR_DIZINI / "egitim_gecmisi.csv", encoding="utf-8")

# --------------------------------------------------------------------------
# 3.6) EĞİTİM EĞRİLERİ - Öğrenme sağlıklı mı?
# --------------------------------------------------------------------------
en_iyi_epoch = int(np.argmin(gecmis.history["val_loss"]))
fig, eksenler = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, metrik, baslik in zip(eksenler, ["loss", "accuracy"], ["Kayıp (Loss)", "Doğruluk (Accuracy)"]):
    ax.plot(gecmis.history[metrik], label="Eğitim", lw=2)
    ax.plot(gecmis.history[f"val_{metrik}"], label="Doğrulama", lw=2)
    ax.axvline(en_iyi_epoch, color="gray", ls="--", label=f"En iyi epoch = {en_iyi_epoch + 1}")
    ax.set_title(baslik)
    ax.set_xlabel("Epoch")
    ax.legend()
plt.tight_layout()
fig.savefig(cfg.SEKIL_DIZINI / "06_egitim_egrileri.png", dpi=130)
plt.close(fig)

print(f"\nEn iyi epoch: {en_iyi_epoch + 1}")
print(f"En iyi val_loss: {min(gecmis.history['val_loss']):.4f}")
print(f"O epoch'taki val_accuracy: {gecmis.history['val_accuracy'][en_iyi_epoch]:.4f}")
print(f"Model kaydedildi: {cfg.MODEL_DOSYASI}")
