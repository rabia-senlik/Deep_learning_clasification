"""
Proje genelinde kullanılan ortak ayarlar.

Tüm yollar, sabitler ve hiperparametreler tek yerde tutulur. Böylece bir değeri
değiştirmek istediğimizde (ör. görüntü boyutu) sadece bu dosyayı düzenleriz.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# 1) KLASÖR YOLLARI
# --------------------------------------------------------------------------
# Bu dosya  <proje>/src/config.py  konumunda; proje kökü bir üst klasördür.
PROJE_KOKU = Path(__file__).resolve().parent.parent

HAM_VERI_DIZINI = PROJE_KOKU / "data" / "raw"            # İndirilen orijinal dosyalar (dokunulmaz)
ISLENMIS_VERI_DIZINI = PROJE_KOKU / "data" / "processed"  # PNG klasörleri + temizlenmiş veri
MODEL_DIZINI = PROJE_KOKU / "models"                      # Eğitilmiş modeller
RAPOR_DIZINI = PROJE_KOKU / "reports"                     # Metrik tabloları (csv/json)
SEKIL_DIZINI = RAPOR_DIZINI / "figures"                   # Grafikler (png)

for _dizin in (ISLENMIS_VERI_DIZINI, MODEL_DIZINI, RAPOR_DIZINI, SEKIL_DIZINI):
    _dizin.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# 2) VERİ SETİ BİLGİSİ
# --------------------------------------------------------------------------
HAM_TRAIN_PARQUET = HAM_VERI_DIZINI / "data_train-00000-of-00001.parquet"
HAM_TEST_PARQUET = HAM_VERI_DIZINI / "data_test-00000-of-00001.parquet"

# Etiket (0-5) -> sınıf adı eşlemesi.
# Alfabetik sırayı kullanır (NEU-CLS standardı). Örnek görüntüler elle incelenerek
# doğrulanmıştır: 0=ince çatlak ağı, 1=uzun koyu şeritler, 2=büyük lekeler,
# 3=küçük çukurlar, 4=dağınık koyu benekler, 5=parlak çizgiler.
SINIF_ADLARI = [
    "crazing",          # 0 - Yüzeyde ince çatlak ağı
    "inclusion",        # 1 - Metal içine hapsolmuş yabancı madde (şeritler)
    "patches",          # 2 - Büyük lekeler
    "pitted_surface",   # 3 - Küçük çukurcuklar
    "rolled-in_scale",  # 4 - Hadde sırasında yüzeye ezilmiş oksit tabakası
    "scratches",        # 5 - Çizikler
]
SINIF_SAYISI = len(SINIF_ADLARI)

# --------------------------------------------------------------------------
# 3) MODEL / EĞİTİM AYARLARI
# --------------------------------------------------------------------------
RASTGELE_TOHUM = 42          # Tekrarlanabilirlik (aynı kodu çalıştırınca aynı sonuç)
ORIJINAL_BOYUT = 200         # Ham görüntü boyutu (modelin kabul ettiği giriş boyutu)
GORUNTU_BOYUTU = 128         # Model içinde 200x200 -> 128x128'e küçültülür (CPU'da hızlı eğitim)
KANAL_SAYISI = 1             # Görüntüler gri tonlamalı (tek kanal)
DOGRULAMA_ORANI = 0.20       # Eğitim verisinin %20'si doğrulama (validation) için ayrılır
BATCH_BOYUTU = 32
MAKSIMUM_EPOCH = 60          # EarlyStopping zaten daha erken durduracak
OGRENME_ORANI = 1e-3

MODEL_DOSYASI = MODEL_DIZINI / "neu_cnn_baseline.keras"
SINIF_ADLARI_DOSYASI = MODEL_DIZINI / "class_names.json"
