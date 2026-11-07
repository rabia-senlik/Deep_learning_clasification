# 03 - CNN Mimarisi, Veri Hattı ve Eğitim

> İlgili kod: [`src/03_model_egit.py`](../src/03_model_egit.py)  
> Çıktılar: `models/neu_cnn_baseline.keras`, `reports/figures/06_egitim_egrileri.png`, `reports/egitim_gecmisi.csv`

## 3.1 Veri Hattı (tf.data Pipeline)

Veriyi doğrudan Python döngüsüyle okumak yerine `tf.data` API'si ile optimize edilmiş bir asenkron hat kuruldu:

1. **`cache()`:** İlk epoch'ta görüntüler diskten okunup çözüldükten sonra RAM'de önbelleğe alınır. Böylece sonraki epoch'larda disk I/O darboğazı yaşanmaz.
2. **`shuffle(buffer_size, reshuffle_each_iteration=True)`:** Her epoch'ta örnekler yeniden karıştırılır. Model sıralama ezberleyemez.
3. **`batch(32)`:** 32'lik gruplar halinde işlemciye sunulur.
4. **`prefetch(AUTOTUNE)`:** Model o anki batch'i eğitirken, işlemcinin boş kalan çekirdekleri arka planda bir sonraki batch'i hazırlar. Donanım boşta beklemez.

## 3.2 Model Mimarisi: Neden Bu Katmanlar Seçildi?

Modelimiz Keras Functional API kullanılarak inşa edilmiştir:

```text
Giriş: (200, 200, 1) Gri Tonlamalı Ham Görüntü
  │
  ├── [Resizing (128x128)] + [Rescaling (1/255)]  (Model içi ön işleme)
  │
  ├── [Data Augmentation] (RandomFlip, RandomTranslation, RandomContrast)
  │
  ├── [Blok 1] Conv2D(32) -> Conv2D(32) -> MaxPool2D -> Dropout(0.2)
  ├── [Blok 2] Conv2D(64) -> Conv2D(64) -> MaxPool2D -> Dropout(0.2)
  ├── [Blok 3] Conv2D(128) -> Conv2D(128) -> MaxPool2D -> Dropout(0.2)
  │
  ├── GlobalAveragePooling2D (Flatten yerine parametre tasarrufu)
  ├── Dropout(0.3)
  ├── Dense(64, activation='relu')
  └── Dense(6, activation='softmax') -> 6 Sınıf Olasılığı
```

### Kritik Mühendislik Kararları:
1. **Ön İşleme ve Augmentation Modelin İçinde:** Model canlıya alındığında dışarıdan gelen ham uint8 200x200 resmi doğrudan kabul eder; harici ön işleme koduna bağımlılık ortadan kalkar.
2. **Neden BatchNormalization Çıkarıldı?:** İlk denemelerde 6 adet derin BatchNorm katmanı, küçük veri seti (1.152 eğitim resmi) ve CPU ortamında `training=True` ile `training=False` arasında istatistik uyuşmazlığı yarattı (`val_loss` 6'ya fırlayıp model tek sınıfa kilitlendi). BatchNorm yerine klasik kararlı VGG mimarisi ve Dropout düzenlileştirmesi kullanılarak eğitim anında rayına oturtuldu.
3. **GlobalAveragePooling2D Avantajı:** 16x16x128 özellik haritasını tek hamlede 128 boyuta indirir. Klasik `Flatten`'ın yaratacağı 32.768 nöronu ve milyonlarca ağırlığı engelleyerek aşırı ezberlemeyi (overfitting) yok eder.

## 3.3 Callback'ler ve Eğitim Dinamikleri

* **`EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)`:** Doğrulama hatasını izler, aşırı öğrenme başladığında eğitimi keser ve en iyi epoch ağırlıklarını yükler.
* **`ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5)`:** Öğrenme tıkandığında öğrenme hızını yarıya indirerek ağırlıkların en optimum çukura oturmasını sağlar.
* **`ModelCheckpoint(save_best_only=True)`:** Sadece yeni bir doğrulama rekoru geldiğinde modeli diske yazar.

## 3.4 Eğitim Sonuçları

* **Eğitim Doğruluğu:** %95.57
* **Doğrulama Doğruluğu (Validation Accuracy):** %94.08
* **En İyi Validation Loss:** 0.1467
* Eğitim ve doğrulama eğrileri paralel seyretmiş, ezber makası açılmamıştır.
