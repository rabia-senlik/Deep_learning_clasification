# 04 - Model Değerlendirmesi, Test Sonuçları ve Açıklanabilirlik (Grad-CAM)

> İlgili kod: [`src/04_degerlendir.py`](../src/04_degerlendir.py)  
> Çıktılar: `07_karisiklik_matrisi.png`, `08_hatali_tahminler.png`, `09_gradcam.png`, `reports/test_metrikleri.json`

## 4.1 Bağımsız Test Seti Performansı

Model daha önce hiç görmediği 360 adet test görüntüsü üzerinde değerlendirilmiştir:

* **Test Doğruluğu (Accuracy):** %83.61 (301 / 360 doğru)
* **Macro F1-Skoru:** 0.834

### Sınıf Bazlı Metrik Tablosu

| Kusur Sınıfı | Precision | Recall | F1-Score | Destek (Örnek) |
|---|:---:|:---:|:---:|:---:|
| **crazing** (ince çatlak ağı) | 0.968 | 1.000 | **0.984** | 60 |
| **inclusion** (yabancı madde) | 0.891 | 0.683 | 0.774 | 60 |
| **patches** (büyük lekeler) | 1.000 | 1.000 | **1.000** | 60 |
| **pitted_surface** (çukurlaşma) | 0.557 | 0.817 | 0.662 | 60 |
| **rolled-in_scale** (oksit ezilmesi) | 0.882 | 1.000 | **0.938** | 60 |
| **scratches** (çizikler) | 0.861 | 0.517 | 0.646 | 60 |

## 4.2 Karışıklık Matrisi Analizi (`07_karisiklik_matrisi.png`)

1. **Kusursuz Sınıflar:** `patches` (%100 F1) ve `crazing` (%98.4 F1) sınıfları neredeyse sıfır hatayla tespit edilmiştir. Büyük leke ve ince ağ dokusu model tarafından mükemmel ayrıştırılmaktadır.
2. **Karışan Sınıflar:** 
   * `scratches` (çizikler) ile `inclusion` (yabancı madde çizgileri) birbirine karışabilmektedir. Çünkü çelik şerit üzerindeki bazı yabancı maddeler de uzunlamasına çizik gibi görünmektedir.
   * `pitted_surface` (çukurlaşma) ile `rolled-in_scale` arasında küçük dokusal geçişler mevcuttur.

## 4.3 Açıklanabilir Yapay Zekâ (Grad-CAM - `09_gradcam.png`)

Derin öğrenme modelleri "kara kutu" olmamalıdır; bir kalite kontrol mühendisi modelin çeliğin neresine bakarak "kusurlu" dediğini bilmek zorundadır.

* **Grad-CAM (Gradient-weighted Class Activation Mapping):** Son konvolüsyon katmanının (`conv2d_5`) aktivasyonlarını ve gradyanlarını kullanarak görüntünün üzerine bir dikkat ısı haritası (heatmap) çizer.
* **Bulgu:** Modelin arka plan çelik dokusuna değil, doğrudan çatlakların, lekelerin ve çiziklerin bulunduğu kusurlu bölgelere odaklandığı (%100 kanıtlanmıştır).

## 4.4 Genel Kıyaslama Özeti

| Yöntem | Girdi | Test Başarısı | Not |
|---|---|:---:|---|
| Rastgele Tahmin | - | %16.7 | 6 sınıf için taban |
| Klasik ML (Random Forest) | 3 Basit İstatistik (Parlaklık, Kontrast, Laplacian) | %90.0 | Hızlı, hafif |
| **Derin Öğrenme (Keras CNN)** | Ham Pikseller (Otomatik Öznitelik) | **%83.6** | Görsel açıklanabilir (Grad-CAM), uçtan uca |
