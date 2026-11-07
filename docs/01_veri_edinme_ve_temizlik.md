# 01 - Veriyi Edinme, Doğrulama ve Temizlik

> İlgili kod: [`src/01_veri_hazirla.py`](../src/01_veri_hazirla.py)  
> Çıktı: `data/processed/` (PNG klasörleri + `metadata.csv`)

## 1.1 Veri seti nedir?

**NEU Surface Defect Database (NEU-CLS)**: Kuzeydoğu Üniversitesi (Çin) tarafından yayımlanan, sıcak haddelenmiş çelik şerit yüzeylerindeki 6 tipik kusurun gri tonlamalı fotoğrafları.

| Özellik | Değer |
|---|---|
| Toplam görüntü | 1800 (1440 train + 360 test) |
| Sınıf sayısı | 6 |
| Görüntü boyutu | 200 × 200 piksel |
| Orijinal sınıf başına örnek | 300 |
| Görev | Çok sınıflı görüntü sınıflandırma |

**Kusur sınıfları:** `crazing` (ince çatlak ağı), `inclusion` (yabancı madde), `patches` (leke), `pitted_surface` (çukurlaşma), `rolled-in_scale` (ezilmiş oksit), `scratches` (çizik).

### Veri nereden alındı?
 Hugging Face aynasından (`newguyme/neu_cls`, parquet formatında) indirildi. Dosyalar `data/raw/` altındadır ve **asla değiştirilmez** (ham veri her zaman dokunulmaz tutulur; tüm dönüşümler `data/processed/` içine yazılır).

> **Atıf ve lisans notu:** Veri seti akademik bir kaynaktan gelir (K. Song & Y. Yan, 2013). Hugging Face aynasında açık bir lisans etiketi yok; bu yüzden proje **eğitim/öğrenme amaçlı** kullanılmalıdır. Ticari bir üründe kullanmadan önce orijinal sahiplerinden izin kontrol edilmelidir.

## 1.2 Neden bu veri seti? (Sektör bağlamı)

Yüzey kusuru tespiti, **üretimde kalite kontrolün** en yaygın yapay zekâ uygulamasıdır (çelik, otomotiv, tekstil, yarı iletken). Bir görüntü işleme mühendisi için doğrudan portföy değeri taşır ve şu soruyu sordurur: *klasik görüntü işleme (eşikleme, kenar bulma) mi, yoksa derin öğrenme mi?* Bu projenin sonunda bu sorunun cevabını **ölçümle** vereceğiz (bkz. [04](04_degerlendirme.md)).

## 1.3 Kalite kontrol: model kurmadan önce neleri doğruladık?

Her kontrol bir **soruya** cevap verir. Sonuçlar gerçek çalıştırma çıktısıdır:

| # | Kontrol | Neden önemli? | Sonuç |
|---|---|---|---|
| 1 | Bozuk/okunamayan dosya var mı? | Eğitim sırasında çökmeye yol açar | **0 bozuk** |
| 2 | Tüm görüntüler aynı boyutta mı? | CNN sabit boyutlu girdi ister | **Hepsi 200×200** |
| 3 | RGB'nin 3 kanalı aynı mı? | Aynıysa veri aslında gri; 3 kanal gereksiz bellek | **%100 aynı** → tek kanala indirildi |
| 4 | Sınıflar dengeli mi? | Dengesizlik accuracy'yi yanıltır | **Neredeyse mükemmel dengeli** |
| 5 | Tekrar eden görüntü var mı? | Kopyalar ezberi ödüllendirir | **2 görüntü aynıydı** (aynı bölüm içinde) → 1'i silindi |
| 6 | Train ile test arasında ortak görüntü var mı? | **Veri sızıntısı** — test başarısını şişirir | **0 ortak görüntü** (temiz) |

### Kontrol 3 ayrıntısı: RGB → gri
Ham dosyalar RGB olarak kaydedilmişti ama üç kanal piksel piksel aynıydı. Üç kanalı tutmak belleği 3 katına çıkarır ve modele hiçbir bilgi eklemez. Bu yüzden görüntüleri **tek kanallı (grayscale) PNG** olarak kaydettik.

### Kontrol 5-6 ayrıntısı: Tekrar ve sızıntı (neden kritik?)
Her görüntünün piksel içeriğinden bir **MD5 parmak izi** hesaplanır. Aynı parmak izine sahip iki görüntü piksel piksel aynıdır.

- **Train ↔ test ortaklığı** varsa model, test'te daha önce gördüğü görüntüyü tanır; raporlanan başarı gerçek dünyada tutmaz. Bizde **sıfır** çıktı, iyi haber.
- **Train içi tekrar:** `patches` sınıfında bir görüntü iki kez geçiyordu. Biri silindi. Bu yüzden `patches` train'de 239, diğer sınıflar 240 örnek.

> **Ders:** "Veri temiz" varsayımıyla ilerlemek yerine **ölçtük**. Bu kontrol 10 saniye sürdü; atlansaydı hata sessizce sonuçlara karışabilirdi.

## 1.4 Çıktı klasör yapısı

```
data/processed/
├── train/
│   ├── crazing/            crazing_0001.png ...
│   ├── inclusion/
│   ├── patches/
│   ├── pitted_surface/
│   ├── rolled-in_scale/
│   └── scratches/
├── test/                   (aynı 6 klasör)
└── metadata.csv            her görüntü için: sınıf, boyut, parlaklık, kontrast, netlik, md5, yol
```

Klasör-başına-sınıf yapısı, Keras'ın `image_dataset_from_directory` fonksiyonunun beklediği standart düzendir: etiketler klasör adından otomatik çıkarılır.

## 1.5 Etiket eşlemesi nasıl doğrulandı?

Veri setinde etiketler `0..5` tamsayı olarak geliyordu; **sınıf isimleri verilmemişti**. Alfabetik sıra (NEU-CLS standardı) varsayıldı, ancak varsayımla yetinmedik: her etiketten örnek görüntüler çizilip gözle kontrol edildi (ör. `label=5` parlak çizgiler → `scratches`, `label=2` büyük koyu lekeler → `patches`). Eşleme [`src/config.py`](../src/config.py) içinde `SINIF_ADLARI` listesindedir.

> Bu eşleme, orijinal veri sahiplerinin dokümanıyla ayrıca teyit edilmemiştir; gözle doğrulama yapılmıştır. Sonuçları yorumlarken bu bilinmelidir.
