"""
ADIM 1 - VERİ HAZIRLAMA VE KALİTE KONTROL
=========================================
Ham (parquet) veriyi okur, kalite kontrolünden geçirir ve Keras'ın
`image_dataset_from_directory` fonksiyonunun beklediği klasör yapısına çevirir:

    data/processed/train/<sinif_adi>/*.png
    data/processed/test/<sinif_adi>/*.png

Ayrıca her görüntü için bir "metadata.csv" üretir (EDA adımında kullanılacak).

Çalıştırma:  python src/01_veri_hazirla.py
"""
import hashlib
import io

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import config as cfg


def parquet_oku(yol, bolum_adi):
    """Parquet dosyasını okur; her satırda bir PNG (bayt) ve bir etiket vardır."""
    df = pd.read_parquet(yol)
    df["bolum"] = bolum_adi
    return df


def goruntuyu_coz(bayt_verisi):
    """
    PNG baytlarını gri tonlamalı (tek kanal) numpy dizisine çevirir.

    Ham görüntüler RGB olarak saklanmış olsa da çelik yüzey fotoğrafları aslında
    gri tonlamalıdır (3 kanal birbirinin aynısı). Bunu aşağıda doğrulayacağız.
    """
    rgb = np.array(Image.open(io.BytesIO(bayt_verisi)).convert("RGB"))
    return rgb


def main():
    # ----------------------------------------------------------------------
    # 1.1) HAM VERİYİ OKU
    # ----------------------------------------------------------------------
    egitim_df = parquet_oku(cfg.HAM_TRAIN_PARQUET, "train")
    test_df = parquet_oku(cfg.HAM_TEST_PARQUET, "test")
    tum_df = pd.concat([egitim_df, test_df], ignore_index=True)
    print(f"Toplam görüntü: {len(tum_df)}  (train={len(egitim_df)}, test={len(test_df)})")

    # ----------------------------------------------------------------------
    # 1.2) KALİTE KONTROL: her görüntüyü çöz ve özelliklerini ölç
    # ----------------------------------------------------------------------
    kayitlar = []
    bozuk_sayisi = 0
    kanallar_ayni_mi = []

    for sira, satir in tum_df.iterrows():
        try:
            rgb = goruntuyu_coz(satir["image"]["bytes"])
        except Exception as hata:  # Bozuk dosya yakalanırsa atlarız
            bozuk_sayisi += 1
            print(f"  [UYARI] Bozuk görüntü atlandı (satır {sira}): {hata}")
            continue

        # RGB'nin 3 kanalı birbirine eşit mi? (Eşitse gerçekte gri tonlamalıdır.)
        kanallar_ayni_mi.append(
            np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])
        )
        gri = rgb[..., 0]  # Üç kanal aynı olduğu için birini almak yeterli

        etiket = int(satir["label"])
        sinif = cfg.SINIF_ADLARI[etiket]

        kayitlar.append(
            {
                "bolum": satir["bolum"],
                "etiket": etiket,
                "sinif": sinif,
                "yukseklik": gri.shape[0],
                "genislik": gri.shape[1],
                # Piksel istatistikleri (EDA'da sınıflar arası farka bakacağız)
                "ortalama_parlaklik": float(gri.mean()),
                "std_kontrast": float(gri.std()),
                "min_piksel": int(gri.min()),
                "max_piksel": int(gri.max()),
                # Laplace varyansı: görüntü netliği/doku zenginliği ölçüsü.
                # Yüksek değer = çok kenar/doku, düşük değer = bulanık/düz yüzey.
                "netlik_laplace": float(cv2.Laplacian(gri, cv2.CV_64F).var()),
                # Aynı görüntünün iki kez bulunup bulunmadığını anlamak için parmak izi
                "md5": hashlib.md5(gri.tobytes()).hexdigest(),
                "_gri": gri,  # Kaydetmek için geçici olarak tutulur, CSV'ye yazılmaz
            }
        )

    meta = pd.DataFrame(kayitlar)
    print(f"Bozuk görüntü sayısı: {bozuk_sayisi}")
    print(f"3 kanalı birbirinin aynısı olan görüntü oranı: {np.mean(kanallar_ayni_mi):.1%}")
    print(f"Benzersiz boyutlar: {meta[['yukseklik', 'genislik']].drop_duplicates().values.tolist()}")

    # ----------------------------------------------------------------------
    # 1.3) TEKRAR EDEN GÖRÜNTÜ (DUPLICATE) VE VERİ SIZINTISI KONTROLÜ
    # ----------------------------------------------------------------------
    # Aynı görüntü hem train hem test'te varsa model test'te "kopya çekmiş" olur
    # ve başarısı olduğundan yüksek görünür. Bu yüzden MD5 parmak izine bakarız.
    tekrar_sayisi = meta["md5"].duplicated(keep=False).sum()
    train_hash = set(meta.loc[meta.bolum == "train", "md5"])
    test_hash = set(meta.loc[meta.bolum == "test", "md5"])
    sizinti = train_hash & test_hash
    print(f"Tekrar eden (aynı piksel içeriği) görüntü sayısı: {tekrar_sayisi}")
    print(f"Train ile test arasında ortak görüntü sayısı (sızıntı): {len(sizinti)}")

    # Sızıntı varsa, test'teki kopyaları çıkarırız (train'e dokunmayız).
    if sizinti:
        maske = (meta["bolum"] == "test") & (meta["md5"].isin(sizinti))
        print(f"  -> Test setinden {maske.sum()} kopya görüntü çıkarılıyor.")
        meta = meta[~maske].reset_index(drop=True)

    # Aynı bölüm içindeki tam kopyaları da at (ilkini tut).
    onceki = len(meta)
    meta = meta.drop_duplicates(subset=["bolum", "md5"], keep="first").reset_index(drop=True)
    if len(meta) != onceki:
        print(f"  -> Bölüm içi {onceki - len(meta)} tekrar eden görüntü silindi.")

    # ----------------------------------------------------------------------
    # 1.4) GÖRÜNTÜLERİ SINIF KLASÖRLERİNE PNG OLARAK KAYDET
    # ----------------------------------------------------------------------
    dosya_yollari = []
    for bolum in ("train", "test"):
        for sinif in cfg.SINIF_ADLARI:
            (cfg.ISLENMIS_VERI_DIZINI / bolum / sinif).mkdir(parents=True, exist_ok=True)

    sayac = {}
    for sira, satir in meta.iterrows():
        anahtar = (satir["bolum"], satir["sinif"])
        sayac[anahtar] = sayac.get(anahtar, 0) + 1
        dosya_adi = f"{satir['sinif']}_{sayac[anahtar]:04d}.png"
        yol = cfg.ISLENMIS_VERI_DIZINI / satir["bolum"] / satir["sinif"] / dosya_adi
        cv2.imwrite(str(yol), satir["_gri"])  # Tek kanallı 8-bit PNG
        dosya_yollari.append(str(yol.relative_to(cfg.PROJE_KOKU)))

    meta["dosya_yolu"] = dosya_yollari

    # ----------------------------------------------------------------------
    # 1.5) METADATA TABLOSUNU KAYDET (piksel dizisi hariç)
    # ----------------------------------------------------------------------
    meta_csv = meta.drop(columns=["_gri"])
    meta_csv.to_csv(cfg.ISLENMIS_VERI_DIZINI / "metadata.csv", index=False, encoding="utf-8")

    print("\nSınıf dağılımı (bölüme göre):")
    print(meta_csv.pivot_table(index="sinif", columns="bolum", values="etiket", aggfunc="count"))
    print(f"\nKaydedildi: {cfg.ISLENMIS_VERI_DIZINI / 'metadata.csv'}")


if __name__ == "__main__":
    main()
