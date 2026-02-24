# A100 Statik Denetim Raporu (coderapor)

Bu rapor tamamen statik kod incelemesine dayanır. Kod çalıştırma/simülasyon yapılmamıştır.

## 1. ÖZET TABLO — En Kritik Bulgular


| #   | Dosya/Modül                                        | Satır/Fonksiyon           | Güncel Durum Özeti                                                      | Etki Türü                  | Durum / Risk |
| --- | -------------------------------------------------- | ------------------------- | ----------------------------------------------------------------------- | -------------------------- | ------------- |
| 1   | `gps_training/train.py`                            | `train()` autocast bölümü | CUDA koşullu autocast + CPU fallback eklendi                            | Çökme                      | FIXED / [KAPANDI] |
| 2   | `uav_training/train.py` + `uav_training/config.py` | `_train_single_phase()`   | BF16 (`amp_dtype`) train argümanına koşullu şekilde aktarılıyor         | Hız / Stabilite            | FIXED / [KAPANDI] |
| 3   | `scripts/colab_bootstrap.py`                       | Thread limiter            | Thread limitleri gevşetildi, ancak A100 için daha ileri tuning açık     | Hız                        | PARTIAL / [ORTA] |
| 4   | `gps_training/dataset.py`                          | `__getitem__()`           | `None` dönüşü kaldırıldı, fail-fast + structured logging eklendi        | Doğruluk / Çökme           | FIXED / [KAPANDI] |
| 5   | `gps_training/train.py`                            | Validation loop           | Validation tarafında BF16 autocast devreye alındı                       | Hız                        | FIXED / [KAPANDI] |
| 6   | `uav_training/train.py`                            | `checkpoint_guard()`      | Tek-flight lock eklendi, thread yarış riski düşürüldü                   | Çökme / MLOps              | PARTIAL / [DÜŞÜK-ORTA] |
| 7   | `gps_training/train.py`                            | Seed zinciri              | Global seed + determinism zinciri eklendi                               | Tekrarlanabilirlik         | FIXED / [KAPANDI] |
| 8   | `requirements.txt`                                 | Versiyon yönetimi         | Aralık daraltıldı, tam pinleme hâlâ yok                                 | Tekrarlanabilirlik / MLOps | OPEN / [ORTA] |


---

## 2. DETAYLI BUG AVI VE ÇÖZÜMLER

[BUG-1] CPU fallback'te autocast çökmesi  
  Durum         : FIXED  
  Etkilenen Dosya: `gps_training/train.py`  
  Güncel Risk   : [KAPANDI]
  Doğrulama: Eğitim ve validation tarafında autocast, `device.type == "cuda"` koşuluna bağlandı; CPU'da `nullcontext` fallback çalışıyor.

[BUG-2] UAV hattında BF16 hedefi train argümanına taşınmıyor  
  Durum         : FIXED  
  Etkilenen Dosya: `uav_training/train.py`, `uav_training/config.py`  
  Güncel Risk   : [KAPANDI]
  Doğrulama: `TRAIN_CONFIG["amp_dtype"] == "bf16"` ve BF16 HW desteği varsa `train_args["amp_dtype"]` set ediliyor; config tarafında da `amp_dtype` mevcut.

[BUG-3] A100 için CPU thread kısıtı aşırı agresif  
  Durum         : PARTIAL  
  Etkilenen Dosya: `scripts/colab_bootstrap.py`  
  Güncel Risk   : [ORTA]
  Doğrulama: Sabit `1` thread yaklaşımı kaldırıldı, dinamik thread hesaplama geldi. Ancak A100 üzerinde nihai değerler için runtime benchmark ile ek tuning hâlâ gerekli.

[BUG-4] Dataset exception sonrası `None` dönerek silent failure üretiyor  
  Durum         : FIXED  
  Etkilenen Dosya: `gps_training/dataset.py`  
  Güncel Risk   : [KAPANDI]
  Doğrulama: `__getitem__` içinde hata durumunda `None` dönmek yerine structured log + `RuntimeError` ile fail-fast uygulanıyor.

[BUG-5] Validation BF16 yolunu kullanmıyor  
  Durum         : FIXED  
  Etkilenen Dosya: `gps_training/train.py`  
  Güncel Risk   : [KAPANDI]
  Doğrulama: Validation loop `torch.no_grad()` altında BF16 autocast context içinde çalışacak şekilde güncellendi.

[BUG-6] `checkpoint_guard()` her 5 epoch yeni thread üretiyor  
  Durum         : PARTIAL  
  Etkilenen Dosya: `uav_training/train.py`  
  Güncel Risk   : [DÜŞÜK-ORTA]
  Doğrulama: Tek-flight lock (`_SYNC_LOCK`, `_SYNC_IN_FLIGHT`) eklendi, eşzamanlı sync yarışları azaltıldı. Yine de dönemsel olarak yeni thread oluşturma davranışı tamamen kaldırılmış değil.

[BUG-7] GPS seed zinciri eksik  
  Durum         : FIXED  
  Etkilenen Dosya: `gps_training/train.py`  
  Güncel Risk   : [KAPANDI]
  Doğrulama: Global seed zinciri (`PYTHONHASHSEED`, `random`, `numpy`, `torch`, `cuda`) eğitim başlangıcına eklendi; worker seed ile birlikte çalışıyor.

[BUG-8] `requirements.txt` versiyon yönetimi drift riski  
  Durum         : OPEN  
  Etkilenen Dosya: `requirements.txt`  
  Güncel Risk   : [ORTA]
  Doğrulama: Aralıklar daraltıldı ancak tam pin (`==`) modeline geçilmediği için uzun dönem tekrar üretilebilirlikte sürüm drift riski devam ediyor.

---

## 3. SİSTEM GENELİ OPTİMİZASYON STRATEJİSİ

### 3.1 Colab A100 I/O Mimarisi

- Mevcut akış: `Drive -> DataLoader -> GPU` (kritik noktada Drive/CPU beklemeleri oluşuyor).
- Optimize akış: `Drive -> /content SSD -> RAM Cache -> DataLoader -> A100`.
- Tahmini kazanım:
  - Drive yerine SSD’den eğitim: I/O bekleme süresi ~%60-85 azalır.
  - RAM sıcak cache (özellikle GPS frame pair): decode + seek overhead ~%20-45 azalır.
  - Pre-fetch ve worker tuning sonrası GPU idle süresi ~%40-70 azalır.

### 3.2 A100 GPU Kullanım Analizi

- Mevcut teorik kullanım (kod yapılarına göre):
  - UAV: `%70-90` (I/O kalitesine bağlı)
  - GPS: `%55-80` (frame decode + CPU dönüşümleri baskın)
- Darboğaz sınıfı:
  - UAV: I/O-bound + kısmi CPU-bound
  - GPS: CPU-bound + I/O-bound
- A100 saturation önerisi:
  - `num_workers=min(12, os.cpu_count())`, en az 8
  - `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4`, `drop_last=True`
  - BF16 explicit + TF32 açık + `torch.compile(..., mode="reduce-overhead")`

### 3.3 BF16 Geçiş Planı

- FP16/BELİRSİZ auto yolundan BF16’ye geçiş:
  1. UAV tarafında `train_args["amp_dtype"]="bf16"` explicit aktar.
  2. GPS tarafında autocast’ı CUDA koşullu hale getir.
  3. Validation tarafını da BF16 autocast içine al.
- GradScaler:
  - A100 + BF16’de çoğunlukla kaldırılması güvenli.
  - Sadece özel/custom loss veya legacy kernel numerik sorunlarında kontrollü tutulmalı.

### 3.4 Augmentation Değerlendirmesi

- UAV konfigürasyonunda küçük nesne odaklı yaklaşım genel olarak güçlü (`imgsz`, mosaic, scale ayarları).
- Risk noktası: `flipud=0.5` bazı UAV veri alanlarında fiziksel olarak gerçekçi olmayabilir.
- Öneri:
  - Değiştir: `flipud` -> `0.0-0.1` (domain'e göre)
  - Ekle: class-bazlı recall takibi (uap/uai/human/vehicle)
  - Çıkar/Azalt: domain dışı agresif renk manipülasyonları

### 3.5 Hiperparametre Uyumluluk Matrisi


| Alan        | Mevcut                   | Değerlendirme                           | A100 Önerisi                                |
| ----------- | ------------------------ | --------------------------------------- | ------------------------------------------- |
| Batch       | UAV explicit, GPS high   | GPS için LR scaling disiplini izlenmeli | Batch artışında LR lineer ölçekle           |
| LR / Warmup | `lr0`, `lrf`, warmup var | UAV tutarlı, GPS max_lr disiplini eksik | OneCycle max_lr değerini batch’e bağla      |
| Precision   | TF32 açık, BF16 explicit | Train/val tarafı büyük ölçüde hizalı    | BF16 runtime log denetimini standardize et  |
| Compile     | Kısmen var               | İyi başlangıç                           | A100’de default açık, OOM fallback’te kapat |
| Accumulate  | Açıkça yönetilmiyor      | Büyük efektif batch belirsiz            | VRAM’e göre `accumulate` hesapla/logla      |


### 3.6 Skor Yeniden Hesaplama Rubriği (Kod-Sinyal Bazlı)

- Puanlama yöntemi: her kategori için 0-10 arası skor, kodda gözlenen sinyallere göre yeniden hesaplandı.
- Sinyal ağırlıkları:
  - Güvenlik/stabilite kırılımı (crash, silent failure, checkpoint): %35
  - A100 verim sinyalleri (BF16, TF32, worker/thread, compile): %35
  - Tekrarlanabilirlik/MLOps (seed, versiyonlama, resume): %30
- Bu revizyonda bug dağılımı: `5 FIXED`, `2 PARTIAL`, `1 OPEN`.
- `GENEL` skoru, kategori skorlarının ağırlıklı ortalamasının 1 ondalık yuvarlanmış halidir.

---

## 4. HIZLI KAZANIM PLANI — 5 Dakika Aksiyonları

1. [TAMAMLANDI] GPU/CPU autocast güvenlik düzeltmesi + validation BF16 yolu aktif.
2. [TAMAMLANDI] UAV tarafında BF16 explicit (`amp_dtype`) train argümanına aktarılıyor.
3. [TAMAMLANDI] Dataset silent failure kaldırıldı; fail-fast davranış devrede.
4. [TAMAMLANDI] GPS global seed/determinism zinciri eklendi.
5. [AÇIK] `requirements.txt` için tam pin (`==`) tabanlı kilitleme dosyası (`requirements-lock.txt` veya benzeri) eklenmeli.
6. [KISMİ] A100 thread limiter tuning'i runtime benchmark ile finalize edilmeli (hedef: GPU idle süresini daha da düşürmek).

---

## 5. EĞİTİM SAĞLIK GÖSTERGESİ (Scorecard)


| Kategori                | Puan   | Kısa Yorum                                                       |
| ----------------------- | ------ | ---------------------------------------------------------------- |
| I/O Verimliliği         | 7/10   | SSD + DataLoader optimizasyonu güçlü; thread tuning kısmı kısmi  |
| A100 Kullanım Oranı     | 8/10   | TF32 açık, BF16 yolu explicit, compile/fallback stratejisi mevcut |
| BF16/Precision Kalitesi | 9/10   | Train/val BF16 akışı ve CUDA guard doğru uygulanmış               |
| Eğitim Stabilitesi      | 8/10   | Fail-fast dataset, non-finite loss kontrolü, atomic checkpoint var |
| Augmentation Kalitesi   | 7/10   | Küçük nesne odaklı iyi ayarlar var; `flipud` domain riski sürüyor |
| MLOps Olgunluğu         | 8/10   | Resume + checkpoint guard gelişti; sync modeli daha güvenli       |
| Tekrarlanabilirlik      | 7/10   | Global seed zinciri tamam; dependency tam pinleme eksik           |
| GENEL                   | 7.7/10 | Kritik buglar kapandı; kalan işler tuning ve versiyon kilitleme    |


---

## 6. VARSAYIMLAR VE BELİRSİZLİKLER

- Bu rapor statik analizdir; GPU utilization/epoch süresi gibi metrikler runtime benchmark ile ayrıca doğrulanmalıdır.
- A100 thread ayarları iyileşti, ancak en iyi değerler veri tipi ve storage hızına göre değişebilir.
- `requirements.txt` aralıkları daraltılmış olsa da tam pinleme olmadığı için uzun dönem drift riski tamamen kapanmamıştır.
- `checkpoint_guard` yarış riskini lock ile azaltır; yine de sync modeli için tek kalıcı worker/thread-pool yaklaşımı daha deterministik olabilir.
- `flipud=0.5` bazı UAV domainlerinde fiziksel gerçekçiliği zayıflatabilir; sınıf bazlı recall ile doğrulama önerilir.

---

## 7. HAREKET PLANI (Neyi, Neyiyle Optimize Edeceğiz)

### Faz-1 (0-1 Gün): Reproducibility Kilitleme

- Hedef:
  - Sürüm drift riskini minimize etmek
- Neyi neyle optimize:
  - Kritik paketleri tam pin (`==`) veya lock dosyası ile sabitleme
  - Kurulum çıktısına sürüm snapshot ekleme
- Çıkış kriteri:
  - Aynı ortamda tekrar kurulumda paket sürümleri birebir aynı

### Faz-2 (1-3 Gün): I/O + Thread Tuning Finalizasyonu

- Hedef:
  - A100 üzerinde GPU idle süresini azaltmak
- Neyi neyle optimize:
  - `scripts/colab_bootstrap.py` thread parametrelerini benchmark ile ayarlama
  - DataLoader worker/prefetch ayarlarını veri setine göre ince tuning
- Çıkış kriteri:
  - Ortalama GPU utilization artışı
  - Epoch süresi düşüşü ve stabil batch-time dağılımı

### Faz-3 (3-7 Gün): MLOps Hardening + Veri Domain Kalitesi

- Hedef:
  - Senkronizasyon dayanıklılığı ve domain uyumunu artırmak
- Neyi neyle optimize:
  - Sync mekanizmasında tek worker/thread-pool modelini değerlendirme
  - `flipud` gibi domain-riskli augmentation değerlerini sınıf bazlı metriklerle doğrulama
- Çıkış kriteri:
  - Sync çakışması gözlemi olmadan uzun koşu
  - Domain metriklerinde (özellikle recall) tutarlı iyileşme

---

## 8. KPI CHECKLIST (Uygulama Sonrası Doğrulama)

- A100 BF16 gerçekten aktif mi (`amp_dtype=bf16` ve runtime log doğrulandı)?
- TF32 hem matmul hem cudnn tarafında açık mı?
- Ortalama GPU utilization `%90+` oldu mu?
- DataLoader worker sayısı A100 tier için en az 8 mi?
- Validation BF16 autocast içinde mi?
- NaN loss görülme sıklığı sıfıra indi mi?
- Checkpoint dosyaları atomik ve tutarlı yazılıyor mu?
- Resume akışı bozuk checkpoint senaryosunda güvenli fallback yapıyor mu?
- Aynı seed ile tekrar koşularda metrik oynaklığı kabul edilebilir aralıkta mı?
- Deney çıktıları (config + metrik + model) tekil klasörde çakışmasız saklanıyor mu?

