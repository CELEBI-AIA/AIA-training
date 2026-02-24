# ML Eğitim Hattı — Statik Kod Denetim Raporu

**Tarih:** 2026-02-24  
**Denetçi Rolü:** Kıdemli MLOps Mühendisi & Derin Öğrenme Kod Denetçisi  
**Kapsam:** UAV YOLO11m Nesne Tespiti + GPS Siamese Tracker Görsel Odometri  
**Yöntem:** Tamamen statik inceleme ve mantıksal çıkarım (kod çalıştırılmamıştır)

---

## 1. Özet Bulgular

Bu rapor, iki ayrı ML eğitim hattını (UAV nesne tespiti ve GPS görsel odometri) kapsayan Python kodunun güncel statik denetimini içermektedir. Önceki denetimden bu yana CHANGELOGS kayıtlarına göre birçok iyileştirme uygulanmıştır.

| Kategori | Kritik | Yüksek | Orta | Düşük |
|---|---|---|---|---|
| Mantık ve Veri Güvenliği | 0 | 1 | 2 | 1 |
| Performans ve Kaynak Kullanımı | 0 | 1 | 2 | 1 |
| Eğitim Dinamikleri | 0 | 1 | 2 | 0 |
| Stabilite ve MLOps | 0 | 2 | 3 | 1 |
| Ortam ve Session Riskleri | 0 | 0 | 2 | 1 |
| **Toplam** | **0** | **5** | **11** | **4** |

**Genel Değerlendirme:** Kod tabanı, önceki denetimde tespit edilen kritik risklerin büyük kısmını gidermiş durumdadır. OOM recovery (UAV ve GPS), early stopping (GPS), atomik checkpoint yazımı, çoklu GPU tier desteği, Colab-local SSD optimizasyonu ve bağımlılık üst sınırları mevcuttur. Kalan riskler ağırlıklı olarak orta ve yüksek seviyede olup, deney takibi, konfigürasyon tutarlılığı ve koşullu ortam riskleri odaklıdır.

---

## 2. Kritik Riskler

Önceki denetimde tespit edilen kritik risklerin tamamı giderilmiş veya tasarım kararı olarak belgelenmiştir:

- **KR-01 (collate_drop_none):** `GPSDataset.__getitem__` artık hata durumunda `None` döndürüyor; `collate_drop_none` dayanıklılık katmanı etkin.
- **KR-02 (OneCycleLR resume):** Scheduler kasıtlı olarak yeniden oluşturuluyor; `total_steps` uyumsuzluğu tasarım gereği önleniyor.
- **KR-03 (max_lr):** `gps_training/config.py` içinde `max_lr: 1e-3` tanımlı.
- **KR-04 (dataset.yaml nc):** `build_dataset.py` çıktısında `nc: 4` mevcut.
- **KR-05 (bağımlılık üst sınırları):** `requirements.txt` içinde `ultralytics<9`, `torch<3`, `torchvision<1`, `numpy<3` üst sınırları tanımlı.

**Güncel kritik risk tespit edilmemiştir.**

---

## 3. Performans Değerlendirmesi

### PD-01: GPS Frame Cache Bellek Kullanımı

**Dosya:** `gps_training/dataset.py` (satır 29-30, 120-125)

`frame_cache_size` varsayılan 128 olarak düşürülmüştür. `num_workers=8` ile 8 × 128 = 1.024 frame önbelleklenir. 256×256 RGB frame başına ~192 KB ile toplam ~197 MB civarındadır.

**Risk Seviyesi:** Orta — RAM kısıtlı ortamlarda izlenmeli.

---

### PD-02: Her Epoch Checkpoint + Drive Sync I/O Yükü (UAV)

**Dosya:** `uav_training/config.py` (satır 245) ve `uav_training/train.py` (checkpoint_guard)

`save_period=1` her epoch'ta checkpoint kaydı ve Drive sync tetiklemektedir. YOLO checkpoint'ları ~80-100 MB boyutundadır. 65 epoch boyunca ~5.8 GB disk yazımı ve her epoch'ta non-blocking Drive sync thread'i çalışır.

**Risk Seviyesi:** Orta — Disk-kısıtlı ortamlarda I/O darboğazı oluşturabilir.

---

### PD-03: GPU'da Normalizasyon (GPS)

**Dosya:** `gps_training/train.py` (satır 302-304)

Görüntü normalizasyonu (`/ 255.0`) eğitim döngüsünde GPU'da yapılmaktadır. Dataset `uint8` döndürüp `pin_memory=True` ile transfer etmektedir; bu PCI-e bant genişliği açısından bilinçli bir tercih olabilir.

**Risk Seviyesi:** Düşük — Mimari tutarlı, performans etkisi minimal.

---

### PD-04: OOM Recovery Sınırı (GPS)

**Dosya:** `gps_training/train.py` (satır 332-334)

`max_oom_recoveries=2` ile en fazla 2 OOM sonrası batch yarıya indirilir. Üçüncü OOM'da eğitim sonlanır. H100 tier'da batch=256 ile video decoder + model spike'ları bu sınırı zorlayabilir.

**Risk Seviyesi:** Yüksek — Büyük batch ve uzun videolarda üçüncü OOM olasılığı.

---

## 4. Eğitim Stabilitesi Analizi

### ES-01: phase2_close_mosaic Konfigürasyon Tutarsızlığı (UAV)

**Dosya:** `uav_training/config.py` (satır 201, 294)

Varsayılan `TRAIN_CONFIG` içinde `phase2_close_mosaic: 10` tanımlı; `auto_detect_hardware()` Colab override'ında `phase2_close_mosaic: 5` kullanılıyor. Colab'da çalışan eğitimlerde mosaic augmentation daha erken kapanır; yerel ve Colab çalıştırmaları arasında davranış farkı oluşur.

**Risk:** Orta — İki fazlı eğitim tutarlılığı ve karşılaştırılabilirlik etkilenebilir.

---

### ES-02: İki Fazlı Eğitimde Optimizer Sürekliliği (UAV)

**Dosya:** `uav_training/train.py` (satır 436-447)

Faz 2, faz 1'in `best.pt` ağırlıklarıyla `resume=False` olarak başlatılır. Optimizer momentum ve scheduler state sıfırlanır. Faz 2'nin 15 epoch'luk süresinde warmup (5 epoch) toplam sürenin yaklaşık 1/3'ünü tüketir.

**Risk:** Orta — Fine-tuning bağlamında kabul edilebilir; warmup oranı faz 2 için yüksek olabilir.

---

### ES-03: Augmentation Split Uygulaması

GPS ve UAV hatlarında augmentation yalnızca train split'ine uygulanmaktadır. GPS tarafında horizontal flip uygulandığında `delta[0]` (translation_x) doğru şekilde negatifine çevrilmektedir.

**Risk:** Tespit edilmedi — Augmentation doğru split'te uygulanıyor.

---

### ES-04: Loss ve Model Çıktısı Güvenliği (GPS)

**Dosya:** `gps_training/train.py` (satır 315-319)

Hem model çıktısı (`torch.isfinite(output)`) hem de loss (`torch.isfinite(loss)`) kontrol edilmektedir. AMP altında NaN erken tespit edilir.

**Risk:** Düşük — Savunma katmanları yeterli.

---

## 5. MLOps Olgunluk Değerlendirmesi

### MO-01: Experiment Tracking Eksikliği

Her iki eğitim hattında da W&B, MLflow veya benzeri deney takip sistemi entegre değildir. UAV tarafında Ultralytics'in TensorBoard ve CSV loglaması, GPS tarafında `train_config.json` ve `loss_plot.png` mevcuttur. Hiperparametre ve metrik karşılaştırması manuel dosya incelemesine dayanır.

**Olgunluk:** Düşük — Sistematik deney yönetimi için yetersiz.

---

### MO-02: weights_only=False Güvenlik Riski

**Dosyalar:** `uav_training/train.py`, `gps_training/train.py`

Checkpoint yüklemelerinde `weights_only=False` kullanılmaktadır. PyTorch 2.6+ varsayılanı `weights_only=True`'dur. `weights_only=False` ile checkpoint içindeki rastgele Python kodu çalıştırılabilir. YOLO checkpoint'ları `DetectionModel` içerdiği için bu mod zorunludur; GPS checkpoint'ları yalnızca `state_dict` içerdiğinden teorik olarak `weights_only=True` kullanılabilir.

**Olgunluk:** Orta — Bilinen risk; güvenilmeyen checkpoint kaynaklarından yükleme yapılmamalı.

---

### MO-03: Checkpoint Kayıt Güvenliği

Her iki modülde atomik yazım (tmp dosya → `os.replace`) kullanılmaktadır. GPS'te `last_model.bak` backup rotasyonu, UAV'de Ultralytics checkpoint mekanizması mevcuttur. `_is_checkpoint_valid` fonksiyonları bütünlük kontrolü yapmaktadır.

**Olgunluk:** Yüksek — Checkpoint kaybı riskine karşı güçlü savunma.

---

### MO-04: Hyperparameter Loglama

UAV tarafında Ultralytics `args.yaml` otomatik kaydeder. GPS tarafında `train_config.json` eğitim başlangıcında kaydedilir; epoch/loss geçmişi bu dosyada tutulmaz.

**Olgunluk:** Orta — GPS'te deney karşılaştırması için epoch metrikleri eksik.

---

### MO-05: CI/CD Pipeline

**Dosya:** `.github/workflows/lint.yml`

Mevcut pipeline: flake8 (E9, F63, F7, F82), `compileall`, pytest. Tip kontrolü (mypy/pyright), güvenlik taraması (bandit/safety) ve kod kapsama ölçümü bulunmamaktadır.

**Olgunluk:** Orta — Temel sözdizimi ve test kontrolü mevcut.

---

## 6. Belirsizlikler ve Koşullu Riskler

### BR-01: Megaset Scene-Aware Splitting Determinism

**Dosya:** `uav_training/build_dataset.py` (satır 407-418)

`random.shuffle(scenes)` çağrısı dosya başındaki `set_seed(42)` ile sabitlenmiş RNG'ye bağlıdır. Farklı import sırası veya çağrı bağlamı RNG state'ini değiştirebilir; train/val split'leri değişebilir.

**Koşul:** Farklı modül import sıraları veya build_dataset çağrı bağlamları.

---

### BR-02: GPS Video Decoder ve Multi-Worker I/O

**Dosya:** `gps_training/dataset.py` (satır 159-186)

Her worker kendi `_video_caps` sözlüğünü tutar; doğrudan çakışma yoktur. Aynı video dosyasına çoklu eşzamanlı erişim disk I/O contention'a neden olabilir.

**Koşul:** Yüksek worker sayısı + büyük video dosyaları + yavaş disk (Drive FUSE).

---

### BR-03: UAV İki Fazlı Eğitimde Faz 1 Checkpoint Bulunamazsa

**Dosya:** `uav_training/train.py` (satır 436-442)

Faz 2, faz 1'in `best.pt` veya `last.pt` dosyasına ihtiyaç duyar. Her ikisi de yoksa `FileNotFoundError` fırlatılır ve tüm eğitim süreci sonlanır.

**Koşul:** Colab session timeout, erken GPU kaybı veya Drive bağlantı kesintisi.

---

### BR-04: atexit ile VideoCapture Temizliği (GPS)

**Dosya:** `gps_training/dataset.py` (satır 31, 289-296)

`atexit.register(self._close_video_caps)` main process'te kayıtlıdır. DataLoader worker süreçleri fork/spawn ile oluşturulduğunda, signal ile öldürülen worker'larda `atexit` handler'ları çağrılmayabilir. OpenCV `VideoCapture` nesneleri temizlenmeden kalabilir.

**Koşul:** Worker crash veya forced kill durumunda.

---

### BR-05: GPS Test Seti Eksikliği

**Dosya:** `gps_training/dataset.py` (satır 61-68)

GPS dataset'i yalnızca train (%90) ve val (%10) olarak bölünmektedir. Test seti yoktur. Model performansı validation seti üzerinden değerlendirilir; model seçim yanlılığı riski taşır.

**Koşul:** Gerçek dünya genellemesi değerlendirmesinde.

---

### BR-06: Ultralytics API Bağımlılığı

Kod `ultralytics>=8.2.0,<9.0.0` ile sınırlıdır. Ultralytics API'leri (train argümanları, callback sistemi, results erişimi) değişirse eğitim hattı sessizce kırılabilir.

**Koşul:** Ultralytics major veya minor versiyon güncellemesi.

---

## 7. Genel Sağlık Skoru

| Değerlendirme Alanı | Puan (0-10) | Ağırlık | Ağırlıklı Puan |
|---|---|---|---|
| Mantık ve Veri Güvenliği | 8.0 | %20 | 1.60 |
| Performans ve Kaynak Kullanımı | 7.5 | %20 | 1.50 |
| Eğitim Dinamikleri | 7.5 | %20 | 1.50 |
| Stabilite ve Checkpoint Güvenliği | 8.5 | %15 | 1.275 |
| MLOps Olgunluğu | 6.0 | %15 | 0.90 |
| Ortam ve Bağımlılık Yönetimi | 7.5 | %10 | 0.75 |
| **Genel Sağlık Skoru** | | | **7.5 / 10** |

### Puan Gerekçesi

**Güçlü Yönler:**

- Kritik risklerin tamamı giderilmiş veya tasarım kararı olarak belgelenmiş
- Atomik checkpoint yazımı ve backup rotasyonu ile veri kaybı riski düşük
- UAV ve GPS'te OOM recovery mekanizması mevcut
- GPS'te early stopping, NaN kontrolü ve config snapshot
- Bağımlılık üst sınırları ile tekrarlanabilirlik iyileştirilmiş
- Colab-local SSD mimarisi ile I/O darboğazı azaltılmış
- Çoklu GPU tier desteği ve explicit batch sizing
- Augmentation doğru split'te uygulanıyor; Siamese-aware delta düzeltmeleri mevcut

**Zayıf Yönler:**

- Deney takip sistemi (W&B, MLflow) eksik
- phase2_close_mosaic Colab/yerel tutarsızlığı
- GPS'te test seti yok; epoch metrikleri train_config.json'da tutulmuyor
- CI/CD'de tip kontrolü ve güvenlik taraması eksik
- weights_only=False güvenlik riski (özellikle GPS için iyileştirilebilir)

**Sonuç:** Kod tabanı önceki denetime göre belirgin şekilde olgunlaşmıştır. UAV eğitim hattı üretim seviyesine yakındır; GPS hattı da OOM recovery, early stopping ve config snapshot ile güçlendirilmiştir. Kalan iyileştirmeler deney yönetimi, konfigürasyon tutarlılığı ve koşullu ortam riskleri odaklıdır.
