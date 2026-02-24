# ML Eğitim Hattı — Statik Kod Denetim Raporu

**Tarih:** 2026-02-24  
**Denetçi Rolü:** Kıdemli MLOps Mühendisi & Derin Öğrenme Kod Denetçisi  
**Kapsam:** UAV YOLO11m Nesne Tespiti + GPS Siamese Tracker Görsel Odometri  
**Yöntem:** Tamamen statik inceleme ve mantıksal çıkarım (kod çalıştırılmamıştır)

---

## 1. Özet Bulgular

Bu rapor, iki ayrı ML eğitim hattını (UAV nesne tespiti ve GPS görsel odometri) kapsayan ~3.500+ satır Python kodunun statik denetimini içermektedir.

| Kategori | Kritik | Yüksek | Orta | Düşük |
|---|---|---|---|---|
| Mantık ve Veri Güvenliği | 2 | 3 | 2 | 1 |
| Performans ve Kaynak Kullanımı | 1 | 3 | 3 | 1 |
| Eğitim Dinamikleri | 1 | 2 | 3 | 1 |
| Stabilite ve MLOps | 0 | 3 | 3 | 2 |
| Ortam ve Session Riskleri | 1 | 2 | 2 | 1 |
| **Toplam** | **5** | **13** | **13** | **6** |

**Genel Değerlendirme:** Kod tabanı, üretim seviyesine yakın bir olgunlukta olup OOM recovery (UAV), atomik checkpoint yazımı, çoklu GPU tier desteği ve Colab-local SSD optimizasyonu gibi güçlü mühendislik pratiklerini barındırmaktadır. Ancak GPS eğitim hattında yapısal eksiklikler (early stopping, OOM recovery, scheduler konfigürasyonu) bulunmaktadır. Bağımlılık yönetimi ve debug kod artıkları acil müdahale gerektirmektedir.

---

## 2. Kritik Riskler

### KR-01: `collate_drop_none` ile `__getitem__` Tutarsızlığı (GPS)

**Dosya:** `gps_training/train.py` (satır 161-167) ve `gps_training/dataset.py` (satır 280-287)

`collate_drop_none` fonksiyonu, batch içindeki `None` değerleri filtreleyerek bozuk örneklerin eğitimi çökertmesini engellemeye tasarlanmıştır. Ancak `GPSDataset.__getitem__` metodu hiçbir zaman `None` döndürmemektedir; hata durumlarında `RuntimeError` fırlatmaktadır. Bu tutarsızlık, `collate_drop_none`'ın None-filtreleme mantığını tamamen ölü kod haline getirmektedir.

**Risk:** Tek bir bozuk video frame'i veya eksik dosya tüm eğitim sürecini çökertebilir. Collate fonksiyonunun sağladığı dayanıklılık katmanı fiilen devre dışıdır.

**Etki:** Kritik — Eğitim sırasında veri bütünlüğü hatası durumunda kurtarma mekanizması çalışmaz.

---

### KR-02: OneCycleLR Scheduler Resume Uyumsuzluğu (GPS)

**Dosya:** `gps_training/train.py` (satır 229-253)

Resume senaryosunda scheduler `remaining_epochs` parametresi ile yeni bir `OneCycleLR` oluşturulmakta, ardından eski checkpoint'taki `scheduler_state_dict` yüklenmeye çalışılmaktadır. OneCycleLR'ın total_steps'i oluşturulma anında sabitlediği için, farklı `remaining_epochs` değeri ile oluşturulan yeni scheduler'a eski state'in yüklenmesi şu sonuçlara yol açabilir:

- LR eğrisinin kayması veya aniden sıfıra düşmesi
- `total_steps` uyumsuzluğundan dolayı scheduler'ın patlaması (step > total_steps)

Hata durumunda fallback mesajı yazdırılmakta ancak scheduler yeniden oluşturulmamakta, mevcut (hatalı) scheduler ile devam edilmektedir.

**Etki:** Kritik — Resume sonrası eğitim dinamikleri öngörülemez hale gelebilir, sessiz performans degradasyonuna yol açar.

---

### KR-03: `max_lr` Konfigürasyon Eksikliği (GPS)

**Dosya:** `gps_training/config.py` (satır 85-96) ve `gps_training/train.py` (satır 99-107)

`TRAIN_CONFIG` sözlüğünde `max_lr` anahtarı tanımlanmamıştır. `resolve_scheduler_max_lr` fonksiyonu bu durumda `learning_rate` değerine (1e-4) fallback yapmaktadır. Sonuç olarak `OneCycleLR`'ın `max_lr` ve `base_lr` değerleri eşit olmaktadır. Bu, OneCycleLR'ın temel özelliği olan "düşük LR → yüksek LR → düşük LR" döngüsünü tamamen etkisiz kılmaktadır; scheduler fiilen sabit LR gibi davranmaktadır.

**Etki:** Kritik — OneCycleLR'ın tüm avantajları (super-convergence, otomatik warmup) kaybedilmektedir.

---

### KR-04: `dataset.yaml` Oluşturmada `nc` Anahtarı Eksik, Standart Dışı `values` Anahtarı

**Dosya:** `uav_training/build_dataset.py` (satır 457-468)

Oluşturulan `dataset.yaml` dosyasında YOLO standardı olan `nc` (number of classes) anahtarı bulunmamakta, yerine standart dışı `values` anahtarı kullanılmaktadır. Ultralytics kütüphanesi `nc` değerini `names` sözlüğünden çıkarım yapabilmektedir; ancak bu davranış Ultralytics versiyonuna bağlıdır ve garanti değildir. `values` anahtarı Ultralytics tarafından tanınmaz ve sessizce görmezden gelinir.

**Etki:** Yüksek — Ultralytics major versiyon güncellemesinde sınıf sayısı yanlış çıkarılabilir, eğitim sessizce hatalı çalışabilir.

---

### KR-05: Bağımlılık Versiyonlarında Üst Sınır Eksikliği

**Dosya:** `requirements.txt`

Kritik bağımlılıkların hiçbirinde üst sınır belirtilmemiştir:

- `ultralytics>=8.2.0` — Ultralytics API'si sık değişmektedir (train argümanları, callback yapısı)
- `torch>=2.0.0` — PyTorch 3.x breaking change'leri olası
- `torchvision>=0.15.0` — ResNet18 weights API değişikliği olası
- `numpy>=1.24.0` — NumPy 2.0 birçok deprecated API'yi kaldırmıştır

Ayrıca `tqdm`, `pyyaml`, `opencv-python-headless`, `pandas`, `matplotlib`, `psutil` ve `pytest` paketlerinde hiçbir versiyon kısıtlaması yoktur.

**Etki:** Kritik — `pip install` ile ortam oluşturulduğunda gelecek versiyonlar mevcut kodu kırabilir. Tekrarlanabilirlik ciddi risk altındadır.

---

## 3. Performans Değerlendirmesi

### PD-01: GPS Frame Cache Bellek Çarpanı

**Dosya:** `gps_training/dataset.py` (satır 29-30, 120-125)

Her `GPSDataset` nesnesi 256 frame kapasiteli bir LRU cache tutmaktadır. DataLoader `num_workers=8` ile çalıştığında, her worker kendi sürecinde ayrı bir dataset kopyası oluşturur. Bu durumda bellekte 8 × 256 = 2.048 frame önbelleklenmektedir.

256×256 RGB frame başına ~192 KB olmak üzere, toplam cache bellek kullanımı ~393 MB civarındadır. Daha büyük frame boyutları veya yüksek worker sayıları ile bu değer hızla artabilir.

**Risk Seviyesi:** Yüksek — RAM kısıtlı ortamlarda (T4 Colab, 12 GB RAM) OOM tetikleyebilir.

---

### PD-02: GPS Eğitiminde OOM Recovery Mekanizması Yok

**Dosya:** `gps_training/train.py` (satır 275-378)

UAV eğitim hattı 4 aşamalı bir OOM recovery mekanizmasına sahiptir (compile kapatma → batch düşürme → imgsz düşürme → batch daha da düşürme). GPS eğitim hattında böyle bir mekanizma bulunmamaktadır. CUDA OOM hatası direkt olarak eğitimi sonlandırır.

**Risk Seviyesi:** Yüksek — Büyük batch (H100'de 256) ile eğitim sırasında video decoder + model + gradyan bellek kullanımı öngörülemeyen spike'lar yapabilir.

---

### PD-03: Validation DataLoader Eksik Optimizasyonlar (GPS)

**Dosya:** `gps_training/train.py` (satır 183-190)

`val_loader` oluşturulurken `persistent_workers` parametresi ayarlanmamıştır (varsayılan `False`). Bu, her epoch'ta validation aşamasında worker süreçlerinin yeniden oluşturulmasına neden olur. 100 epoch için bu, 100 kez worker fork/spawn overhead'i demektir.

Ayrıca `prefetch_factor` da `val_loader`'da ayarlanmamıştır (`train_loader`'da 4 olarak ayarlıdır).

**Risk Seviyesi:** Orta — Her epoch'ta validasyon başlangıcında 2-5 saniyelik gecikme, 100 epoch'ta toplam 3-8 dakika kayıp.

---

### PD-04: GPU'da Gereksiz Normalizasyon İşlemi (GPS)

**Dosya:** `gps_training/train.py` (satır 281-282)

Görüntü normalizasyonu (`/ 255.0`) dataset tarafında (CPU'da, `__getitem__` içinde) değil, eğitim döngüsü içinde GPU'da yapılmaktadır. Bu, her batch için GPU'nun trivial bir bölme işlemi yapmasına neden olur. Dataset tarafında `uint8` olarak taşıyıp GPU'da normalize etmek PCI-e bant genişliğini optimize etmek için bilinçli bir tercih olabilir; ancak bu durumda `pin_memory=True` ile birlikte `uint8` tensor'ların GPU'ya non-blocking transfer edilmesi beklenir — ki bu zaten yapılmaktadır.

**Risk Seviyesi:** Düşük — Performans etkisi minimal, ancak mimari tutarsızlık mevcut.

---

### PD-05: ONNX Export Eski Opset Versiyonu

**Dosya:** `gps_training/train.py` (satır 392)

ONNX export `opset_version=11` ile yapılmaktadır. Opset 11, 2019 standardıdır. Modern ONNX Runtime optimizasyonları (operator fusion, attention pattern recognition) opset 17+ gerektirmektedir.

**Risk Seviyesi:** Orta — Inference performansı optimal olmayabilir, bazı modern operatörler desteklenmeyebilir.

---

### PD-06: Her Epoch Checkpoint + Drive Sync I/O Yükü (UAV)

**Dosya:** `uav_training/config.py` (satır 245) ve `uav_training/train.py` (satır 229-253)

`save_period=1` ayarı her epoch'ta checkpoint kaydı tetiklemektedir. YOLO checkpoint'ları ~80-100 MB boyutundadır. 65 epoch'luk eğitim boyunca:

- 65 × ~90 MB = ~5.8 GB disk yazımı (yerel)
- Her epoch'ta Drive sync thread'i başlatılır (Colab'da)
- Google Drive FUSE üzerinden yazım işlemi GPU pipeline'ını dolaylı yoldan bloklamasa da disk I/O bant genişliğini tüketir

Drive sync non-blocking thread ile yapılmaktadır (iyi tasarım), ancak sıklığı yüksektir.

**Risk Seviyesi:** Orta — Özellikle disk-kısıtlı ortamlarda I/O darboğazı oluşturabilir.

---

### PD-07: `torch.compile` Koşulsuz Uygulanması (GPS)

**Dosya:** `gps_training/train.py` (satır 220-225)

`torch.compile` Python < 3.12 koşuluyla koşulsuz olarak uygulanmaktadır. SiameseTracker oldukça küçük bir modeldir (ResNet-18 backbone + 3 katmanlı MLP). `torch.compile`'ın ilk çalışma overhead'i (30-120 saniye) küçük modellerde amortize edilemeyebilir. Ayrıca derleme hataları `except` ile yakalanıp sessizce geçilmektedir.

**Risk Seviyesi:** Orta — İlk epoch'ta beklenmedik uzun süre, debug zorluğu.

---

### PD-08: Debug Log Artıkları (UAV)

**Dosya:** `uav_training/train.py` (satır 338-346, 356-362, 503-511, 542-548)

Kodda 4 adet `#region agent log` bloğu bulunmaktadır. Bu bloklar her eğitim denemesinde `debug-4e729f.log` dosyasına JSON satırları yazmaktadır. Her blok `json.dumps` + dosya açma/yazma/kapatma işlemi gerçekleştirmektedir.

Bu debug blokları üretim kodunda bırakılmıştır ve:

- Her eğitim denemesinde gereksiz I/O üretmektedir
- `_logf = _pl.Path("debug-4e729f.log")` göreceli yol kullanması, çalışma dizinine bağlı olarak farklı yerlere yazılmasına neden olabilir
- Kod okunabilirliğini düşürmektedir

**Risk Seviyesi:** Yüksek — Üretim kodu kirliliği, I/O overhead, bakım zorluğu.

---

## 4. Eğitim Stabilitesi Analizi

### ES-01: GPS Eğitiminde Early Stopping Mekanizması Yok

**Dosya:** `gps_training/train.py` (satır 275-378)

UAV eğitim hattı `patience=30` ile early stopping kullanmaktadır (Ultralytics yerleşik). GPS eğitim hattında ise hiçbir early stopping mekanizması bulunmamaktadır. Eğitim, konfigürasyonda belirtilen 100 epoch boyunca koşulsuz olarak devam eder.

`best_val_loss` takip edilmekte ve en iyi model kaydedilmektedir; ancak bu bilgi eğitimi durdurmak için kullanılmamaktadır.

**Risk:** Overfitting — Validation loss yükselmeye başladıktan sonra onlarca epoch daha gereksiz eğitime devam edilebilir, GPU kaynağı israf edilir.

---

### ES-02: GPS Model Regressor Başlığında Normalizasyon Katmanı Eksik

**Dosya:** `gps_training/model.py` (satır 16-23)

Regressor başlığı `Linear → ReLU → Dropout → Linear → ReLU → Linear` yapısındadır. Araya hiçbir BatchNorm veya LayerNorm katmanı eklenmemiştir.

ResNet-18 backbone'u kendi BatchNorm katmanlarına sahip olduğundan, backbone çıktıları normalize edilmiştir. Ancak iki backbone çıktısının concatenation'ı sonrası (1024 boyutlu vektör) regressor girişinde dağılım kayması olabilir. Bu, öğrenme oranı hassasiyetini artırır ve gradyan stabilitesini düşürür.

**Risk:** Orta — Eğitim başlangıcında yavaş yakınsama veya LR hassasiyeti.

---

### ES-03: UAV `torch.cuda.manual_seed()` CUDA Kontrolü Olmadan

**Dosya:** `uav_training/train.py` (satır 431-432)

`setup_seed` fonksiyonunda `torch.cuda.manual_seed(seed)` ve `torch.cuda.manual_seed_all(seed)` çağrıları CUDA kullanılabilirlik kontrolü olmadan yapılmaktadır. GPU olmayan bir ortamda (CPU-only) bu çağrılar hata fırlatmaz (PyTorch bunu sessizce yönetir), ancak GPS modülündeki aynı fonksiyon `if torch.cuda.is_available()` kontrolü yapmaktadır. İki modül arasındaki bu tutarsızlık, kod bakımını zorlaştırmaktadır.

**Risk:** Düşük — Fonksiyonel hata yok, ancak kod tutarsızlığı mevcut.

---

### ES-04: İki Fazlı Eğitimde Optimizer Sürekliliği Kesilmesi (UAV)

**Dosya:** `uav_training/train.py` (satır 630-638)

İki fazlı eğitimde faz 2, faz 1'in `best.pt` ağırlıklarıyla `resume=False` olarak başlatılmaktadır. Bu, optimizer momentum'unun, scheduler state'inin ve tüm eğitim geçmişinin sıfırlanması anlamına gelir. Fine-tuning bağlamında bu kabul edilebilir bir stratejidir; ancak faz 2'nin 15 epoch'luk kısa süresi göz önüne alındığında, yeni bir warmup (5 epoch) bu sürenin 1/3'ünü tüketecektir.

Faz 2 batch boyutu da `max(1, batch // 2)` olarak yarıya indirilmektedir ki bu daha yüksek çözünürlük (896px) ile tutarlıdır.

**Risk:** Orta — Faz 2 warmup oranı toplam süreye göre orantısız yüksek olabilir.

---

### ES-05: Gradient Clipping Stratejisi Farklılığı

GPS eğitim hattında `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` ile açık gradyan kırpma uygulanmaktadır. UAV eğitim hattı ise Ultralytics'in dahili gradyan yönetimini kullanmaktadır.

GPS tarafında bu önlem mevcuttur ve non-finite loss kontrolü ile birlikte iyi bir savunma katmanı oluşturmaktadır. Ancak `max_norm=1.0` değerinin SiameseTracker mimarisi için uygunluğu deneysel olarak doğrulanmalıdır.

**Risk:** Düşük — Gradyan kırpma mevcut, uygunluğu koşullu.

---

### ES-06: Augmentation Split Uygulaması

GPS eğitim hattında augmentation yalnızca `self.split == "train"` koşuluyla uygulanmaktadır — bu doğrudur. Horizontal flip uygulandığında `delta[0]` (translation_x) doğru şekilde negatifine çevrilmektedir — bu, Siamese mimarisi için kritik ve doğru bir uygulamadır.

UAV eğitim hattında augmentation Ultralytics tarafından yönetilmektedir ve yalnızca eğitim split'ine uygulanmaktadır.

Her iki hatta da augmentation doğru split'te uygulanmaktadır. Bu alanda risk tespit edilmemiştir.

---

### ES-07: Loss Fonksiyonu Güvenliği

GPS eğitiminde `MSELoss` kullanılmakta ve her batch sonrası `torch.isfinite(loss)` kontrolü yapılarak non-finite loss durumunda `RuntimeError` fırlatılmaktadır. Bu iyi bir pratiktir.

Ancak NaN/Inf kontrolü yalnızca loss değerinde yapılmaktadır. Model çıktısının (output) NaN olup olmadığı kontrol edilmemektedir. AMP (mixed precision) ile özellikle FP16 kullanıldığında, model çıktısında NaN oluşabilir ve loss hesabına kadar fark edilmeyebilir.

UAV eğitim hattında loss güvenliği tamamen Ultralytics'e devredilmiştir.

**Risk:** Orta — NaN propagation model çıktısından loss'a kadar gecikebilir.

---

## 5. MLOps Olgunluk Değerlendirmesi

### MO-01: Experiment Tracking Eksikliği

Her iki eğitim hattında da W&B, MLflow, ClearML veya benzeri bir deney takip sistemi entegre edilmemiştir. UAV tarafında Ultralytics'in yerleşik TensorBoard ve CSV loglaması mevcuttur, GPS tarafında ise yalnızca konsol çıktısı ve `loss_plot.png` grafiği bulunmaktadır.

Hiperparametre, metrik ve artifact'ların sistematik kaydı yapılmamaktadır. Deneyler arası karşılaştırma yalnızca manuel dosya incelemesi ile mümkündür.

**Olgunluk:** Düşük — Tekrarlanabilir deney yönetimi için yetersiz.

---

### MO-02: `torch.load(weights_only=False)` Güvenlik Riski

**Dosyalar:** `uav_training/train.py` (satır 64), `gps_training/train.py` (satır 91)

Her iki modülde de checkpoint yükleme `weights_only=False` parametresi ile yapılmaktadır. PyTorch 2.6+'da varsayılan `weights_only=True` olarak değiştirilmiştir. `weights_only=False` kullanımı, checkpoint dosyası içine gömülmüş rastgele Python kodunun `pickle.load` ile çalıştırılmasına olanak tanır.

YOLO checkpoint'larının `DetectionModel` gibi özel sınıflar içermesi nedeniyle `weights_only=True` kullanımı kısıtlıdır; ancak GPS checkpoint'ları yalnızca standart `state_dict` içerdiğinden güvenli moda geçirilebilir.

**Olgunluk:** Orta — Bilinen bir güvenlik riski bilinçli olarak kabul edilmiş durumda.

---

### MO-03: Checkpoint Kayıt Güvenliği

Her iki modülde de atomik yazım deseni (tmp dosya → `os.replace`) kullanılmaktadır. GPS modülünde ek olarak `last_model.bak` backup rotasyonu bulunmaktadır. UAV modülünde Ultralytics'in kendi checkpoint mekanizması kullanılmaktadır.

`_is_checkpoint_valid` fonksiyonları her iki modülde de mevcuttur ve dosya boyutu + `torch.load` ile bütünlük kontrolü yapmaktadır.

**Olgunluk:** Yüksek — Checkpoint kaybı riskine karşı güçlü savunma katmanları mevcuttur.

---

### MO-04: Resume Dayanıklılığı

UAV eğitim hattı 3 kaynaklı resume stratejisi uygulamaktadır (CLI → yerel dizin → Google Drive). GPS eğitim hattı 2 kaynaklı strateji kullanmaktadır (son checkpoint → backup). Her iki durumda da corrupt checkpoint tespiti yapılmakta ve alternatif kaynaklara geçilmektedir.

Ancak KR-02'de belirtildiği gibi, GPS resume'unda scheduler uyumsuzluğu mevcuttur.

**Olgunluk:** Orta — Resume mekanizması robust, ancak scheduler uyumsuzluğu var.

---

### MO-05: Versiyon Yönetimi ve Senkronizasyon

`__version__` değişkeni `uav_training/__init__.py` ve `uav_training/train.py` dosyalarında tutulmaktadır. Her iki dosyada da değer `"0.8.12"` olarak senkrondur. CHANGELOGS.md ile versiyon takibi yapılmaktadır.

GPS modülünün `__init__.py` dosyasında versiyon bilgisi bulunmamaktadır.

**Olgunluk:** Orta — UAV modülü için iyi, GPS modülü için eksik.

---

### MO-06: Hyperparameter Loglama

Her iki modülde de `print_training_config` fonksiyonu ile tüm hiperparametreler eğitim başlangıcında konsola yazdırılmaktadır. UAV modülünde ek olarak `args.yaml` dosyası Ultralytics tarafından otomatik olarak kaydedilmektedir.

GPS modülünde hiperparametreler yalnızca konsol çıktısıdır; dosyaya kalıcı olarak kaydedilmemektedir.

**Olgunluk:** Orta — UAV iyi, GPS'te kalıcı kayıt eksik.

---

### MO-07: `rsync` Komutu Shell Injection Riski

**Dosya:** `gps_training/train.py` (satır 125-127)

`subprocess.run(_sync_cmd, shell=True)` kullanımı, dosya yollarında özel karakter bulunması durumunda shell injection riski taşımaktadır. Yollar kullanıcı girdisi değil sabit konfigürasyondan gelmektedir, ancak `shell=True` kullanımı güvenlik best practice'lerine aykırıdır.

**Olgunluk:** Düşük — Güvenlik pratikleri açısından iyileştirilmelidir.

---

### MO-08: CI/CD Pipeline Değerlendirmesi

**Dosya:** `.github/workflows/lint.yml`

Mevcut CI pipeline'ı:

- Sözdizimi odaklı flake8 (E9, F63, F7, F82)
- `python -m compileall` ile derleme kontrolü
- `pytest tests/` ile test çalıştırma

Eksikler:

- Tip kontrolü (mypy/pyright) yok
- Güvenlik taraması (bandit/safety) yok
- Bağımlılık güvenlik kontrolü yok
- Kod kapsama (coverage) ölçümü yok

**Olgunluk:** Düşük-Orta — Temel sözdizimi kontrolü mevcut, derinlemesine analiz eksik.

---

## 6. Belirsizlikler ve Koşullu Riskler

### BR-01: Megaset Scene-Aware Splitting Deterministik Olmayabilir

**Dosya:** `uav_training/build_dataset.py` (satır 407-418)

`random.shuffle(scenes)` çağrısı, dosyanın başında `set_seed(42)` ile sabitlenmiş global RNG'ye bağımlıdır. Eğer `build_dataset()` fonksiyonu farklı bir import sırasıyla veya farklı bir modülden çağrılırsa, RNG state'i farklı olabilir ve train/val split'leri değişebilir. Bu durumun gerçekleşip gerçekleşmediği, fonksiyonun çağrılma bağlamına bağlıdır.

**Koşul:** Farklı çağrı sıraları veya modül import'ları durumunda.

---

### BR-02: GPS Video Decoder State Tutarsızlığı

**Dosya:** `gps_training/dataset.py` (satır 159-186)

Ardışık frame okuma optimizasyonu (`_read_video_pair`) video decoder'ın iç state'ine bağımlıdır. Eğer farklı worker'lar aynı videoyu eş zamanlı açarsa (multi-worker DataLoader), her worker kendi `_video_caps` sözlüğünü tuttuğundan doğrudan çakışma olmaz. Ancak OS seviyesinde aynı video dosyasına çoklu eşzamanlı erişim disk I/O contention'a neden olabilir.

**Koşul:** Yüksek worker sayısı + büyük video dosyaları + yavaş disk (Drive FUSE) durumunda.

---

### BR-03: UAV İki Fazlı Eğitimde Faz 1 Checkpoint Bulunamazsa

**Dosya:** `uav_training/train.py` (satır 609-618)

Faz 2'nin başlaması için faz 1'in `best.pt` dosyasına ihtiyaç vardır. Eğer faz 1 erken sonlandırılırsa (OOM, Colab timeout, Drive bağlantı kaybı), `best.pt` oluşturulmamış olabilir. Kod `last.pt`'ye fallback yapmaktadır; ancak o da yoksa `FileNotFoundError` fırlatılır ve tüm eğitim sürecini çökerterek faz 1'in kısmi sonuçlarını da tehlikeye atar.

**Koşul:** Colab session timeout'u veya erken GPU kaybı durumunda.

---

### BR-04: Ultralytics Versiyon Bağımlılığı

Kod tabanı `ultralytics>=8.2.0` ile kısıtlanmıştır ancak üst sınır yoktur. Ultralytics kütüphanesi aktif geliştirme altındadır ve şu API'lere doğrudan bağımlılık mevcuttur:

- `YOLO(model_path).train(**kwargs)` argüman yapısı
- `model.add_callback("on_fit_epoch_end", ...)` callback sistemi
- `results.box.map50`, `results.box.map` metrik erişimi
- `results.csv` dosya formatı ve sütun isimleri

Bu API'lerin herhangi birindeki değişiklik, eğitim hattını sessizce kırabilir.

**Koşul:** Ultralytics major veya minor versiyon güncellemesi durumunda.

---

### BR-05: `atexit` ile Video Capture Temizliği Multi-Worker Durumunda

**Dosya:** `gps_training/dataset.py` (satır 31, 289-296)

`atexit.register(self._close_video_caps)` main process'te kayıt edilmektedir. Ancak DataLoader worker süreçleri fork/spawn ile oluşturulduğunda, `atexit` handler'ları worker süreçlerinde çağrılmayabilir (özellikle signal ile öldürülen worker'larda). Bu, OpenCV `VideoCapture` nesnelerinin temizlenmeden kalmasına yol açabilir.

**Koşul:** Worker crash veya forced kill durumunda.

---

### BR-06: GPS Train/Val Split'inde Test Seti Eksik

**Dosya:** `gps_training/dataset.py` (satır 61-68)

GPS dataset'i yalnızca train (%90) ve val (%10) olarak bölünmektedir. Test seti bulunmamaktadır. Model performansının nihai değerlendirmesi validation seti üzerinden yapılmaktadır, bu da model seçim yanlılığı (selection bias) riski taşımaktadır.

**Koşul:** Model performansının gerçek dünya genellemesini değerlendirirken.

---

## 7. Genel Sağlık Skoru

| Değerlendirme Alanı | Puan (0-10) | Ağırlık | Ağırlıklı Puan |
|---|---|---|---|
| Mantık ve Veri Güvenliği | 7.0 | %20 | 1.40 |
| Performans ve Kaynak Kullanımı | 7.5 | %20 | 1.50 |
| Eğitim Dinamikleri | 6.5 | %20 | 1.30 |
| Stabilite ve Checkpoint Güvenliği | 8.0 | %15 | 1.20 |
| MLOps Olgunluğu | 5.5 | %15 | 0.825 |
| Ortam ve Bağımlılık Yönetimi | 5.0 | %10 | 0.50 |
| **Genel Sağlık Skoru** | | | **6.7 / 10** |

### Puan Gerekçesi

**Güçlü Yönler:**

- Atomik checkpoint yazımı ve backup rotasyonu ile veri kaybı riski minimumda
- Çoklu GPU tier desteği ve explicit batch sizing ile kaynak kullanımı optimize
- Colab-local SSD mimarisi ile I/O darboğazı etkin biçimde çözülmüş
- UAV eğitiminde 4 aşamalı OOM recovery mekanizması
- İki fazlı eğitim stratejisi (50+15 epoch) ile fine-tuning desteği
- Augmentation'ın doğru split'te uygulanması ve Siamese-aware delta düzeltmeleri
- Mixed precision stratejisi GPU kapabilitesine göre otomatik seçim (BF16 vs FP16)
- Bbox validasyonu (NaN, range, minimum boyut) ile veri kalitesi kontrolü

**Zayıf Yönler:**

- GPS eğitim hattındaki yapısal eksiklikler (early stopping, OOM recovery, OneCycleLR konfigürasyonu)
- Bağımlılık versiyonlarının sabitlenmemesi, tekrarlanabilirliği tehdit ediyor
- Deney takip sistemi eksikliği, sistematik model karşılaştırmasını engelliyor
- Debug kod artıkları üretim kodunda bırakılmış
- GPS modülünde kalıcı hiperparametre kaydı yapılmıyor
- Test seti eksikliği (GPS) model değerlendirmesini zayıflatıyor

**Sonuç:** Kod tabanı, UAV eğitim hattı açısından üretim seviyesine yakın bir olgunlukta olup güçlü hata kurtarma ve optimizasyon mekanizmaları barındırmaktadır. GPS eğitim hattı ise UAV hattına kıyasla daha erken bir aşamadadır ve yapısal iyileştirmeler gerektirmektedir. Bağımlılık yönetimi ve MLOps pratikleri her iki hat için de güçlendirilmelidir.
