# ML Eğitim Hattı — Statik Kod Denetim Raporu

**Tarih:** 2026-02-24  
**Kapsam:** UAV (YOLO11m) + GPS (ResNet-18 Siamese) eğitim modülleri  
**Yöntem:** Tüm kaynak dosyalar satır satır statik olarak incelendi; runtime çalıştırılmadı.

---

## 1. Özet Bulgular

| #  | Modül | Önem       | Kategori          | Kısa Açıklama                                                    | Dosya & Satır                          |
|----|-------|------------|-------------------|------------------------------------------------------------------|----------------------------------------|
| 1  | GPS   | **Kritik** | Platform          | `import fcntl` üst düzey — Windows'ta `ModuleNotFoundError`      | `gps_training/train.py:24`             |
| 2  | GPS   | **Kritik** | Eğitim            | `OneCycleLR` resume uyumsuzluğu — scheduler toplam adım hatası   | `gps_training/train.py:218-223`        |
| 3  | GPS   | **Kritik** | Eğitim            | `GradScaler` eksik — FP16 modda (pre-Ampere) gradyan taşması     | `gps_training/train.py:264-275`        |
| 4  | GPS   | **Kritik** | Veri Güvenliği    | `best_model.pt` atomik yazma yok — yarım dosya riski             | `gps_training/train.py:344`            |
| 5  | UAV   | **Yüksek** | Bellek/Performans | `_is_checkpoint_valid` her çağrıda tüm modeli CPU'ya yüklüyor    | `uav_training/train.py:52-63`          |
| 6  | UAV   | **Yüksek** | Veri Tutarlığı    | `TARGET_CLASSES` sınıf indeksleri MAPPINGS ile tutarsız           | `uav_training/config.py:33-44`         |
| 7  | GPS   | **Yüksek** | Performans        | Video frame seek `CAP_PROP_POS_FRAMES` — rastgele erişimde yavaş | `gps_training/dataset.py:138`          |
| 8  | GPS   | **Yüksek** | Eğitim            | Veri augmentasyonu yok (sadece resize) — overfitting riski        | `gps_training/dataset.py:230-231`      |
| 9  | UAV   | **Orta**   | Thread Safety     | Drive sync sırasında checkpoint dosyası eş zamanlı yazılabilir    | `uav_training/train.py:201-234`        |
| 10 | GPS   | **Orta**   | Eğitim            | `torch.compile` bare `except` ile — hata yutulur                 | `gps_training/train.py:211-214`        |
| 11 | Deps  | **Orta**   | MLOps             | `requirements.txt` üst sınır yok — major versiyon kırılma riski  | `requirements.txt`                     |
| 12 | CI    | **Orta**   | MLOps             | CI yalnız flake8 sözdizimi — unit test çalıştırmıyor             | `.github/workflows/lint.yml`           |
| 13 | GPS   | **Düşük**  | Veri              | Çok az sequence ile time-split (90/10) — genelleme riski          | `gps_training/dataset.py:60-66`        |
| 14 | UAV   | **Düşük**  | Performans        | Oversampling `shutil.copy2` fiziksel kopya (fallback durumunda)   | `uav_training/build_dataset.py:334-335`|

---

## 2. Kritik Riskler

### 2.1 GPS: `fcntl` Windows Uyumsuzluğu

**Dosya:** `gps_training/train.py:24`  
**Durum:** `import fcntl` satırı modül düzeyinde koşulsuz olarak çağrılıyor. `fcntl` yalnızca POSIX sistemlerde (Linux/macOS) mevcuttur; Windows'ta `ModuleNotFoundError` ile anında çöker.

**Karşılaştırma:** `uav_training/build_dataset.py:11-14` aynı problemi doğru şekilde çözmüş:

```python
import platform
if platform.system() == "Windows":
    import msvcrt
else:
    import fcntl
```

**Etki:** GPS modülü Windows ortamında hiçbir koşulda çalışamaz. Dosya kilitleme fonksiyonları (`_acquire_file_lock`, `_release_file_lock`) da yalnızca `fcntl` API'si kullanıyor.

**Öneri:** `build_dataset.py` ile aynı platform-aware pattern uygulanmalı; `_acquire_file_lock` / `_release_file_lock` fonksiyonları Windows dalı (`msvcrt.locking`) ile genişletilmeli.

---

### 2.2 GPS: `OneCycleLR` Resume Uyumsuzluğu

**Dosya:** `gps_training/train.py:218-223`

```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    steps_per_epoch=len(train_loader),
    epochs=TRAIN_CONFIG["epochs"],       # Toplam epoch sayısı
)
```

`OneCycleLR` oluşturulurken `epochs` parametresi yapılandırmadaki toplam epoch sayısını (100) alıyor. Resume durumunda `start_epoch > 0` olmasına rağmen scheduler hâlâ 100 epoch'luk bir toplam adım sayısıyla başlatılıyor. Ardından satır 239'da `scheduler.load_state_dict()` ile eski state yükleniyor.

**Problem senaryoları:**
- `train_loader` uzunluğu değişirse (farklı batch size veya veri seti boyutu), `total_steps` uyuşmaz ve scheduler patlar ya da LR profili bozulur.
- State dict yükleme başarısız olursa (uyarı verilip devam ediliyor, satır 241-243), scheduler sıfırdan başlar ama epoch döngüsü `start_epoch`'tan devam eder — LR warmup fazı atlanır, pik LR yanlış epoch'a denk gelir.

**Öneri:** Resume durumunda scheduler'ı `epochs=TRAIN_CONFIG["epochs"] - start_epoch` veya `last_epoch=start_epoch * steps_per_epoch` ile oluşturmak ya da daha sağlam bir scheduler (CosineAnnealingWarmRestarts gibi) kullanmak.

---

### 2.3 GPS: `GradScaler` Eksikliği

**Dosya:** `gps_training/train.py:264-275`

```python
with amp_ctx:                    # BF16 autocast
    output = model(img1, img2)
    loss = criterion(output, target)

loss.backward()                  # GradScaler yok!
```

AMP konteksti `dtype=torch.bfloat16` ile sabit kodlanmış. Ampere+ (sm_80+) GPU'larda BF16'nın geniş üs aralığı sayesinde GradScaler'a ihtiyaç duyulmaz — burası sorunsuz. Ancak pre-Ampere donanımda (T4, V100) BF16 donanım desteği yoktur ve PyTorch sessizce FP16'ya düşer. FP16'nın dar üs aralığı nedeniyle küçük gradyanlar sıfıra yuvarlanır (underflow); GradScaler olmadan eğitim diverge edebilir veya öğrenme durabilir.

**Etki:** T4 (Colab ücretsiz tier) üzerinde GPS eğitimi sessizce başarısız olabilir — loss düşmez veya NaN çıkar.

**Öneri:**
1. GPU capability kontrolü ekleyip pre-Ampere'de `GradScaler` kullanmak, veya
2. Pre-Ampere'de `dtype=torch.float16` ile açıkça FP16 + GradScaler kullanmak.

---

### 2.4 GPS: `best_model.pt` Atomik Olmayan Yazma

**Dosya:** `gps_training/train.py:344`

```python
torch.save(model.state_dict(), ARTIFACTS_DIR / "best_model.pt")
```

`last_model.pt` için satır 323-337'de doğru bir atomik yazma paterni uygulanmış (`.pt.tmp` → `os.replace()`). Ancak `best_model.pt` için bu pattern kullanılmamış — doğrudan hedef dosyaya yazılıyor.

**Etki:** Colab runtime bağlantı kopması veya SIGKILL sırasında `best_model.pt` yarım yazılabilir. En iyi model kaybedilir ve kurtarılamaz.

**Öneri:** `last_model.pt` ile aynı `tmp + os.replace` pattern'ini `best_model.pt` için de uygulamak.

---

## 3. Performans Değerlendirmesi

### 3.1 GPU Kullanımı

| Özellik                    | UAV Modülü                        | GPS Modülü                        |
|---------------------------|-----------------------------------|-----------------------------------|
| Mixed Precision (AMP)     | Ultralytics native AMP (BF16/FP16)| Manuel `torch.autocast` (BF16)    |
| TF32                      | Etkin (`allow_tf32 = True`)       | Etkin (`allow_tf32 = True`)       |
| cuDNN Benchmark           | Ultralytics yönetiyor             | `benchmark = True`                |
| `torch.compile`           | Koşullu (Ampere+ & Python <3.12)  | Koşulsuz, bare `except` ile       |
| GradScaler                | Ultralytics yönetiyor             | **Yok** (2.3'te detaylandırıldı)  |
| VRAM Fragmentation Guard  | `expandable_segments:True`        | Yok                               |

**UAV modülü** GPU kullanımında olgun: katmanlı OOM fallback (compile kapatma → batch yarılama → imgsz düşürme → batch tekrar yarılama), explicit batch sizing ile ~85-90% VRAM doluluğu, `kill_gpu_hogs()` ile agresif bellek temizliği.

**GPS modülü** daha yalın: tek batch size, OOM recovery yok, VRAM fragmentation koruması yok. Ancak ResNet-18 küçük bir model olduğundan pratikte OOM riski düşük.

### 3.2 I/O Darboğazları

**GPS Video Frame Okuma** (`gps_training/dataset.py:138`):

```python
cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
ret, frame = cap.read()
```

`CAP_PROP_POS_FRAMES` ile rastgele seek, özellikle H.264/H.265 kodlanmış videolarda keyframe'e kadar geri sarıp tüm ara frame'leri decode etmek zorundadır. DataLoader `shuffle=True` ile çalıştığında rastgele frame erişimi çok yavaşlar.

**Azaltıcı faktörler:** LRU frame cache (`_frame_cache_size=256`) ve ardışık frame çiftleri için tek seek optimizasyonu (`_read_video_pair` satır 164-172) mevcut. Ancak shuffle modda cache hit oranı düşük kalır.

**Öneri:** Eğitim öncesi tüm frame'leri diske webp/jpg olarak çıkarıp image loader kullanmak; veya LMDB/HDF5 gibi rastgele erişime uygun format tercih etmek.

**UAV Dataset Build** (`uav_training/build_dataset.py:332-335`):

```python
try:
    os.link(img_path, target_img_path)   # Hard link (hızlı, disk tasarrufu)
except OSError:
    shutil.copy2(img_path, target_img_path)  # Fallback: fiziksel kopya
```

İlk tercih hard link (sıfır disk kullanımı, anlık). `OSError` yalnızca cross-device durumunda düşer (Colab'da Drive → SSD arası). Oversampling `copy_suffix` ile yapıldığından aynı dosyanın birden fazla kopyası oluşabilir, ancak bu Colab SSD'de kabul edilebilir bir trade-off.

### 3.3 Bellek

**UAV `_is_checkpoint_valid`** (`uav_training/train.py:52-63`):

```python
torch.load(ckpt_path, map_location='cpu')
return True
```

Checkpoint geçerliliğini kontrol etmek için tüm model CPU belleğine yükleniyor (~50-100 MB per call). Yüklenen tensörler `try` bloğu sonunda scope'tan düşer ama GC'ye bağımlıdır — peş peşe çağrılarda geçici bellek şişmesi olabilir. Resume arama döngüsünde (satır 461-491) 3 ayrı aday için 3 kez tam yükleme yapılabilir.

**Öneri:** `torch.load` yerine dosya boyutu + magic bytes kontrolü ya da `torch.load(..., weights_only=True)` ile yalnızca tensör meta verisini yüklemek yeterli olabilir.

---

## 4. Eğitim Stabilitesi Analizi

### 4.1 UAV Modülü

| Parametre         | Değer                  | Değerlendirme                                           |
|-------------------|------------------------|---------------------------------------------------------|
| Optimizer         | AdamW                  | Doğru tercih; weight decay decoupled                    |
| LR (lr0)          | 0.001                  | AdamW için standart                                     |
| LR Schedule       | Cosine (`cos_lr=True`) | İyi; son epoch'larda LR yumuşak azalma                  |
| LR Final (lrf)    | 0.01                   | lr0'ın %1'i (1e-5) — kabul edilir minimum               |
| Warmup            | 5 epoch                | Yeterli; büyük batch'lerde kararlı başlangıç            |
| NBS               | batch ile eşit         | Ultralytics iç LR ölçeklemesi devre dışı — doğru        |
| Weight Decay      | 0.0005                 | Standart                                                |
| Gradient Clipping | Ultralytics yönetiyor  | Varsayılan `max_norm=10.0`                              |
| Augmentation      | Zengin                 | mosaic, copy_paste, flipud/lr, HSV, scale, bgr          |
| Two-Phase         | 50+15 epoch            | Phase 2: düşük LR, düşük mosaic, yüksek imgsz — iyi    |
| Early Stopping    | `patience=30`          | 65 epoch toplam ile orantılı                            |

**Değerlendirme:** UAV eğitim stabilitesi yüksek. İki fazlı profil, güçlü augmentasyon pipeline'ı ve katmanlı OOM recovery mekanizması ile endüstriyel seviyeye yakın.

### 4.2 GPS Modülü

| Parametre         | Değer                  | Değerlendirme                                           |
|-------------------|------------------------|---------------------------------------------------------|
| Optimizer         | AdamW                  | Doğru                                                   |
| LR               | 1e-4                   | ResNet-18 fine-tune için uygun                          |
| LR Schedule       | OneCycleLR             | Resume ile uyumsuz (bkz. 2.2)                          |
| Warmup            | OneCycleLR iç warmup   | ~%30 epoch; kabul edilir                                |
| Gradient Clipping | `max_norm=1.0`         | Var ve doğru uygulanmış                                 |
| NaN Guard         | `torch.isfinite(loss)` | Var — diverge algılama mevcut (satır 272)               |
| Loss              | MSELoss                | Regresyon için standart; ancak outlier'lara hassas       |
| Augmentation      | **Yok**                | Sadece `cv2.resize` — ciddi overfitting riski           |
| Early Stopping    | **Yok**                | 100 epoch boyunca çalışır, overfit kontrolü yok         |

**Değerlendirme:** GPS eğitim stabilitesi düşük-orta. NaN guard ve gradient clipping pozitif; ancak augmentasyon eksikliği, early stopping yokluğu ve scheduler resume sorunu ciddi riskler oluşturuyor.

**Augmentasyon önerileri:** Rastgele yatay çevirme, renk jitter (brightness, contrast), Gaussian blur, hafif rotasyon (±5°). Siamese ağ için her iki frame'e aynı augmentasyon uygulanmalı.

---

## 5. MLOps Olgunluk Değerlendirmesi

### 5.1 Checkpoint Yönetimi

| Özellik                    | UAV                                | GPS                               |
|---------------------------|------------------------------------|------------------------------------|
| Auto-save periyodu        | Her 5 epoch (`save_period=5`)      | Her epoch                          |
| Atomik yazma              | Ultralytics kendi yönetiyor        | `last_model.pt` atomik, `best_model.pt` **değil** |
| Backup rotasyonu          | Drive sync (async thread)          | `last_model.bak` — 1 nesil        |
| Resume desteği            | CLI + lokal + Drive arama zinciri  | Lokal + backup fallback            |
| Checkpoint validasyonu    | `_is_checkpoint_valid` (boyut + load) | `_is_checkpoint_valid` (boyut + load) |
| Corrupt recovery          | Zincirli fallback (CLI → lokal → Drive) | Backup fallback                |

### 5.2 Logging ve İzlenebilirlik

| Özellik              | UAV                              | GPS                          |
|----------------------|----------------------------------|------------------------------|
| Metrik kaydı         | Ultralytics `results.csv`        | Python listesi (sadece stdout) |
| Precision policy log | `_log_precision_policy()` — var  | Yok                          |
| Hardware banner      | Auto-detect → detaylı print      | Auto-detect → detaylı print  |
| Config dump          | `print_training_config()` — var  | `print_training_config()` — var |
| Loss plot            | Ultralytics otomatik             | `matplotlib` ile (son epoch) |
| TensorBoard/WandB    | Yok                              | Yok                          |

### 5.3 CI/CD

**Mevcut durum** (`.github/workflows/lint.yml`):

```yaml
- flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
- python -m compileall -q .
```

Yalnızca sözdizimi hataları (`E9`: syntax error, `F63`: assertion test, `F7`: syntax, `F82`: undefined name) ve derleme kontrolü yapılıyor.

**Eksikler:**
- `pytest` çalıştırılmıyor (5 test dosyası mevcut ama CI'da çalışmıyor)
- Type checking (mypy/pyright) yok
- Code coverage ölçümü yok
- GPU-free smoke test yok
- Dependency vulnerability taraması yok

### 5.4 Versiyon Yönetimi

- `__version__` iki dosyada senkron tutulması gereken bir pattern: `uav_training/__init__.py` ve `uav_training/train.py` — her ikisi de `0.8.8`; şu anda senkron.
- `CHANGELOGS.md` düzenli güncelleniyor (son giriş: `0.0.28`).
- GPS modülünde versiyon yönetimi **hiç yok** — `gps_training/__init__.py` dosyası mevcut değil.

### 5.5 Bağımlılık Yönetimi

`requirements.txt` yalnızca alt sınır içeriyor:

```
ultralytics>=8.2.0
torch>=2.0.0
torchvision>=0.15.0
...
```

Major versiyon değişikliklerinde (ör. `ultralytics` 9.x veya `torch` 3.x) API kırılmaları kaçınılmaz. Üst sınır yokluğu bilinçli bir Colab uyumluluk tercihidir (Colab'ın önceden yüklü paketleriyle çakışma önlenir), ancak üretim ortamı için risklidir.

---

## 6. Belirsizlikler ve Koşullu Riskler

### 6.1 Runtime'a Bağlı Riskler

| Risk                                        | Koşul                                    | Etkisi                                      |
|---------------------------------------------|------------------------------------------|---------------------------------------------|
| GPS BF16 → FP16 sessiz düşüş               | Pre-Ampere GPU (T4, V100)               | GradScaler olmadan gradyan underflow         |
| UAV `torch.compile` OOM                     | Python ≥3.12 veya yetersiz VRAM         | Compile devre dışı kalır; performans kaybı   |
| Drive FUSE senkronizasyon gecikmesi         | Colab yüksek Drive trafiği               | Checkpoint kaybı riski (async sync)          |
| `OneCycleLR` state uyumsuzluğu             | Resume + farklı batch size               | LR profili bozulması                         |
| `cv2.VideoCapture` thread safety            | `num_workers > 0` (DataLoader)           | Potansiyel race condition veya segfault      |
| Hard link fallback → fiziksel kopya         | Cross-device (Drive → SSD)               | Disk alanı 2-3x kullanım, build süresi artış|

### 6.2 Veri Boyutuna Bağlı Riskler

| Risk                                        | Koşul                                    | Etkisi                                      |
|---------------------------------------------|------------------------------------------|---------------------------------------------|
| GPS az sequence ile overfit                 | Mevcut ~8 sequence                       | Modelin yeni rotalara genelleyememesi        |
| Megaset smart sampling dengesizliği         | Vehicle:Human oranı çok farklı           | Sınıf dengesizliği; vehicle öğrenme baskısı  |
| RAM cache yetersizliği                      | Veri seti > 100GB, RAM < 20GB           | `cache="disk"` fallback — I/O yavaşlaması   |

### 6.3 Sınıf Mapping Tutarsızlığı

**`uav_training/config.py:33-44` — `TARGET_CLASSES`:**

```python
TARGET_CLASSES = {
    "uap": 0, "uai": 1, "human": 2, "vehicle": 3, ...
}
```

**`uav_training/build_dataset.py:446-457` — Gerçek Eğitim Mapping'i:**

```python
'names': {0: 'vehicle', 1: 'human', 2: 'uap', 3: 'uai'}
```

Bu iki mapping **birbiriyle çelişiyor**:

| Sınıf   | `TARGET_CLASSES` | Gerçek Eğitim (`dataset.yaml`) |
|---------|------------------|--------------------------------|
| vehicle | 3                | **0**                          |
| human   | 2                | **1**                          |
| uap     | 0                | **2**                          |
| uai     | 1                | **3**                          |

`TARGET_CLASSES` yalnızca `audit.py`'de veri seti tarama amacıyla kullanılıyor; eğitim sırasında `build_dataset.py` MAPPINGS'teki açık `map` sözlükleri esas alınıyor. Bu nedenle **eğitim doğru çalışıyor**. Ancak `TARGET_CLASSES`'ı referans alan herhangi bir gelecek kod (inference, post-processing, API) yanlış sınıf eşleştirmesi yapacaktır.

**Öneri:** `TARGET_CLASSES`'ı `dataset.yaml` ile tutarlı hale getirmek veya `build_dataset.py`'den gerçek mapping'i import ederek tek kaynak (single source of truth) sağlamak.

---

## 7. Genel Sağlık Skoru

### Puanlama Tablosu

| Kategori                  | Puan (0-10) | Ağırlık | Katkı  |
|---------------------------|-------------|---------|--------|
| Kod Kalitesi & Okunabilirlik | 7          | %15     | 1.05   |
| Eğitim Stabilitesi (UAV)    | 8          | %15     | 1.20   |
| Eğitim Stabilitesi (GPS)    | 4          | %15     | 0.60   |
| Platform Uyumluluğu         | 5          | %10     | 0.50   |
| Veri Güvenliği & Bütünlüğü  | 6          | %10     | 0.60   |
| MLOps Olgunluğu             | 5          | %10     | 0.50   |
| Performans Optimizasyonu     | 7          | %10     | 0.70   |
| Hata Kurtarma & Dayanıklılık| 7          | %10     | 0.70   |
| Test & CI Kapsama            | 3          | %5      | 0.15   |

### Genel Skor: **6.0 / 10**

### Gerekçe

**Güçlü yönler:**
- UAV eğitim hattı endüstriyel kaliteye yakın: katmanlı OOM recovery, otomatik donanım algılama, iki fazlı eğitim profili, zengin augmentasyon pipeline'ı ve Drive senkronizasyonu.
- Veri bütünlüğü iyi düşünülmüş: megaset sahne tabanlı split (veri sızıntısı önleme), bbox NaN/range/size guard, duplicate line filter.
- Seed yönetimi kapsamlı (PYTHONHASHSEED, random, numpy, torch, CUDA).
- Checkpoint resume zinciri (CLI → lokal → Drive) sağlam.

**Zayıf yönler:**
- GPS modülü UAV'a kıyasla belirgin şekilde daha az olgun: `fcntl` platform hatası, GradScaler eksikliği, augmentasyon yokluğu, atomik olmayan best_model yazımı.
- Sınıf indeks tutarsızlığı (`TARGET_CLASSES` vs gerçek mapping) gelecekte ciddi hatalara neden olabilecek bir teknik borç.
- CI pipeline'ında testler çalıştırılmıyor; mevcut 5 test dosyası sadece lokalde anlamlı.
- TensorBoard/WandB gibi experiment tracking entegrasyonu yok.

**Sonuç:** UAV modülü üretime yakın ancak GPS modülü henüz prototip aşamasında. Kritik riskler (özellikle #1 ve #3) düzeltilmeden GPS eğitiminin güvenilir sonuç vermesi beklenmemelidir.
