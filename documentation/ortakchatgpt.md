# Ortak Eğitim ve MLOps Denetim Raporu (YOLO11m Fine-Tune + Colab A100)
**Dosya:** `ortakchatgpt.md`  
**Kapsam:** `claudeai_rapor.md` + `chatgpt_rapor.md` birleşimi, çifte doğrulama (statik kontrol + koşum kanıtı) yaklaşımı.

> Bu rapor, iki kaynaktaki önerileri çakışma ve risklerine göre temizleyip tek plana indirger. Her madde için: **Sorun → Önem → Nasıl düzeltilir → Nasıl yapılır → Kanıt (çifte doğrulama) → Öncelik sırası** verilir.

---

## 0) Hedefler ve Sınırlar
### Hedef metrikler
- **mAP50 (IoU=0.5)** artışı (özellikle UAP UAI sınıfları)
- **Recall** artışı (küçük hedefler ve dengesiz sınıflar)
- **Eğitim throughput (it/s)** artışı (A100 avantajını kullan)
- **Kararlılık:** Colab restart sonrası **resume** ile kayıpsız devam

### Temel prensip: Çifte doğrulama
Her değişiklik iki kanıtla “tamamlandı” sayılır:
1) **Statik kontrol:** Kodda veya konfigde gerçekten yapıldı mı?
2) **Koşum kanıtı:** Logda veya çıktıda gerçekten çalıştığını gösteren satır var mı?

---

## 1) Önem Ölçeği
- **CRITICAL:** Zamanı yakar, eğitimi bozar, resume kırılır, skor dramatik düşer.
- **HIGH:** Skoru veya hızı belirgin etkiler, yarışma gününe risk taşır.
- **MEDIUM:** Performansı dalgalandırır, birikince can yakar.
- **LOW:** Hijyen, sürdürülebilirlik, sürpriz kırılmaları azaltır.

---

## 2) Sorunlar Listesi (Önem Düzeyi + Çözüm)
Aşağıdaki maddeler iki raporun ortak sonucu olarak düzenlenmiştir.

### CRITICAL-1: Colab restart sonrası resume kırılgan (checkpoint yolu kayboluyor veya yanlış aranıyor)
**Sorun:** Runtime reset olduğunda eğitim sıfırdan başlayabiliyor. İşlem birimi ve zaman kaybı.  
**Risk azaltımı / metrik artışı:** Süreklilik artar, boşa eğitim engellenir.

**Nasıl düzeltilir (tasarım):**
- Bootstrap script checkpoint bulduğunda `--resume` ile birlikte **checkpoint path** de geçirilmeli.
- Train tarafı sadece local run klasörüne değil Drive ve alternatif run dizinlerine de fallback arama yapmalı.

**Nasıl yapılır (uygulama adımı):**
- `scripts/colab_bootstrap.py`:
  - Drive mount sonrası `runs/` altında `last.pt` araması
  - Bulursa `python ... train.py --resume --model <last.pt>` şeklinde çağır
- `uav_training/train.py`:
  - “checkpoint candidate paths” listesi oluştur (local + Drive)
  - Bulunan en yeni checkpoint ile resume et

**Çifte doğrulama (kanıt):**
- Statik: bootstrap komutunda `--model <ckpt>` var, train tarafında fallback arama var.
- Koşum: logda ikisi birlikte görünmeli:
  - `Resuming from checkpoint: .../last.pt`
  - Epoch 0’dan başlamamalı (ör: epoch 12’den devam)

**Öncelik:** 1

---

### CRITICAL-2: Determinism zorla açık kalırsa A100 hız düşer ve it/s dalgalanır
**Sorun:** `cudnn.deterministic=True` ve `cudnn.benchmark=False` hızın canını yakar.  
**Risk azaltımı / metrik artışı:** it/s artar, GPU satürasyonu yükselir.

**Nasıl düzeltilir (tasarım):**
- `setup_seed(deterministic: bool)` parametreli olsun.
- Default **deterministic=False** (hız modu). Debug gerektiğinde True açılabilsin.

**Nasıl yapılır (uygulama adımı):**
- `uav_training/config.py` içine `deterministic: false` ekle
- `uav_training/utils/seed.py` benzeri dosyada:
  - deterministic False iken `cudnn.benchmark=True`
  - deterministic True iken `cudnn.benchmark=False` ve gerekli ayarlar

**Çifte doğrulama (kanıt):**
- Statik: configte `deterministic` var ve tek yerden yönetiliyor.
- Koşum: eğitim başında tek satır log:
  - `deterministic=False cudnn.benchmark=True`

**Öncelik:** 2

---

### HIGH-4: TF32 kapalıysa A100 avantajı boşa gider
**Sorun:** A100 tensor core verimi düşer, it/s düşer.  
**Risk azaltımı / metrik artışı:** it/s artar, eğitim süresi kısalır.

**Nasıl düzeltilir (tasarım):**
- TF32 açık olmalı, matmul precision ayarı belirlenmeli.

**Nasıl yapılır (uygulama adımı):**
- Eğitim başında bir fonksiyonda:
  - `allow_tf32=True`
  - `matmul_precision="high"`
- Tek bir yerde yönet ve logla.

**Çifte doğrulama (kanıt):**
- Statik: tek fonksiyon, config ile kontrol.
- Koşum: `TF32 ENABLED` satırı.

**Öncelik:** 3

---

### HIGH-2: `optimizer=auto` LR ve momentum’u sessizce override edebilir
**Sorun:** Fine tune ayarların etkisiz kalabilir. Yakınsama bozulur.  
**Risk azaltımı / metrik artışı:** mAP artışı daha öngörülebilir olur.

**Nasıl düzeltilir (tasarım):**
- Optimizer açıkça seçilsin (SGD veya AdamW).
- LR ve momentum bilinçli yönetilsin.

**Nasıl yapılır (uygulama adımı):**
- `uav_training/config.py`:
  - `optimizer: SGD` veya `AdamW`
  - `lr0`, `momentum`, `weight_decay` net tanımla
- Eğitim çağrısında `optimizer=auto` kullanma.

**Çifte doğrulama (kanıt):**
- Statik: configte optimizer açık, train çağrısında auto yok.
- Koşum: Ultralytics logunda optimizer satırı senin seçtiğin gibi görünüyor ve lr0 uyuyor.

**Öncelik:** 4

---

### HIGH-3: Etiket hijyeni (bbox clamp) hatayı maskeleyip mAP’i öldürebilir
**Sorun:** Clamp ile invalid bbox’lar “düzgünmüş gibi” görünür, eğitim ground truth bozulur.  
**Risk azaltımı / metrik artışı:** mAP ve recall artar, hatalar erken yakalanır.

**Nasıl düzeltilir (tasarım):**
- Clamp yerine “strict validation + raporlama” yap.
- Çok küçük taşmalarda eps toleransı olabilir, ciddi hatada discard + sayaç.

**Nasıl yapılır (uygulama adımı):**
- `uav_training/build_dataset.py`:
  - `out_of_range_bbox_count`
  - `too_small_bbox_count`
  - `discarded_labels_count`
  - class bazlı örnek sayımı
- Build sonunda JSON veya düz metin raporu üret.

**Çifte doğrulama (kanıt):**
- Statik: clamp yok, sayaçlar var, rapor çıktısı var.
- Koşum: dataset build sonunda sayılar basılıyor ve split bazında özet var.

**Öncelik:** 5

---

### HIGH-1: BF16 hedefleniyor ama gerçekten aktif olduğu garanti değil
**Sorun:** BF16 “niyet” olarak var, pratikte FP16 veya FP32’ye düşebilir. Bu durumda beklenen hız ve VRAM kazanımı gelmez.  
**Risk azaltımı / metrik artışı:** it/s artar, overflow riski düşer, VRAM headroom artar.

**Nasıl düzeltilir (tasarım):**
- BF16 sadece Ampere+ GPU (A100 vb) için denenmeli.
- En önemli şart: “BF16 aktif” iddiası **logla kanıtlanmalı**.
- Ultralytics sürümü bounded pin ile kontrol edilmeli.

**Nasıl yapılır (uygulama adımı):**
- Başlangıçta GPU adını ve compute capability’yi logla.
- A100 algılandıysa BF16 için:
  - `autocast(dtype=torch.bfloat16)` kullanımı veya framework destekli ayar
- `requirements.txt`:
  - `ultralytics` için bounded aralık kullan (ör: `>=8.x,<8.y` mantığı)
  - Torch CUDA sürümü uyumlu olmalı

**Çifte doğrulama (kanıt):**
- Statik: BF16 guard koşulu var, bounded pin var.
- Koşum: logda şu satırlar var:
  - `GPU: NVIDIA A100 ...`
  - `AMP dtype: bfloat16` veya `BF16 ENABLED`

**Öncelik:** 6

---

### MEDIUM-1: Drive sync thread I/O ile çakışıp it/s dalgalandırabilir
**Sorun:** Sync işlemleri epoch ortasında I/O çakışması yaratır.  
**Risk azaltımı / metrik artışı:** it/s stabil olur.

**Nasıl düzeltilir:**
- Sync periyodunu büyüt, mümkünse epoch sonuna al.
- “quiet window” ile batch okuma anında sync yapma.

**Çifte doğrulama:**
- Koşum: sync logları epoch ortasında spam değil.

**Öncelik:** 7

---

### MEDIUM-2: DataLoader worker ve CPU thread politikası yanlışsa GPU bekler
**Sorun:** CPU data pipeline dar boğaz olur.  
**Risk azaltımı / metrik artışı:** GPU utilization artar.

**Nasıl düzeltilir:**
- `workers` otomatik seçilsin (cpu_count’a göre).
- Gereksiz `torch.set_num_threads(1)` gibi throttle import-time kullanılmasın.

**Çifte doğrulama:**
- Koşum: GPU utilization daha yüksek, it/s daha stabil.

**Öncelik:** 8

---

### LOW-1: Bağımlılık sürümleri pin veya bounded değilse Colab güncellemesiyle sessiz kırılır
**Sorun:** Aynı kod farklı günde farklı sonuç veya hata.  
**Risk azaltımı / metrik artışı:** tekrar üretilebilirlik artar.

**Nasıl düzeltilir:**
- Kritik paketlerde bounded pin.
- Torch CUDA uyumuna dikkat.

**Öncelik:** 9

---

### LOW-2: torch.compile önerisi güçlü ama opsiyonel tutulmalı
**Sorun:** Compile bazı sürümlerde kırılgan veya warmup süresi ekleyebilir.  
**Risk azaltımı / metrik artışı:** it/s artabilir, ama kontrollü açılmalı.

**Nasıl düzeltilir:**
- Config flag ile aç kapa.
- A100’da `reduce-overhead` denenebilir.

**Öncelik:** 10

---

## 3) Uygulama Sırası (Önerilen)
1) **Resume sağlamlaştır** (CRITICAL-1)  
2) **Determinism parametreleştir** (CRITICAL-2)  
3) **TF32 + matmul precision** (HIGH-4)  
4) **Optimizer auto kaldır** (HIGH-2)  
5) **Label strict validation + raporlama** (HIGH-3)  
6) **BF16 guard + kanıt logu + bounded pin** (HIGH-1)  
7) **Drive sync çatışmasını azalt** (MEDIUM-1)  
8) **Workers ve thread politikası** (MEDIUM-2)  
9) **Dependency hijyeni** (LOW-1)  
10) **torch.compile opsiyonel** (LOW-2)

> Kural: Her adımın sonunda “çifte doğrulama” kanıtı logda yoksa o adım tamam değildir.

---

## 4) Uygulama Planı (Modül Bazlı)
### Modüller
- `scripts/colab_bootstrap.py`: Drive mount, run klasörü keşfi, checkpoint bulma, doğru komutla eğitimi başlatma
- `uav_training/train.py`: config okuma, fallback checkpoint arama, determinism ve compute optimizasyonları, eğitim başlatma
- `uav_training/config.py`: tek kaynak konfig (determinism, optimizer, amp bf16, tf32, compile, workers)
- `uav_training/build_dataset.py`: label audit, sayaçlar, rapor üretimi

### Veri akışı
- Dataset build → audit raporu üret
- Train bootstrap → checkpoint keşfi → train çağrısı
- Train → per epoch log → opsiyonel sync (epoch sonu)

---

## 5) Test Planı (Simülasyon + Metrikler)
### 5.1 Eğitim metrikleri
- **mAP50 (IoU=0.5)**: train ve val
- **Recall**: class bazlı (özellikle UAP UAI)
- **Label audit raporu**:
  - discard edilen bbox sayısı
  - out_of_range sayısı
  - class distribution

### 5.2 Hız metrikleri
- it/s trendi: ilk epoch sonrası stabil olmalı
- GPU utilization: data loader beklemesi düşük olmalı
- VRAM tepe: OOM payı bırakacak şekilde

### 5.3 Dayanıklılık testleri
- Colab restart simülasyonu:
  - checkpoint var iken resume doğru mu?
  - epoch numarası devam ediyor mu?
- Drive bağlantısı geç gelirse:
  - bootstrap retry/backoff var mı?

---

## 6) Yarışma Günü Operasyon Listesi (Kısa)
- **Network:** ethernet, tek IP, retry/backoff
- **Log:** timestamp, frame_id, latency, model_version
- **Fail-safe:** frame yoksa kilitlenme yok, yeniden dene
- **Model path:** tek final model yolu, inference default’u uyumlu
- **Sürüm kilidi:** bağımlılıklar dondurulmuş olmalı
- **Zaman yönetimi:** son gün sadece “risk azaltan” değişiklik

---

## 7) Değişikliklerin Beklenen Etkisi Özeti
- Resume sağlamlaştırma → işlem birimi kaybını engeller
- Determinism off (default) → it/s artar
- TF32 on → A100 tensor core verimi artar
- Optimizer auto kaldırma → LR kontrolü geri gelir, yakınsama iyileşir
- Label strict audit → mAP düşüren sessiz hatalar yakalanır
- BF16 guard + kanıt → hız ve VRAM kazanımı gerçek olur
- I/O ve workers ayarı → throughput stabil olur
- Dependency bounded → Colab sürpriz kırılmaları azalır

---

## 8) Kabul Kriterleri (Bu rapor bitti sayılması için)
- Her CRITICAL ve HIGH madde için:
  - Statik kontrol tamam
  - Koşum log kanıtı mevcut
- Dataset audit raporu üretiyor ve saklanıyor
- Resume testinde eğitim sıfırdan başlamıyor
