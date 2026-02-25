# ML Eğitim Hattı — Statik Kod Denetim Raporu

**Proje:** UAV Training — YOLO Object Detection  
**Sürüm:** v0.8.22  
**İnceleme Kapsamı:** `build_dataset.py`, `train.py`, `config.py`, `audit.py`, `inference.py`, `visualize_dataset.py`, `colab_bootstrap.py`, `requirements.txt`, `requirements-dev.txt`, `__init__.py`, `datasets.md`, `README.md`, `cleanup.sh`

---

## 1. Özet Bulgular

1. **Tekrarlanabilirlik açığı (Kritik):** `build_dataset.py` içindeki `sampling_rate`-tabanlı rastgele örnekleme (`random.sample()`), megaset dışındaki tüm datasetler için `set_seed()` çağrısı olmaksızın çalışmaktadır. Bu, her çalıştırmada farklı görüntü seçimine yol açabilir ve dataset kompozisyonunu non-deterministik hale getirir.

2. **OOM kurtarma sonrası `nbs` tutarsızlığı:** `train.py` / `_train_single_phase()` içinde OOM kurtarma zinciri `imgsz=640` adımında `nbs` değerini güncellemekte başarısız olmaktadır. Bu, Ultralytics dahili LR ölçekleme mantığının yanlış referans batch boyutuna dayanmasına neden olur.

3. **`dataset.yaml` göreli yol riski:** `build_dataset.py` son bölümünde `path: '.'` ile yaml üretilmektedir. Bu, YOLO'nun DATASET_DIR ile eşleşmeyen bir CWD'den çalıştırılması durumunda görüntü yollarını yanlış çözümlemesine ve sessiz bir yükleme hatasına yol açabilir.

4. **`is_sample` tespiti sessiz hata (audit.py):** `audit_directory()` içinde `is_sample = True` atama koşulu hem içerik eşleşmesi hem de `r.name.lower()` içinde `"örnek"` veya `"sample"` gerektirmektedir. İçerik eşleşmesi tek başına yeterli değildir; bu nedenle standart adlı klasörlerdeki "inference only" README'ler atlanacaktır.

5. **Phase 2 `lrf` mirası:** `train.py` içindeki `phase2_overrides`, `lr0`'ı geçersiz kılmakta fakat `lrf`'i kılmamaktadır. Phase 2, TRAIN_CONFIG'ten `lrf=0.01` değerini miras almakta ve azaltılmış `lr0` ile birlikte son öğrenme oranı ondalık bir düzey daha düşmektedir. Bu örtük bir davranıştır; kasıtlı fine-tuning ise de belgede açıkça yer almamaktadır.

6. **Bağımlılık sürümü sabitleme eksikliği:** `requirements.txt` yalnızca minimum sürüm kısıtları ve geniş aralıklar içermektedir; kilit dosyası (lock file) yoktur. Ortamlar arasında minor sürüm sapması olasıdır.

7. **`_SYNC_IN_FLIGHT` globali — teorik kilitlenme riski:** `kill_gpu_hogs()` içindeki while döngüsü `_SYNC_IN_FLIGHT=True` kalırsa sonsuza kadar bekler. Daemon thread'in `finally` bloğunu tamamlamaması (süreç kapatımı veya zorla sonlandırma) bu durumu tetikleyebilir.

8. **Git commit hash loglama eksikliği:** `full_attempt_args.yaml` ve `args.yaml` kayıt mekanizması mevcut olmakla birlikte, eğitim sırasında kaynak commit hash'i hiçbir artefakt dosyasına yazılmamaktadır. Bu, model sürümü ile kod revizyonu arasındaki izlenebilirliği kırmaktadır.

---

## 2. Kritik Riskler

### R-01 — Non-megaset Datasetlerde Deterministik Olmayan Örnekleme

**Risk Tanımı:** `build_dataset.py` / `_process_image_files()` fonksiyonu, `sampling_rate < 1.0` olduğunda `random.sample(image_files, k)` çağrısı yapmaktadır (yaklaşık satır 192). `set_seed()` yalnızca megaset'in scene-based split bloğunda çağrılmaktadır (satır 434). `drone-vision-project`, `Uap-UaiAlanlariVeriSeti` ve benzeri datasetler için rastgele örnekleme, Python işleminin anlık random durumuna bağlıdır.

**Etkisi:** Doğruluk / Tekrarlanabilirlik — Her `build_dataset` çalıştırmasında farklı eğitim görüntüsü alt kümesi seçilir; eğitimler arası karşılaştırma geçersizleşir.

**Olasılık:** Orta — Şu an `sampling_rate=1.0` olduğu için aktif değil. Ancak config değiştiğinde anında devreye girer; koruyucu yoktur.

**Tetikleyici Senaryolar:** Herhangi bir MAPPINGS girişinde `sampling_rate < 1.0` ayarlandığında.

**Tespit Yöntemi:** Aynı kaynak veriyle iki kez `build_dataset.py` çalıştırıp `train/images` içindeki dosya isimlerini karşılaştırarak fark gözlemlenir.

**Mimari Düzeyde İyileştirme Yönü:** Tüm split işleme döngüsü öncesinde veya fonksiyon girişinde merkezi bir seed mekanizması devreye alınmalıdır; megaset seed çağrısı da bu merkezi mekanizmaya taşınmalıdır.

---

### R-02 — OOM Kurtarma Zincirinde `nbs` / `batch` Tutarsızlığı

**Risk Tanımı:** `train.py` / `_train_single_phase()` — OOM kurtarma kodu `imgsz=640` fallback adımında (satır 370–372) `nbs` değerini güncellemez. `batch` değişmediği hâlde `nbs` önceki adımdan miras kalır. Ultralytics, dahili LR ölçeklemesini `nbs / current_batch` oranına göre yapar; bu oran bozulduğunda gerçek LR eğitim genelinde hedeflenenden farklılaşır.

**Etkisi:** Eğitim Dinamikleri — Yanlış efektif LR, özellikle geniş batch ile yüksek imgsz kombinasyonundan düşük imgsz'ye geçişte kayıp divergence'ına veya yakınsamanın yavaşlamasına neden olabilir.

**Olasılık:** Orta — Yalnızca OOM kurtarma zinciri ikinci OOM adımını yaşadığında tetiklenir (compile=False zaten uygulandıysa).

**Tetikleyici Senaryo:** A100/H100'de `compile=True` ile başlayan eğitimde ilk OOM → `compile=False`, ikinci OOM → `imgsz=640` (nbs güncellenmez).

**Tespit Yöntemi:** Fallback sonrası loglanmış `full_attempt_args.yaml` içinde `nbs` ve `batch` ayrışmasına bakılır.

**Mimari Düzeyde İyileştirme Yönü:** Her OOM kurtarma adımı sonrasında `nbs` değeri, o adımdaki geçerli `batch` değerine eşitlenecek şekilde OOM kurtarma blokları normalize edilmelidir.

---

### R-03 — `dataset.yaml` Göreli Yol ile CWD Bağımlılığı

**Risk Tanımı:** `build_dataset.py` sonunda üretilen `dataset.yaml` dosyasında `path: '.'` kullanılmaktadır. YOLO bu değeri CWD'ye göre çözümler. `colab_bootstrap.py` satır 517'de `os.chdir(REPO_DIR)` yapmaktadır; REPO_DIR ile DATASET_DIR farklı konumlardaysa (`/content/repo` vs `/content/dataset_built`) görüntü yolları yanlış çözümlenir.

**Etkisi:** Çökme / Sessiz Veri Kaybı — YOLO başarısız görüntü yüklemesini atlar; eğitim sıfır örnekle veya kısmi veriyle ilerleyebilir ve metrik anormalliği geç fark edilebilir.

**Olasılık:** Orta — Yerel geliştirmede CWD muhtemelen proje köküdür ve sorun yaşanmaz. Colab modunda `chdir(REPO_DIR)` sonrası `DATASET_DIR=/content/dataset_built` olduğunda uyumsuzluk oluşur.

**Tetikleyici Senaryo:** Colab çalışma modunda `build_dataset.py` çalıştıktan sonra `train.py` başlatılması.

**Tespit Yöntemi:** Eğitim loglarında `No labels found` uyarısı veya ilk epoch'ta train görüntüsü = 0 bildirimi.

**Mimari Düzeyde İyileştirme Yönü:** `path` alanı, DATASET_DIR'ın mutlak yolu olarak yazılmalı veya hiç kullanılmamalıdır. Alternatif olarak, `train`, `val`, `test` alanlarında da mutlak yollar kullanılabilir.

---

### R-04 — `audit.py` Sessiz `is_sample` Tespiti Hatası

**Risk Tanımı:** `audit_directory()` içinde `is_sample = True` atama koşulu (satır 107–110) yalnızca `r.name.lower()` içinde `"örnek"` veya `"sample"` bulunduğunda aktif olmaktadır; içerik eşleşmesi (README içinde "test only", "inference only") tek başına yeterli değildir. Standart adlı klasörlerdeki (örn. `Uap-UaiAlanlariVeriSeti`) inference-only README'ler bu kontrolü atlar; ilgili dataset INCLUDE olarak işaretlenebilir.

**Etkisi:** Doğruluk — Inference-only veya kısmi bir dataset yanlışlıkla eğitime dahil edilir; etiket kalitesi bilinmez.

**Olasılık:** Düşük–Orta — Koşul hem içerik hem dosya adı eşleşmesini gerektirir; bu, pratikte nadiren aynı anda sağlanır.

**Tespit Yöntemi:** Audit raporunda INCLUDE statüsüyle işaretlenmiş datasetlerin README içeriklerine manuel bakılması.

**Mimari Düzeyde İyileştirme Yönü:** `is_sample` atama mantığı, içerik tespitinin dosya adından bağımsız çalışacağı şekilde ayrıştırılmalıdır.

---

## 3. Performans Değerlendirmesi

### P-01 — Koşullu `rsync` ile Drive Tam Dizin Senkronizasyonu

**Risk Tanımı:** `colab_bootstrap.py` / `_periodic_runs_sync()` (satır 561), `save_period=1` ile yüksek frekanslı checkpoint kaydında `/content/runs/` altındaki tüm yapıyı (confusion matrix, batch görsel, results.png vb.) Drive'a rsync eder. Toplam artifact boyutu epoch başına onlarca MB olabilir.

**Etkisi:** Performans — Drive FUSE'a yönelik ağır yazmalar GPU batch döngüsüne asenkron bir yük bindirir; Colab'da FUSE latensi birikimiyle toplam eğitim süresi uzayabilir.

**Olasılık:** Orta — `UAV_SYNC_INTERVAL_SEC > 0` olduğunda aktif.

**Tetikleyici Senaryo:** Yüksek frekanslı epoch bazlı kayıt (`save_period=1`) ile aktif periyodik senkronizasyon.

**Tespit Yöntemi:** `rsync` çağrısı etrafında zaman ölçümü; Drive yazma süresi ile GPU utilization düşüşünün örtüşmesi.

**Mimari Düzeyde İyileştirme Yönü:** Periyodik rsync yalnızca `weights/` altındaki checkpoint dosyalarını hedeflemeli; görsel artifact dosyaları yalnızca eğitim sonu final sync'inde kopyalanmalıdır.

---

### P-02 — Oversample Döngüsünde `_seen_lines` Set'inin Kapsam Sınırı

**Risk Tanımı:** `build_dataset.py` / `_process_image_files()` — `_seen_lines = set()` her görüntü için sıfırlanmaktadır. Bu, aynı görüntünün farklı oversample kopyalarında aynı bbox satırlarını tutmaktadır; bu beklenen davranıştır. Ancak çok sayıda tiny bbox'a sahip görüntülerde (`oversample_count=3`) tek bir görüntüden gelen bbox setleri bellekte üç kez tutulmaktadır.

**Etkisi:** Performans — Sınırlı RAM ortamında (8 GB altı) yoğun UAP/UAI datasetiyle büyük label dosyaları yığılabilir.

**Olasılık:** Düşük — Mevcut veri ölçeğinde kritik değil; ölçek büyüdükçe belirginleşir.

**Tespit Yöntemi:** Büyük datasette `_process_image_files()` çalışırken RSS bellek izleme.

**Mimari Düzeyde İyileştirme Yönü:** `_seen_lines` bir görüntü içi duplikasyon önleyicisi olarak tasarlanmıştır; oversample semantiği değişmez. Referans güncellenmesi gerekirse satır bazlı deduplication dışarı taşınabilir.

---

### P-03 — Hard-Link ile Image Kopyalama ve Cache Senkronizasyonu

**Risk Tanımı:** `build_dataset.py` satır 361 — aynı cihaz durumunda görüntüler `os.link()` ile hard-link olarak kopyalanmaktadır. YOLO'nun label cache mekanizması (`*.cache` dosyaları) görüntü yolunu değil label yolunu kullanır. Hard-link'lenmiş görüntüler üzerinde varsayılan bir sorun yoktur. Ancak oversample döngüsünde ilk kopya hard-link, sonraki kopyalar ise `shutil.copy2` ile oluşturulabilecekse (satır 362–363 fallback) karışık inode yapısı oluşur; bu durum checksum bazlı cache invalidasyon mekanizmalarını yanıltabilir.

**Etkisi:** Performans / Veri Güvenliği — Düşük risk; ancak Ultralytics cache'in cache-miss oranı hard-link sayısına göre değişebilir.

**Olasılık:** Düşük.

**Tespit Yöntemi:** Cache build loglarındaki scan süresi ve uyarılar incelenir.

---

## 4. Eğitim Stabilitesi Analizi

### S-01 — Phase 2 `lrf` Örtük Mirası

**Risk Tanımı:** `train.py` / `train()` — `phase2_overrides` sözlüğü (satır 686–692) `lr0=phase2_lr0` ayarlamakta ancak `lrf` değerini geçersiz kılmamaktadır. `_train_single_phase()` içinde `optional_params` döngüsü `lrf=0.01` değerini TRAIN_CONFIG'ten almaktadır. Sonuç olarak phase 2'nin son LR'si `phase2_lr0 * lrf = 0.0001 * 0.01 = 0.000001` olmaktadır; bu, phase 1 son LR'sinin (`0.001 * 0.01 = 0.00001`) on katı aşağısındadır.

**Etkisi:** Eğitim Dinamikleri — Phase 2 sonunda öğrenme oranı son derece düşük bir platoya ulaşır; bu 15 epoch'luk ince ayar periyodu için gradient sinyalini bastırabilir. Ancak bu bilinçli bir seçim de olabilir.

**Olasılık:** Orta — Mevcut config ile bu davranış gerçekleşmektedir; kasıtlı olup olmadığı belgelenmemiştir.

**Tetikleyici Senaryo:** Her iki fazlı eğitim çalıştırmasında.

**Tespit Yöntemi:** Phase 2 `full_attempt_args.yaml` içindeki `lr0` ve `lrf` değerlerinin karşılaştırılması.

**Mimari Düzeyde İyileştirme Yönü:** Phase 2 override sözlüğünde `lrf` de açıkça belirtilmeli ya da `phase2_lrf` config anahtarı eklenerek belgeleme netleştirilmelidir.

---

### S-02 — `deterministic=False` ile `cudnn.benchmark=True` — Seed Sınırlaması

**Risk Tanımı:** `train.py` / `setup_seed()` (satır 489–526) — Varsayılan `deterministic=False` ile `cudnn.benchmark=True` aktif edilmektedir. Bu, CUDA kernel seçimini GPU ve tensor boyutuna göre optimize eder; ancak CUDA kernal otomatik seçimi non-deterministik olup seed ile kontrol edilemez. Ayrıca YOLO dataloader'ı çok sayıda `worker` ile çalıştığında (`deterministic=False`) her epoch'ta farklı augmentasyon sırası oluşabilir.

**Etkisi:** Tekrarlanabilirlik — Aynı seed ile iki farklı makinede veya iki farklı Colab oturumunda özdeş sonuç elde etmek mümkün olmayabilir.

**Olasılık:** Yüksek — Non-deterministik mod bilinçli tercih edilmiştir; bu nedenle "risk" değil kabul edilmiş tasarım kararıdır. Ancak yarışma raporlaması için tekrarlanabilirlik gerektiğinde sınırlılık oluşturur.

**Tespit Yöntemi:** Aynı config ile iki ayrı eğitimde epoch-başı loss değerlerinin karşılaştırılması.

---

### S-03 — Phase 2 Başlangıcında `kill_gpu_hogs()` Çağrısı Eksikliği

**Risk Tanımı:** `train.py` / `train()` — `kill_gpu_hogs()` yalnızca `train()` fonksiyonunun başında çağrılmaktadır. İki fazlı eğitimde phase 1 tamamlandıktan sonra `kill_gpu_hogs()` tekrar çağrılmamaktadır. Phase 1'den kalan CUDA allocator fragment'ları ve sync durumu, phase 2 başlangıcında OOM riskini artırabilir.

**Etkisi:** Kararlılık — Phase 2'de ilk OOM olasılığı artar; kurtarma zinciri devreye girse de batch küçülür ve performans beklenenden düşük olabilir.

**Olasılık:** Düşük–Orta — `expandable_segments:True` ile kısmen hafifletilmiştir.

**Tetikleyici Senaryo:** Yüksek VRAM kullanımıyla biten phase 1'den sonra phase 2 daha büyük `phase2_imgsz=896` ile başladığında.

---

### S-04 — Augmentasyon Val/Test Split'ine Sızma Riski

**Varsayım:** YOLO Ultralytics train argümanları incelendiğinde `mosaic`, `copy_paste`, `flipud`, `fliplr` gibi augmentasyonlar YOLO tarafından yalnızca train split'ine uygulandığı bilinmektedir. Bu davranış Ultralytics v8+ sürümlerinde standarttır.

**Risk Tanımı:** `build_dataset.py` içinde `oversample_count > 1` olduğunda kopyalama yalnızca `target_split == "train"` koşuluyla yapılmaktadır (satır 179). Val ve test split'leri oversampling almamaktadır. Bu doğru tasarımdır.

**Etkisi:** Mevcut yapıda val/test augmentation sızıntısı gözlemlenmemiştir.

**Tespit Yöntemi:** Ultralytics sürüm değişikliklerinde bu varsayımın korunup korunmadığı doğrulanmalıdır.

---

## 5. MLOps Olgunluk Değerlendirmesi

### M-01 — Git Commit Hash Loglama Eksikliği

**Risk Tanımı:** `train.py` / `_train_single_phase()` — `full_attempt_args.yaml` ve Ultralytics'in `args.yaml` dosyaları eğitim parametrelerini kaydetmektedir. Ancak hangi kod revizyonuyla eğitimin gerçekleştiğini gösteren git commit hash, dataset build timestamp veya audit_report snapshot hiçbir artefakta yazılmamaktadır.

**Etkisi:** İzlenebilirlik — Belirli bir modelin hangi kod ve veri versiyonu ile üretildiği sonradan belirlenemez.

**Olasılık:** Yüksek — Eğitim pipeline'ı aktif geliştirilme aşamasında olduğundan versiyonlar sık değişmektedir.

**Mimari Düzeyde İyileştirme Yönü:** Eğitim başlangıcında `git rev-parse HEAD` çıktısı ve `audit_report.json` checksum'u `full_attempt_args.yaml`'a eklenmeli veya ayrı bir `run_metadata.json` dosyasına kaydedilmelidir.

---

### M-02 — `TRAIN_CONFIG` Global Mutable Sözlük — Test İzolasyonu Riski

**Risk Tanımı:** `config.py` — `TRAIN_CONFIG` modül seviyesinde global sözlüktür; `ensure_colab_config()` bu sözlüğü yerinde mutate eder. `ARTIFACTS_DIR.mkdir()` modül import'unda çalışır. Unit test veya CI ortamında `config.py` import edilmesi disk üzerinde dizin oluşturmakta ve global state'i kirletmektedir.

**Etkisi:** MLOps Olgunluk / Tekrarlanabilirlik — Modül-seviyesi yan etkiler test izolasyonunu bozar; CI ortamında beklenmedik dizin oluşturma riski doğar.

**Olasılık:** Orta — `requirements-dev.txt` yalnızca `pytest` içermekte ve kapsamlı test altyapısı görünmemektedir; bu nedenle mevcut durumda aktif sorun değildir. Gelecekteki test genişlemesinde kritik hale gelir.

**Mimari Düzeyde İyileştirme Yönü:** Dizin oluşturma işlemleri explicit initialization fonksiyonuna taşınmalı; `TRAIN_CONFIG` değiştirilemez bir yapıya (dataclass veya Pydantic model) dönüştürülmeli ya da kopyalanarak geçersiz kılınmalıdır.

---

### M-03 — Checkpoint Resume Öncesi Epoch/Step Tutarlılığı Doğrulanmıyor

**Risk Tanımı:** `train.py` / `_resume_preflight_check()` — Preflight check yalnızca `dataset.yaml` ve klasör varlığını doğrulamaktadır. Checkpoint içindeki `epoch` değeri ile `--epochs` argümanı karşılaştırılmamaktadır. Yanlış epoch sayısıyla resume yapıldığında Ultralytics hedeflenen toplam epoch kadar daha eğitim yaparak fazladan eğitim gerçekleştirir.

**Etkisi:** MLOps — Fazla eğitim nedeniyle beklenmedik overfitting veya hesap maliyeti artışı.

**Olasılık:** Düşük — Kullanıcı epoch argümanını bilinçli olarak veriyorsa sorun oluşmaz.

**Tespit Yöntemi:** Resume sonrası `results.csv` satır sayısı ile hedef epoch karşılaştırması.

---

### M-04 — `requirements.txt` Lock Dosyası Eksikliği

**Risk Tanımı:** `requirements.txt` içinde `ultralytics==8.3.0` tam pin, ancak `tqdm>=4.60.0`, `pandas>=1.5.0`, `numpy>=1.24.0,<3.0.0`, `opencv-python-headless>=4.5.0` gibi geniş aralıklar kullanılmaktadır. `requirements-dev.txt` lock içermemektedir. Ortamlar arasında minor sürüm farklılıkları augmentasyon davranışı, CSV okuma veya görüntü decode farklılıklarına neden olabilir.

**Etkisi:** Tekrarlanabilirlik — Farklı Colab session veya yerel ortamlar arasında sonuç farklılığı.

**Olasılık:** Orta — `ultralytics==8.3.0` tam pinlenmiş olduğundan en kritik bağımlılık sabitlenmiştir. Yan bağımlılıklarda minor değişim riski düşük-orta düzeydedir.

**Mimari Düzeyde İyileştirme Yönü:** `pip freeze > requirements.lock` veya `pip-compile` ile kilit dosyası üretilmeli ve versiyona işlenmelidir.

---

### M-05 — Colab Bootstrap Hardcoded Repo URL

**Risk Tanımı:** `colab_bootstrap.py` satır 9 — `REPO_URL` kaynak dosyasında hardcoded olarak bulunmaktadır. Private repo'ya credential olmadan `git clone` başarısız olur; hata mesajı branch veya erişim sorununu açık biçimde bildirmez.

**Etkisi:** MLOps — Takım üyeleri veya fork kullananlar için bootstrap doğrudan çalışmaz.

**Olasılık:** Orta — Yarışma bağlamında tek repo kullanımında sorun değildir; takım genişlemesinde kritik olabilir.

---

## 6. Belirsizlikler ve Koşullu Riskler

### B-01 — Megaset Scene-ID Formatı Varsayımı

**Varsayım:** `build_dataset.py` satır 429–431 — Scene-based split, `_frame_` ayracı içeren dosya adlarına dayanmaktadır. Bu format megaset dokümantasyonunda belirtilmemiş; datasets.md'de yalnızca dolaylı olarak atıfta bulunulmaktadır.

**Koşullu Risk:** Megaset görüntülerinin bir kısmı `_frame_` içermiyorsa `scene_id = stem` olarak değerlendirilir; bu görüntülerin tamamı (tüm "kopyaları") aynı split'e düşer ve beklenenden daha az val görüntüsü oluşabilir.

**Doğrulama Adımı:** Megaset dosya adları içinde `_frame_` formatı bulunmayan örneklerin oranı kontrol edilmelidir.

---

### B-02 — `id_map` ile `source_names/map` Eş Zamanlı Kullanımı

**Varsayım:** `build_dataset.py` satır 281–292 — `id_map` varsa `source_names/map` atlanmaktadır. Yalnızca `megaset` `id_map` kullanmaktadır. Ancak `id_map = {0: 0, 1: 1}` tüm class ID'leri kapsamıyorsa (örn. kaynak datasette 3. class mevcutsa) bu class `unmapped_cls` sayacına eklenir, görsel uyarı basar fakat işlem durdurmaz.

**Doğrulama Adımı:** `id_map` tanımlı datasetlerin kaynak `data.yaml`'ındaki sınıf sayısıyla `id_map` anahtar sayısı karşılaştırılmalıdır.

---

### B-03 — Leakage Denetimi Yalnızca Kaynak Yapıya Bakıyor

**Varsayım:** `train.py` / `_check_leakage_from_audit()` — Leakage kontrolü `audit.py`'ın ürettiği raporu okumaktadır. Audit ise kaynak datasetlerin orijinal `train/val/test` klasör yapısını incelemektedir. `build_dataset.py`'ın megaset için uyguladığı scene-based re-split, kaynak yapıyı değiştirmez; dolayısıyla yeni oluşturulan `train/val` split'i audit tarafından denetlenmez.

**Koşullu Risk:** Megaset'in kaynak train ve val klasörleri arasında örtüşme olsa bile bu durum audit'te "INCLUDE" olarak geçilebilir çünkü audit yalnızca `datasets/TRAIN/megaset/train` ve `datasets/TRAIN/megaset/val` stemlerini karşılaştırmakta; `datasets/TRAIN/megaset/valid` klasörü de taranmaktadır (`"valid"` split için canonical = "val"). Scene-based split ise tüm train+valid+val görüntülerini `all_megaset_images`'e toplamaktadır; bu görüntülerden bazıları hem kaynak train hem kaynak val'de bulunabilir.

**Doğrulama Adımı:** `build_dataset.py` çalıştırıldıktan sonra üretilen dataset üzerinde ayrıca stem bazlı overlap kontrolü yapılmalıdır.

---

### B-04 — `colab_bootstrap.py` Satır 203–457 Görüntülenemedi

Dosyanın bu aralığı veri transferi nedeniyle kesilmiştir. İçerik incelenmemiştir.

**Varsayım:** Bu bölüm muhtemelen psutil/ultralytics varlık kontrolü, dataset çıkarma ve dataset hazırlama adımlarını içermektedir.

**Doğrulama Adımı:** Eksik bölüm okunarak dataset çıkarma (tar.gz), sembolik link kurulumu ve drive bağlantısı doğrulanmalıdır.

---

## 7. Genel Sağlık Skoru

**Puan: 6.5 / 10**

**Gerekçe:**

- **Güçlü yönler:** Leakage denetim altyapısı (audit + train entegrasyonu) iyi tasarlanmış; OOM kurtarma zinciri çok adımlı ve düşünceli; checkpoint bütünlük kontrolü zipfile ile yapılıyor; Drive sync güvenli tmp→rename pattern kullanıyor; megaset scene-aware split data leakage'ı pratikte önlüyor; TF32 ve AMP konfigürasyonu doğru.

- **Zayıf yönler:** Non-megaset örnekleme non-deterministik; `dataset.yaml` göreli yol riski; `nbs` OOM kurtarma tutarsızlığı; `is_sample` audit tespiti güvenilmez; git commit hash izlenebilirliği yok; lock dosyası eksikliği; phase 2 `lrf` örtük mirasının belgelenmemiş olması.
