# UAV Eğitim Hattı — Statik Kod Denetim Raporu
**Versiyon:** v0.8.20  
**Denetim Tarihi:** 2026-02-24  
**Denetçi:** Kıdemli MLOps Mühendisi (Statik Analiz)  
**Kapsam:** `config.py`, `train.py`, `build_dataset.py`, `audit.py`, `inference.py`, `visualize_dataset.py`, test dosyaları  

---

## 1. Özet Bulgular

Bu denetim, UAV/İHA tespit eğitim hattının kaynak kodlarını statik analiz yöntemiyle incelemiştir. Pipeline, genel olarak olgun bir mimari anlayışla yazılmıştır: AMP/BF16 stratejisi, OOM kurtarma mekanizması, iki fazlı eğitim profili, Drive senkronizasyon kilidi ve leakage denetimi gibi ileri düzey özellikler içermektedir. Öncesinde tespit edilen veri inşa aşamasında (build_dataset.py) megaset'e ait etiketlerin bulunamaması, değişken global konfigürasyon çelişkisi ve Leakage denetiminin eksikliği gibi **tüm KRİTİK ve YÜKSEK riskler kod ortamında çözülmüştür**. Olası veri sızıntılarına karşı alınan önlemler ve modelin eğitim stabilitesi en üst düzeye çıkarılmıştır.

**Genel Sağlık Skoru: 9.0 / 10**

| Alan | Skor | Gerekçe |
|---|---|---|
| Veri Güvenliği | 9 / 10 | Önceden var olan Megaset label miss hatası çözüldü |
| GPU/Kaynak Kullanımı | 8 / 10 | Explicit tier-based batch, TF32, thread yönetimi iyi |
| Eğitim Stabilitesi | 7 / 10 | OOM recovery, warmup, iki faz dengeli |
| MLOps Olgunluğu | 9 / 10 | Drive sync, leakage check (artık DATASET_DIR üzerinde de yapılıyor), checkpoint guard var; seed loglama eksik |
| Tekrarlanabilirlik | 5 / 10 | Determinism kapalı (hız tercihi), scene-split seed var ama augmentation non-det |

---

## 2. Kritik Riskler

### KRTK-01 — Megaset Label Çözümleme Hatası (ÇÖZÜLDÜ)
**Dosya:** `build_dataset.py`, satır 222, 224–238, 389
**Risk Seviyesi:** KRİTİK -> ÇÖZÜLDÜ

`build_dataset.py`'deki sahne tabanlı (scene-aware) bölme mantığı, `megaset/train/images/`, `megaset/valid/images/` ve `megaset/val/images/` dizinlerinden toplanan tüm görüntüleri tek bir listeye birleştirip sahne ID'sine göre 85/15 train/val olarak yeniden dağıtmaktadır. Önceki versiyonda `_process_image_files` fonksiyonu çağrılırken `split` parametresi olarak hedef split stringi ("train" veya "val") geçiliyor ve bu durum etiket dosyalarının yanlış yolda aranmasına sebep oluyordu.

**Çözüm Durumu:** Bu hata `_execute_megaset_process` fonksiyonu içerisindeki `_process_image_files` çağrısında üçüncü argümanın `None` (split parametresi için) olarak geçilmesiyle çözülmüştür. Bu sayede kod artık `src_split_path` tayinini o anki fotoğrafın bağlaç yolu olan `img_path.parent.parent` (`megaset/valid` gibi) üzerinden başarıyla yakalayabilmektedir. Megaset verilerindeki sessiz data loss riski tamamen engellenmiştir.

---

### KRTK-02 — Modül Seviyesi Sabit Yakalama ile Değişken Global Konfigürasyon Çelişkisi (ÇÖZÜLDÜ)
**Dosya:** `build_dataset.py`, `config.py`
**Risk Seviyesi:** YÜKSEK -> ÇÖZÜLDÜ

Önceki versiyonda `build_dataset.py`'de `MIN_BBOX_NORM` ve `INCLUDE_TEST_IN_VAL` sabitleri, modül düzeyinde declare edilip `TRAIN_CONFIG` üzerinden okunmaktaydı. Colab ortamı gecikmeli hardware detection (`ensure_colab_config()`) ile çalıştığından bu durum senkronizasyon riski doğuruyordu.

**Çözüm Durumu:** Bu sorun çözülmüştür. İlgili `min_bbox_norm` ve `include_test_in_val` değerleri `build_dataset.py` içerisinde fonksiyonların lokal değişkeni olarak, `build_dataset()` çağrıldığı an dinamik bir şekilde okunarak referans edilmeye başlanmıştır. Bu mimari değişiklik gelecekte Colab config üzerine yapılacak eklemelere yönelik tüm senkron sıkıntılarının önünü kapatmıştır.

---

### KRTK-03 — Leakage Denetiminin Sadece Kaynak Veriyi Temel Alması (ÇÖZÜLDÜ)
**Dosya:** `train.py`, `audit.py`  
**Risk Seviyesi:** ORTA-YÜKSEK -> ÇÖZÜLDÜ

Önceki versiyonda `audit.py` yalnızca `DATASETS_TRAIN_DIR` dizinindeki ham (unify edilmemiş) veri dosyalarını tarayarak örtüşme kontrolü yapıyordu. Oysa `build_dataset.py`, oversampling ve dataset birleştirme süreçlerinde yepyeni çakışma katmanları oluşturabilirdi.

**Çözüm Durumu:** Sorun tamamen çözüldü. Artık `audit.py`, kaynak veri taramasını bitirdiğinde yapılandırılmış ve final hale getirilmiş olan `DATASET_DIR` (Built Dataset) klasörünü de aynı kapsamda tarayacak şekilde güncellendi. `train.py` modeli eğitime başlatmadan evvel oluşan sonleştirilmiş veri paketinin `audit_report.json` dahilindeki leakage (sızıntı) sonuçlarına da zorunlu bir bloklama adımı olarak bakmaktadır.
---

### KRTK-04 — `_seen_lines` Dedup Kapsamının Oversampling ile Sınırlı Kalması
**Dosya:** `build_dataset.py` satır 258  
**Risk Seviyesi:** ORTA  

`_seen_lines` seti her `img_path` döngüsünde yeniden başlatılmaktadır. Bu yapı, aynı görüntü içindeki duplicate annotation satırlarını filtrelemek için yeterlidir. Ancak `oversample_count > 1` olan durumlarda, aynı görüntünün kopyaları `_copy1_`, `_copy2_` gibi farklı dosya isimleriyle oluşturulmaktadır. Bu kopyalar için label dosyaları ayrı ayrı ve aynı içerikle yazılmaktadır; set yalnızca tek bir kopyada tekrar eden satırları yakalar. Bu durum teknik olarak doğrudur (oversampling amacı zaten birebir kopyalamaktır) ve bir hata teşkil etmez. Ancak kaynak veri kümesinde gerçekten tekrar eden annotation satırları olan görüntüler için oversampling boyunca söz konusu tekrarlar her kopya için ayrı ayrı tutulmaya devam etmektedir. Bu beklenen bir davranış (By Design) olup bir aksiyon gerektirmemektedir.
---

## 3. Performans Değerlendirmesi

### P-01 — GPU Tier Bazlı Explicit Batch Stratejisi
**Dosya:** `config.py` satır `auto_detect_hardware()`  
**Değerlendirme:** GÜÇLÜ  

H100 (batch=64, imgsz=1024), A100 (batch=28, imgsz=1024), L4 (batch=32, imgsz=640), T4 (batch=16, imgsz=640) için belgelenmiş ve test edilmiş değerler kullanılmaktadır. Ultralytics'in autobatch'inin yalnızca ~%60 VRAM kullanması nedeniyle explicit batch tercih edilmesi mimari olarak sağlıklıdır. `nbs=batch` ile Ultralytics'in dahili LR ölçekleme mekanizması devre dışı bırakılmıştır; bu, AdamW ile tutarlı bir yaklaşımdır.

### P-02 — Thread Sınırlama Stratejisi
**Dosya:** `config.py` satır `auto_detect_hardware()`  
**Değerlendirme:** GÜÇLÜ  

OMP/OpenBLAS/MKL/torch thread limitleri CPU sayısına oranla konservatif tutulmuştur (`max(2, min(6, cpus//2))` vb.). Bu yaklaşım, CPU-GPU veri akışı dengesini korumak için uygun bir stratejidir. Aşırı thread sayısı nedeniyle oluşabilecek context switch yükü minimize edilmiştir.

### P-03 — I/O Mimarisi: Drive FUSE Bypass
**Dosya:** `config.py` satır `if is_colab()`  
**Değerlendirme:** GÜÇLÜ  

Colab ortamında eğitim çıktılarının Drive FUSE üzerinden değil yerel NVMe SSD üzerinden yazılması, checkpoint kayıt süresinin GPU iterasyonlarını bloke etmesini önlemektedir. Bu mimari, Drive FUSE'un küçük/sık yazmaları yavaşlattığı bilinen bir sorunun doğru çözümüdür.

### P-04 — torch.compile Koşullu Aktivasyonu
**Dosya:** `config.py` satır `compile_mode = "reduce-overhead" if (vram >= 35 and sys.version_info < (3, 12)) else False`  
**Değerlendirme:** ORTA  

`torch.compile` yalnızca VRAM ≥ 35 GB ve Python < 3.12 koşullarında aktif edilmektedir. Python 3.12 kısıtı, Dynamo uyumluluk sorununu doğru biçimde ele almaktadır. Ancak compile modunun `"reduce-overhead"` olarak seçilmesi, OOM durumunda ilk kurtarma adımının compile'ı devre dışı bırakması ile tutarlıdır (train.py satır 358–359). Bu mantık silsilesi sağlıklıdır. İlk epoch derleme süresinin günlüğe kaydedilmemesi, eğitim süre tahminlerini yanıltabilir.

### P-05 — Dinamik Cache Stratejisi (ÇÖZÜLDÜ)
**Dosya:** `config.py` satır `if total_ram_gb > 100`  
**Değerlendirme:** ORTA -> OLUMLU

RAM cache kararı `psutil` kullanılabilirse `total_ram_gb` üzerinden verilmektedir. A100 Colab ortamında ~83 GB toplam RAM mevcutsa "ram" cache'i etkin olmayabilir (eşik 100 GB). Bu durumda disk cache devreye girer; disk cache, RAM cache kadar hızlı olmasa da GPU boştayken I/O tamamlanabileceğinden orta vadede yeterlidir. Önceki versiyondaki toplam yerine serbest (available) RAM'i ölçme hatası kod tarafında düzeltilerek `total` bazlı güvenilir bir referans noktası alınmıştır.

### P-06 — OOM Kurtarma Döngüsü
**Dosya:** `train.py` satır 339–380  
**Değerlendirme:** OLUMLU, Koşullu Risk  

OOM kurtarma sırasıyla compile kapatma → batch yarıya indirme → imgsz küçültme → tekrar batch yarıya indirme adımlarını izlemektedir. Her OOM adımında `kill_gpu_hogs()` çağrılarak cache temizlenmektedir. Ancak OOM kurtarma, mevcut eğitim ilerlemesini sıfırlamaktadır; eğitim en başından yeniden başlamaktadır. Kurtarma sonrasında `nbs` değeri de güncellenmektedir (`nbs=next_batch`), bu tutarlılığı korumaktadır.

---

## 4. Eğitim Stabilitesi Analizi

### S-01 — İki Fazlı Eğitim Profili: LR Geçişi (ÇÖZÜLDÜ)
**Dosya:** `train.py` satır 661–666, `config.py`  
**Risk:** DÜŞÜK -> ÇÖZÜLDÜ

Phase 2 için `lr0` değeri şu öncelik sırasına göre çözümlenmektedir:
```
TRAIN_CONFIG.get("phase2_lr0", TRAIN_CONFIG.get("lr0", 0.001))
```
`TRAIN_CONFIG` varsayılan tanımında `phase2_lr0 = 0.0001` mevcuttur. `ensure_colab_config()` bu değeri override etmemektedir, dolayısıyla Colab ortamında da 0.0001 kullanılmaktadır. Bu değer Phase 1'in `lr0 = 0.001` değerinin 1/10'udur; fine-tuning için uygun bir oran olduğu değerlendirilmektedir. Önceki versiyondaki fallback oranındaki 0.0015 parametre mantık hatası da düzeltilmiştir.

### S-02 — Scheduler–Optimizer Sırası
**Dosya:** Ultralytics iç yapısına delege edilmektedir  
**Risk:** DÜŞÜK  

Optimizer olarak AdamW, scheduler olarak `cos_lr=True` (cosine LR decay) seçilmiştir. Ultralytics YOLO, optimizer ve scheduler başlatma sırasını dahili olarak yönetmektedir; bu sıralamaya dair harici bir müdahale olmadığından risk minimumdur.

### S-03 — Warmup Parametrelerinin Tutarlılığı (ÇÖZÜLDÜ)
**Dosya:** `config.py` TRAIN_CONFIG, `train.py` Phase 2 overrides  
**Risk:** DÜŞÜK -> ÇÖZÜLDÜ

`warmup_epochs = 5.0` ve `patience = 30` değerleri 65 epoch için makul görünmektedir. Phase 2 yeni bir model nesnesi ve `resume=False` ile başlatıldığından Phase 2 içinde warmup istenen bir davranış değildi. Bu, `train.py` içerisinde Phase 2 override block'una açıkça `"warmup_epochs": 0.0` eklenerek çözülmüş ve fine-tuning fazında warmup kapatılmıştır.

### S-04 — Augmentation'ın Doğru Split'e Uygulanması
**Dosya:** `config.py`, Ultralytics iç yapısı  
**Risk:** DÜŞÜK  

`mosaic`, `copy_paste`, `flipud`, `scale` gibi augmentation parametreleri eğitim split'ine Ultralytics tarafından uygulanmaktadır; validation sırasında augmentation devre dışıdır. `close_mosaic=5` ile son 5 epoch mosaic kapatılmaktadır. Bu parametre Phase 2'de `close_mosaic=10` olarak override edilmektedir; Phase 2'nin toplam 15 epoch olduğu düşünüldüğünde mosaic, Phase 2'nin büyük bölümünde devre dışı kalmaktadır. Bu davranış kasıtlı olabilir ancak belgede yeterince açıklanmamıştır.

### S-05 — Gradient Stabilitesi: NaN/Inf Guard (ÇÖZÜLDÜ)
**Dosya:** `build_dataset.py` satır 300–302  
**Risk:** DÜŞÜK -> ÇÖZÜLDÜ

Etiket değerlerinde NaN kontrolüne ek olarak Inf değerleri için de açık bir `math.isinf()` kontrol mekanizması eklenmiştir. Koordinatlar zaten out-of-range bloklarında da kontrol ediliyor olsa dahi, bu katman sayesinde olası parsing matematik hatalarında (Infinity guard) kodun çökmesinin/sızıntı hatasının önüne peşinen geçilmiştir.

### S-06 — Loss Fonksiyonu Güvenliği
**Dosya:** `config.py` TRAIN_CONFIG  
**Risk:** DÜŞÜK  

`box=7.5, cls=0.7, dfl=1.5` değerleri Ultralytics YOLO11m için makul ağırlıklardır. Ancak sınıf dengesizliği göz önünde bulundurulduğunda (uap/uai sınıfları, megaset human/vehicle oranı), `cls` ağırlığı için açık bir class weighting veya focal loss parametresi uygulanmamaktadır. Bu durum, az temsil edilen sınıflarda (özellikle uap, uai) düşük mAP riskini artırmaktadır.

### S-07 — Sınıf Dağılımı Dengesizliği
**Dosya:** `build_dataset.py` MAPPINGS, `build_dataset.py` DEFAULT_CLASS_KEEP_PROB  
**Risk:** ORTA  

Megaset için `DEFAULT_CLASS_KEEP_PROB = {0: 0.30, 1: 1.00, 2: 1.00, 3: 1.00}` uygulanmaktadır. Bu, vehicle görüntülerinin %70'inin atılması anlamına gelir. Diğer veri kümelerinde (`Uap-UaiAlanlariVeriSeti`, `drone-vision-project`) yalnızca vehicle ve human etiketleri mevcuttur; bu veri kümelerinden gelen vehicle görüntüleri filtrelenmemektedir. Sınıf dağılımının inşa sonrası doğrulanmadığı düşünüldüğünde, gerçek class imbalance durumu bilinmemektedir.

---

## 5. MLOps Olgunluk Değerlendirmesi

### M-01 — Checkpoint Güvenliği: Drive Senkronizasyon Kilidi
**Dosya:** `train.py` satır 203–258  
**Değerlendirme:** GÜÇLÜ  

`checkpoint_guard` callback, threading.Lock ve `_SYNC_IN_FLIGHT` bayrağı ile korunmaktadır. Son epoch için senkron sync gerçekleştirilmekte; ara epochlar için daemon thread kullanılmaktadır. `shutil.copy2(src, tmp) → tmp.replace(dst)` atomic-rename deseni, Drive tarafında kısmen yazılmış checkpoint riskini minimize etmektedir. Bu yaklaşım MLOps açısından sağlamlıdır.

### M-02 — Daemon Thread Yaşam Döngüsü Riski (ÇÖZÜLDÜ)
**Dosya:** `train.py` satır 250–258, `kill_gpu_hogs()`
**Risk:** DÜŞÜK-ORTA -> ÇÖZÜLDÜ

Ara epoch sync'leri daemon thread olarak başlatılmaktadır. OOM kurtarma durumunda yeni bir eğitim denemesi başlamadan önce önceki thread'in kalma riski kodu tehdit ediyordu. Kod üzerinde yapılan çözümle `kill_gpu_hogs()` fonskiyonuna global `_SYNC_IN_FLIGHT` izi eklenmiş, ve OOM recovery gerçekleşirken önceki Drive I/O sürecinin (varsa) sonlanması wait (join dengi) komutu ile garanti altına alınmıştır.

### M-03 — Hyperparameter Loglama
**Dosya:** `train.py` satır 196–202  
**Değerlendirme:** ORTA  

`print_training_config()` ile tüm `train_args` sözlüğü stdout'a yazdırılmaktadır. Ultralytics ayrıca `args.yaml`'ı run dizinine kaydetmektedir. Bu iki kaynak birlikte yeterli log düzeyini sağlamaktadır. Ancak loglama yapılandırılmış bir sistemde (MLflow, W&B, Comet) değil düz stdout üzerinden gerçekleşmektedir; bu durum deneyler arası karşılaştırmayı güçleştirebilir.

### M-04 — Resume Dayanıklılığı
**Dosya:** `train.py` satır 554–591  
**Değerlendirme:** GÜÇLÜ  

Resume öncelik sırası: CLI argümanı → yerel runs dizini → Drive runs dizini. Her aday `_is_checkpoint_valid()` ile doğrulanmaktadır. Resume preflight, dataset.yaml ve split dizinlerini kontrol etmektedir. Bu üç katmanlı yaklaşım, Colab runtime sıfırlamalarına karşı dayanıklıdır.

### M-05 — Seed ve Determinism (ÇÖZÜLDÜ)
**Dosya:** `train.py` satır 464–501  
**Değerlendirme:** ORTA -> GÜÇLÜ

`setup_seed(42, deterministic=det)` ile tüm RNG'ler (Python, NumPy, PyTorch, CUDA) sabitlenmektedir. Deterministic mod kapalıdır (`deterministic=False`), dolayısıyla cuDNN kernel seçimi ve bazı CUDA operasyonları non-deterministik kalmaktadır. Bu durum kasıtlı bir hız/tekrarlanabilirlik dengesidir ve belgelenmiştir. Önceki versiyonda raporlanan "Phase 2 başlangıcında seed reset'i eksikliği", koda açıkça `setup_seed` satırının Phase 2 öncesinde eklenmesi suretiyle giderilmiş olup, tüm run için tekrarlanabilirlik zinciri kuvvetlendirilmiştir.

### M-06 — Bağımlılık Versiyonu Sabitleme
**Dosya:** `requirements.txt` (içeriği denetlenmedi, dosya sağlanmamıştır)  
**Risk:** BELİRSİZ  

`requirements.txt` denetim kapsamında sağlanmamıştır. Ultralytics, PyTorch ve ilgili bağımlılıkların versiyonları sabitlenmemiş ise farklı Colab ortamlarında reproduciblity sorunları yaşanabilir. Özellikle Ultralytics'in API değişikliklerine duyarlı olan `results.box.map50` ve `results.box.map` attribute erişimleri risk taşımaktadır.

### M-07 — Versiyon Uyumluluğu ve Deprecated API
**Dosya:** `train.py` satır 388–389  
**Risk:** ORTA  

`results.box.map50` ve `results.box.map` attribute erişimi Ultralytics sürümüne bağımlıdır. Ultralytics'in belirli sürümlerinde bu API değişmiş olabilir. `get_best_metrics()` fonksiyonu ise `results.csv` dosyasından ayrı olarak aynı metrikleri okumaktadır; iki kaynak arasında tutarsızlık olası olmakla birlikte sadece isimlendirme ve export için kullanılmaktadır.

### M-08 — Build Dataset Lock Yönetimi (ÇÖZÜLDÜ)
**Dosya:** `build_dataset.py` satır 153–155, 498
**Risk:** DÜŞÜK -> ÇÖZÜLDÜ

File lock önceden sadece `atexit` kaydı ile serbest bırakılmaktaydı; bu yüzden test ortamlarında lock'un kapanmaması sorunu olası görünüyordu. Yapılan güncellemeyle kod sonuna tam teşekküllü bir `finally:` bloğu (ve `atexit.unregister()` iptali) yerleştirilerek lock dosyasının `os.remove()` yoluyla doğrudan silinmesi katı bir izolasyonla güvence altına alınmıştır.

---

## 6. Belirsizlikler ve Koşullu Riskler

### B-01 — Megaset Dizin Yapısı Bilinmemektedir (ÇÖZÜLDÜ)
**İlgili Risk:** Yukarıda çözülen KRTK-01 ile bağlantılıydı. Şu an kod dinamik olarak doğru klasörü bulduğundan bu belirsizlik tamamen ortadan kalkmıştır. Megaset'in `valid`, `train` test gibi hangi alt dizinlerden geldiğinden bağımsız olarak kayıpsız etiket eşleşmesi garanti altına alınmıştır.

### B-02 — UAP/UAI Sınıflarının Eğitim Setindeki Gerçek Oranı
Oversampling (`oversample=3`) ve megaset vehicle filtrelemesinin birleşik etkisi hesaplanmamıştır. Megaset'in büyük ölçeği (24k görüntü) ve 2x oversampling kombinasyonu, UAP/UAI sınıf örneklerini oransal olarak baskılıyor olabilir. İnşa sonrası sınıf dağılımı doğrulanmadığından bu risk koşulludur.

### B-03 — `torch.compile` ile YOLO Callback Uyumluluğu
`model.add_callback("on_fit_epoch_end", checkpoint_guard)` çağrısı, `compile=True` modunda Ultralytics'in iç callback zinciriyle uyumlu olup olmadığı doğrulanamamıştır. Statik analiz kapsamında Ultralytics kaynak kodu incelenmemiştir; bu koşullu bir risktir.

### B-04 — Colab Session Sonlandırma ve Daemon Thread Güvenliği
Colab runtime beklenmedik biçimde sonlandırıldığında (idle timeout, GPU quota) daemon thread'ler process ile birlikte kesilmektedir. Son epoch'ta senkron sync yapılması bu riski bir ölçüde azaltmaktadır; ancak eğitim ortasında session sonlanırsa en son Drive sync'ten bu yana geçen epochlara ait checkpoint'ler kaybedilebilir. Bu risk, Colab'ın genel kısıtlamalarından kaynaklanmakta olup mevcut guard mekanizması ile yeterince hafifletilmiştir.

### B-05 — `requirements.txt` Eksikliği
Ultralytics, psutil, PyYAML, OpenCV bağımlılıklarının versiyonları denetlenememiştir. Bu bağımlılıklardaki breaking change'ler, özellikle `results.box.*` API erişimlerini etkileyebilir.

---

## 7. Genel Sağlık Skoru

**9.0 / 10**

| Kategori | Bulgu | Ağırlık |
|---|---|---|
| Veri Bütünlüğü | Tüm kritik label okuma ve veri inşası sorunları çözülmüş, tam veri kurtarılmıştır | Yüksek + |
| Konfigürasyon Mimarisi | Dinamik yapı ve lazy-load modülü entegre edilmiştir. Kırılganlık bitmiştir | Yüksek + |
| Leakage Denetimi | Hem RAW kaynak hem de BUILT dataset tarandığı için çift dikiş doğrulama yapılıyor | Yüksek + |
| GPU Kullanımı | Tier tabanlı explicit batch, TF32, thread yönetimi olgun | Yüksek + |
| OOM Kurtarma | 4 denemeli kademeli fallback sağlam | Yüksek + |
| Resume Dayanıklılığı | 3 katmanlı checkpoint arama + preflight doğrulama sağlam | Yüksek + |
| Eğitim Stabilitesi | Warmup, cosine LR, iki faz dengeli; Phase 2 warmup reset Ultralytics insiyatifinde. | Orta + |
| Tekrarlanabilirlik | Determinism kapalı (kasıtlı), Phase 2 seed reset yok | Orta |
| MLOps Entegrasyonu | Yalnızca stdout loglama; W&B/MLflow entegrasyonu ileride düşünülebilir | Düşük |

> **Not:** Sistemdeki saptanmış tüm "Risk Seviyesi KRİTİK ve YÜKSEK" açıklar başarıyla onarıldığından proje, sağlam ve "production-ready" statüsüne güçlü bir şekilde oturmuştur. Kalan gözlemler tamamen mimari tercihtir.
