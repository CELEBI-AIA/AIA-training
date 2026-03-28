# Dataset Reference

## Archive & Path Layout

**TRAIN_DATA_COMBINED.tar.gz** (`MyDrive/AIA/datasets/TRAIN_DATA_COMBINED.tar.gz`): Tüm datasetleri `TRAIN_DATA/` wrapper altında içerir.

```
TRAIN_DATA_COMBINED.tar.gz
└── TRAIN_DATA/
    ├── UAI_UAP/
    ├── drone-vision-project/
    ├── megaset/
    ├── teknofest_01/
    ├── teknofest_02/
    ├── teknofest_04/
    ├── teknofest_05/
    ├── teknofest_06/
    ├── teknofest_08/
    ├── teknofest_09/
    ├── teknofest_10/
    ├── teknofest_11/
    ├── teknofest_12/
    ├── teknofest_14/
    ├── teknofest_15/
    ├── teknofest_16/
    └── teknofest_17/
```

- **Colab:** Tar extract edilir → `/content/datasets_local/TRAIN_DATA/{UAI_UAP, drone-vision-project, megaset, teknofest_XX}`. `repo/datasets` symlink ile `/content/datasets_local`'e bağlanır. `DATASETS_TRAIN_DIR` → `datasets/TRAIN_DATA/`.
- **Local:** `datasets/TRAIN_DATA/` altında tüm klasörler olmalı. Alternatif olarak `datasets/` altına doğrudan da çıkartılabilir (direct-root layout).

---

## Evrensel Sınıf Şeması

Tüm kaynak datasetlerdeki sınıflar `build_dataset.py` tarafından 4 evrensel hedefe dönüştürülür:

| Target ID | Target Name | Açıklama | Eşlenen Kaynak Sınıflar |
|:---------:|-------------|----------|--------------------------|
| `0` | **vehicle** | Her türlü kara taşıtı | vehicle, car, tasit, arac, araba, Vehicle |
| `1` | **human** | İnsan | pedestrian, people, person, human, insan, Person, Human |
| `2` | **uap** | UAP (Unmanned Aerial Platform) | UAP, UAP-, uap, 2_uap |
| `3` | **uai** | UAI (Unmanned Aerial Integration area) | UAI, UAI-, uai, 3_uai, ambulans |

Canonical mapping: `uav_training/config.py` (`TARGET_CLASSES`) ve `uav_training/build_dataset.py` (`MAPPINGS`).

Çıktı `dataset.yaml`:

```yaml
nc: 4
names:
  0: vehicle
  1: human
  2: uap
  3: uai
```

### UAI / UAP Semantiği

- **UAI** — Unmanned Aerial Vehicle Integration area, iniş için uygun alan.
- **UAI-** — Aynı alan tipi, iniş için **uygun olmayan**.
- **UAP** — Unmanned Aerial Vehicle Platform, iniş için uygun platform.
- **UAP-** — Aynı alan tipi, iniş için **uygun olmayan**.

Build sırasında uygun/uygun olmayan varyantlar aynı hedef sınıfa **birleştirilir** — amaç konumları tespit etmektir.

---

## Orijinal Datasetler (TRAIN_DATA.tar.gz)

### 1. UAI_UAP

| Orijinal İsim | → Hedef ID | Hedef Sınıf |
|---------------|:----------:|-------------|
| UAI           | **3**      | uai         |
| UAP           | **2**      | uap         |

UAI/UAP iniş alanı dataseti. `oversample: 5`.

### 2. drone-vision-project

| Orijinal İsim | → Hedef ID | Hedef Sınıf |
|---------------|:----------:|-------------|
| car           | **0**      | vehicle     |
| pedestrian    | **1**      | human       |

Aerial drone footage — araç ve yaya sınıfları. `oversample: 2`.

### 3. megaset

| Orijinal İsim | → Hedef ID | Hedef Sınıf |
|---------------|:----------:|-------------|
| vehicle       | **0**      | vehicle     |
| pedestrian    | **1**      | human       |

Büyük ölçekli dataset (~24k görüntü). Smart sampling: %100 insan, %5 sadece-araç görüntüleri. `oversample: 3`, `human_extra_oversample: 3`.

---

## Teknofest Datasetleri (uaiuapdataset.tar.gz)

> **Okuma kılavuzu:** `orijinal_id (orijinal_isim) → yeni_id` formatındadır.
> Label dosyalarındaki sınıf numarası bu tablolara göre dönüştürülür.

### teknofest_01
**Orijinal:** `['0', '1', '2', '3']` · **Kaynak:** berk / teknofest-tvmlr / v1

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | 0 | → | **0** | vehicle |
| 1 | 1 | → | **1** | human |
| 2 | 2 | → | **2** | uap |
| 3 | 3 | → | **3** | uai |

---

### teknofest_02
**Orijinal:** `['UAI', 'UAP', 'person', 'vehicle']` · **Kaynak:** workspace-71uin / teknofest-layzj / v12

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | UAI | → | **3** | uai |
| 1 | UAP | → | **2** | uap |
| 2 | person | → | **1** | human |
| 3 | vehicle | → | **0** | vehicle |

---

### ~~teknofest_03~~
> ❌ **Duplicate temizliği sonrası tamamen boşaldı, pakete dahil edilmedi.**
> teknofest_02 ile aynı kaynaktan geliyordu.

---

### teknofest_04
**Orijinal:** `['uai', 'uap']` · **Kaynak:** hyz-moa5c / uap-uai-demo / v1

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | uai | → | **3** | uai |
| 1 | uap | → | **2** | uap |

> ℹ️ Duplicate temizliği sonrası yalnızca train/187 görüntü kaldı.

---

### teknofest_05
**Orijinal:** `['0', '1', '2', '3']` · **Kaynak:** teknofest-gpscr / teknofest-ejiwb / v1

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | 0 | → | **1** | human |
| 1 | 1 | → | **0** | vehicle |
| 2 | 2 | → | **3** | uai |
| 3 | 3 | → | **2** | uap |

---

### teknofest_06
**Orijinal:** `['UAI', 'UAP', 'insan', 'tasit']` · **Kaynak:** teknofest-m874p / teknofest-jfohi / v19

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | UAI | → | **3** | uai |
| 1 | UAP | → | **2** | uap |
| 2 | insan | → | **1** | human |
| 3 | tasit | → | **0** | vehicle |

---

### ~~teknofest_07~~
> ❌ **Duplicate temizliği sonrası tamamen boşaldı, pakete dahil edilmedi.**

---

### teknofest_08
**Orijinal:** `['Person', 'Vehicle']` · **Kaynak:** nez-6dv1g / teknofest-vmp4c / v2

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | Person | → | **1** | human |
| 1 | Vehicle | → | **0** | vehicle |

---

### teknofest_09
**Orijinal:** `['0', '1', '2', '3']` · **Kaynak:** teknofest-tcg1m / teknofest-apxmr / v3

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | 0 | → | **0** | vehicle |
| 1 | 1 | → | **1** | human |
| 2 | 2 | → | **3** | uai |
| 3 | 3 | → | **2** | uap |

---

### teknofest_10
**Orijinal:** `['0', '1', '2_uap', '3_uai']` · **Kaynak:** cars-wiqof / uap-uai-m8ass / v1

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | 0 | → | **0** | vehicle |
| 1 | 1 | → | **1** | human |
| 2 | 2_uap | → | **2** | uap |
| 3 | 3_uai | → | **3** | uai |

---

### teknofest_11
**Orijinal:** `['person', 'uai', 'uap']` · **Kaynak:** cay-xglwe / teknofest-dde1g-lacm6 / v1

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | person | → | **1** | human |
| 1 | uai | → | **3** | uai |
| 2 | uap | → | **2** | uap |

---

### teknofest_12
**Orijinal:** `['insan', 'tasit', 'uai', 'uap']` · **Kaynak:** yarma / teknofest-veri-2024-antalya / v1

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | insan | → | **1** | human |
| 1 | tasit | → | **0** | vehicle |
| 2 | uai | → | **3** | uai |
| 3 | uap | → | **2** | uap |

---

### ~~teknofest_13~~
> ❌ **Bu dataset silindi.** 9 sınıflı karmaşık yapı, standart eşlemeyle uyumsuz.

---

### teknofest_14
**Orijinal:** `['ambulans']` · **Kaynak:** hamza-vvz7g / wg-4lmct / v1

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | ambulans | → | **3** | uai |

> ⚠️ Ambulans etiketi UAI olarak yorumlanmıştır. Sadece 54 train görüntüsü.

---

### teknofest_15
**Orijinal:** `['car', 'insan', 'uai', 'uap']` · **Kaynak:** denem5 / uap-uai-iutyw / v6

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | car | → | **0** | vehicle |
| 1 | insan | → | **1** | human |
| 2 | uai | → | **3** | uai |
| 3 | uap | → | **2** | uap |

---

### teknofest_16
**Orijinal:** `['Human', 'UAI', 'UAP', 'Vehicle']` · **Kaynak:** tulpar-etiketleme-2 / tulpar-uap-uai / v2

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | Human | → | **1** | human |
| 1 | UAI | → | **3** | uai |
| 2 | UAP | → | **2** | uap |
| 3 | Vehicle | → | **0** | vehicle |

---

### teknofest_17
**Orijinal:** `{0: uap, 1: uai}` · **Kaynak:** yerel dataset (uaiuap)

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | uap | → | **2** | uap |
| 1 | uai | → | **3** | uai |

---

## Hızlı Başvuru — Dönüşüm Tablosu

| Dataset | nc | 0→ | 1→ | 2→ | 3→ |
|---------|:--:|:--:|:--:|:--:|:--:|
| UAI_UAP | 2 | — | — | — | — |
| drone-vision-project | 2 | 0 | 1 | — | — |
| megaset | 2 | 0 | 1 | — | — |
| teknofest_01 | 4 | 0 | 1 | 2 | 3 |
| teknofest_02 | 4 | 3 | 2 | 1 | 0 |
| ~~teknofest_03~~ | — | duplicate | — | — | — |
| teknofest_04 | 2 | 3 | 2 | — | — |
| teknofest_05 | 4 | 1 | 0 | 3 | 2 |
| teknofest_06 | 4 | 3 | 2 | 1 | 0 |
| ~~teknofest_07~~ | — | duplicate | — | — | — |
| teknofest_08 | 2 | 1 | 0 | — | — |
| teknofest_09 | 4 | 0 | 1 | 3 | 2 |
| teknofest_10 | 4 | 0 | 1 | 2 | 3 |
| teknofest_11 | 3 | 1 | 3 | 2 | — |
| teknofest_12 | 4 | 1 | 0 | 3 | 2 |
| ~~teknofest_13~~ | — | silindi | — | — | — |
| teknofest_14 | 1 | 3 | — | — | — |
| teknofest_15 | 4 | 0 | 1 | 3 | 2 |
| teknofest_16 | 4 | 1 | 3 | 2 | 0 |
| teknofest_17 | 2 | 2 | 3 | — | — |

---

## Görüntü & Label İstatistikleri

| Dataset | Train | Valid | Test | Toplam | Not |
|---------|------:|------:|-----:|-------:|-----|
| UAI_UAP | ~1,500 | — | — | ~1,500 | oversample=5 |
| drone-vision-project | ~5,000 | ~1,000 | ~1,000 | ~7,000 | oversample=2 |
| megaset | ~24,000 | — | — | ~24,000 | smart_sample, oversample=3 |
| teknofest_01 | 1,574 | — | — | 1,574 | |
| teknofest_02 | 19,217 | 1,934 | 1,113 | 22,264 | |
| ~~teknofest_03~~ | ~~0~~ | ~~0~~ | ~~0~~ | ~~boş~~ | duplicate |
| teknofest_04 | 187 | — | — | 187 | valid+test duplicate'ti |
| teknofest_05 | 1,840 | 533 | 264 | 2,637 | |
| teknofest_06 | 28,668 | 577 | 2,552 | 31,797 | |
| ~~teknofest_07~~ | ~~0~~ | ~~0~~ | ~~0~~ | ~~boş~~ | duplicate |
| teknofest_08 | 535 | — | — | 535 | |
| teknofest_09 | 5,535 | 405 | — | 5,940 | |
| teknofest_10 | 664 | — | — | 664 | |
| teknofest_11 | 1,897 | 21 | 7 | 1,925 | |
| teknofest_12 | 450 | — | — | 450 | |
| ~~teknofest_13~~ | — | — | — | silindi | 9 sınıf, uyumsuz |
| teknofest_14 | 54 | — | — | 54 | |
| teknofest_15 | 2,479 | 215 | — | 2,694 | |
| teknofest_16 | 130 | — | — | 130 | |
| teknofest_17 | 3,314 | 413 | 411 | 4,138 | |
| **TOPLAM** | **~97,500** | **~5,100** | **~5,350** | **~107,500** | |
