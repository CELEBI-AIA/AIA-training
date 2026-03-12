# UAI/UAP Dataset — Class Mapping Dokümantasyonu

## Evrensel Sınıf Şeması

Birleştirilmiş veriseti için tüm kaynak datasetlerdeki sınıflar aşağıdaki 4 evrensel ID'ye dönüştürülür:

| ID | Sınıf | Açıklama |
|----|-------|----------|
| `0` | **taşıt** | Her türlü kara taşıtı (araba, araç, vehicle, car, tasit) |
| `1` | **insan** | İnsan (person, human, insan) |
| `2` | **uap** | Unidentified Aerial Phenomenon |
| `3` | **uai** | Unidentified Aerial Intelligence |

---

## Hedef `data.yaml`

```yaml
nc: 4
names:
  0: tasit
  1: insan
  2: uap
  3: uai
```

---

## Dataset Başına Sınıf Eşlemeleri

> **Okuma kılavuzu:** `orijinal_id (orijinal_isim) → yeni_id` formatındadır.  
> Label dosyalarındaki her satırdaki sınıf numarası bu tabloya göre dönüştürülmelidir.

---

### teknofest_01
**Orijinal:** `['0', '1', '2', '3']`  
**Kaynak:** berk / teknofest-tvmlr / v1

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | 0 | → | **0** | taşıt |
| 1 | 1 | → | **1** | insan |
| 2 | 2 | → | **2** | uap |
| 3 | 3 | → | **3** | uai |

---

### teknofest_02
**Orijinal:** `['UAI', 'UAP', 'person', 'vehicle']`  
**Kaynak:** workspace-71uin / teknofest-layzj / v12

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | UAI | → | **3** | uai |
| 1 | UAP | → | **2** | uap |
| 2 | person | → | **1** | insan |
| 3 | vehicle | → | **0** | taşıt |

---

### ~~teknofest_03~~
> ❌ **Duplicate temizliği sonrası tamamen boşaldı, klasör pakete dahil edilmedi.**  
> teknofest_02 ile aynı Roboflow kaynağından geliyordu (workspace-71uin / teknofest-layzj / v12), tüm görüntüleri duplicate olarak silindi.

---

### teknofest_04
**Orijinal:** `['uai', 'uap']`  
**Kaynak:** hyz-moa5c / uap-uai-demo / v1

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | uai | → | **3** | uai |
| 1 | uap | → | **2** | uap |

> ℹ️ Duplicate temizliği sonrası yalnızca train/187 görüntü kaldı (valid ve test tamamen duplicate'ti).

---

### teknofest_05
**Orijinal:** `['0', '1', '2', '3']`  
**Kaynak:** teknofest-gpscr / teknofest-ejiwb / v1

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | 0 | → | **1** | insan |
| 1 | 1 | → | **0** | taşıt |
| 2 | 2 | → | **3** | uai |
| 3 | 3 | → | **2** | uap |

---

### teknofest_06
**Orijinal:** `['UAI', 'UAP', 'insan', 'tasit']`  
**Kaynak:** teknofest-m874p / teknofest-jfohi / v19

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | UAI | → | **3** | uai |
| 1 | UAP | → | **2** | uap |
| 2 | insan | → | **1** | insan |
| 3 | tasit | → | **0** | taşıt |

---

### ~~teknofest_07~~
> ❌ **Duplicate temizliği sonrası tamamen boşaldı, klasör pakete dahil edilmedi.**  
> Tüm görüntüleri başka datasetlerde duplicate olarak tespit edildi ve silindi.

---

### teknofest_08
**Orijinal:** `['Person', 'Vehicle']`  
**Kaynak:** nez-6dv1g / teknofest-vmp4c / v2

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | Person | → | **1** | insan |
| 1 | Vehicle | → | **0** | taşıt |

---

### teknofest_09
**Orijinal:** `['0', '1', '2', '3']`  
**Kaynak:** teknofest-tcg1m / teknofest-apxmr / v3

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | 0 | → | **0** | taşıt |
| 1 | 1 | → | **1** | insan |
| 2 | 2 | → | **3** | uai |
| 3 | 3 | → | **2** | uap |

---

### teknofest_10
**Orijinal:** `['0', '1', '2_uap', '3_uai']`  
**Kaynak:** cars-wiqof / uap-uai-m8ass / v1

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | 0 | → | **0** | taşıt |
| 1 | 1 | → | **1** | insan |
| 2 | 2_uap | → | **2** | uap |
| 3 | 3_uai | → | **3** | uai |

---

### teknofest_11
**Orijinal:** `['person', 'uai', 'uap']`  
**Kaynak:** cay-xglwe / teknofest-dde1g-lacm6 / v1

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | person | → | **1** | insan |
| 1 | uai | → | **3** | uai |
| 2 | uap | → | **2** | uap |

---

### teknofest_12
**Orijinal:** `['insan', 'tasit', 'uai', 'uap']`  
**Kaynak:** yarma / teknofest-veri-2024-antalya / v1

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | insan | → | **1** | insan |
| 1 | tasit | → | **0** | taşıt |
| 2 | uai | → | **3** | uai |
| 3 | uap | → | **2** | uap |

---

### ~~teknofest_13~~
> ❌ **Bu dataset verisetine dahil edilmedi ve silindi.**  
> Sebep: 9 sınıflı karmaşık yapı (UAI/UAP rotation/track/keyframe varyantları), standart eşlemeyle uyumsuz.

---

### teknofest_14
**Orijinal:** `['ambulans']`  
**Kaynak:** hamza-vvz7g / wg-4lmct / v1

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | ambulans | → | **3** | uai |

> ⚠️ Ambulans etiketi UAI olarak yorumlanmıştır. Sadece 54 train görüntüsü içerir.

---

### teknofest_15
**Orijinal:** `['car', 'insan', 'uai', 'uap']`  
**Kaynak:** denem5 / uap-uai-iutyw / v6

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | car | → | **0** | taşıt |
| 1 | insan | → | **1** | insan |
| 2 | uai | → | **3** | uai |
| 3 | uap | → | **2** | uap |

---

### teknofest_16
**Orijinal:** `['Human', 'UAI', 'UAP', 'Vehicle']`  
**Kaynak:** tulpar-etiketleme-2 / tulpar-uap-uai / v2

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | Human | → | **1** | insan |
| 1 | UAI | → | **3** | uai |
| 2 | UAP | → | **2** | uap |
| 3 | Vehicle | → | **0** | taşıt |

---

### teknofest_17
**Orijinal:** `{0: uap, 1: uai}`  
**Kaynak:** yerel dataset (uaiuap)

| Orijinal ID | Orijinal İsim | → | Yeni ID | Yeni Sınıf |
|:-----------:|---------------|:-:|:-------:|------------|
| 0 | uap | → | **2** | uap |
| 1 | uai | → | **3** | uai |

---

## Hızlı Başvuru — Dönüşüm Tablosu

| Dataset | nc | 0→ | 1→ | 2→ | 3→ |
|---------|:--:|:--:|:--:|:--:|:--:|
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

## Görüntü & Label İstatistikleri (Duplicate Temizliği Sonrası)

| Dataset | Train | Valid | Test | Toplam | Not |
|---------|------:|------:|-----:|-------:|-----|
| teknofest_01 | 1,574 | — | — | 1,574 | |
| teknofest_02 | 19,217 | 1,934 | 1,113 | 22,264 | |
| ~~teknofest_03~~ | ~~0~~ | ~~0~~ | ~~0~~ | ~~boş~~ | teknofest_02 ile birebir aynıydı |
| teknofest_04 | 187 | — | — | 187 | valid+test duplicate'ti |
| teknofest_05 | 1,840 | 533 | 264 | 2,637 | |
| teknofest_06 | 28,668 | 577 | 2,552 | 31,797 | |
| ~~teknofest_07~~ | ~~0~~ | ~~0~~ | ~~0~~ | ~~boş~~ | tüm görüntüler duplicate'ti |
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
| **AKTİF TOPLAM** | **66,544** | **4,098** | **4,347** | **74,989** | |
