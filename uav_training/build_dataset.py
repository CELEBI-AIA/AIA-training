
import shutil
import os
from pathlib import Path
from tqdm import tqdm
import yaml
import json
import random
import atexit
import math
import platform

_EPS = 1e-6

if platform.system() == "Windows":
    import msvcrt
else:
    import fcntl

def set_seed(seed=42):
    import random, numpy, torch
    random.seed(seed)
    numpy.random.seed(seed)
    # ultralytics/torch might be imported later but we set it just in case
    try:
        torch.manual_seed(seed)
    except NameError:
        pass

# Use configuration from config.py if needed, or define constants here for standalone utility
from config import PROJECT_ROOT, DATASET_DIR, DATASETS_TRAIN_DIR, ARTIFACTS_DIR, TRAIN_CONFIG

# Using the optimized MAPPINGS from the successful unify_datasets.py
# Updated MAPPINGS for "TRAIN" folder
MAPPINGS = {
    # 1. Uap-UaiAlanlariVeriSeti.v2i.yolov8 (Existed before, confirmed in TRAIN)
    # nc: 6, names: ['UAI', 'UAI-', 'UAP', 'UAP-', 'car', 'people']
    # UAI  = iniş yapmaya uygun UAI alanı (suitable landing area)
    # UAI- = iniş yapmaya uygun OLMAYAN UAI alanı (unsuitable landing area)
    # UAP  = iniş yapmaya uygun UAP alanı (suitable landing area)
    # UAP- = iniş yapmaya uygun OLMAYAN UAP alanı (unsuitable landing area)
    # Both suitable/unsuitable variants are merged into the same target class.
    "Uap-UaiAlanlariVeriSeti.v2i.yolov8": {
        "source_names": ['UAI', 'UAI-', 'UAP', 'UAP-', 'car', 'people'],
        "map": {
            'UAI': 3, 'UAI-': 3, 
            'UAP': 2, 'UAP-': 2, 
            'car': 0, 
            'people': 1
        },
        "oversample": 3, 
        "sampling_rate": 1.0
    },
    
    # REMOVED Teknofest.v2 as per user request (Archived)

    # 3. drone-vision-project (New found in TRAIN)
    # names: ['car', 'pedestrian']
    "drone-vision-project": {
        "source_names": ['car', 'pedestrian'],
        "map": {
            'car': 0,
            'pedestrian': 1
        },
        "oversample": 3, 
        "sampling_rate": 1.0
    },

    # 4. megaset (Existed before, confirmed in TRAIN)
    # HUGE dataset (24k images).
    # SMART SAMPLING: Keep 100% of humans, 30% of vehicles.
    "megaset": {
        "source_names": ['vehicle', 'pedestrian'], 
        "map": {
            'vehicle': 0,
            'pedestrian': 1
        },
        "id_map": {0: 0, 1: 1},
        "oversample": 2,
        "sampling_rate": 1.0, # Process ALL images effectively, but filter inside
        "smart_sample": True  # Enable class-based filtering
    },
    
    # 5. Uap-UaiAlanlariVeriSeti (Confirmed via data.yaml)
    # nc: 6, names: ['UAI', 'UAI-', 'UAP', 'UAP-', 'car', 'people']
    # UAI  = iniş yapmaya uygun UAI alanı (suitable landing area)
    # UAI- = iniş yapmaya uygun OLMAYAN UAI alanı (unsuitable landing area)
    # UAP  = iniş yapmaya uygun UAP alanı (suitable landing area)
    # UAP- = iniş yapmaya uygun OLMAYAN UAP alanı (unsuitable landing area)
    # Both suitable/unsuitable variants are merged into the same target class.
    "Uap-UaiAlanlariVeriSeti": {
        "source_names": ['UAI', 'UAI-', 'UAP', 'UAP-', 'car', 'people'],
        "map": {
            'UAI': 3, 'UAI-': 3, 
            'UAP': 2, 'UAP-': 2, 
            'car': 0, 
            'people': 1
        },
        "oversample": 3, # High priority UAP/UAI data
        "sampling_rate": 1.0
    }
}

DEFAULT_CLASS_KEEP_PROB = {0: 0.30, 1: 1.00, 2: 1.00, 3: 1.00}


def resolve_target_split(split_name: str, include_test_in_val: bool) -> str:
    """Map source split names into final split names with explicit test handling."""
    split = (split_name or "").strip().lower()
    if split in {"train", "training"}:
        return "train"
    if split in {"valid", "val", "validation"}:
        return "val"
    if split in {"test", "testing"}:
        return "val" if include_test_in_val else "test"
    return "val"


def _list_images(base_path: Path) -> list[Path]:
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(base_path.glob(ext))
    return image_files


def _acquire_file_lock(lock_path: Path) -> int:
    os.makedirs(lock_path.parent, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    if platform.system() == "Windows":
        # Lock a single byte to serialize concurrent builders on Windows.
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _release_file_lock(fd: int, lock_path: Path) -> None:
    try:
        if platform.system() == "Windows":
            try:
                os.lseek(fd, 0, os.SEEK_SET)
            except OSError:
                pass
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        os.remove(lock_path)
    except OSError:
        pass

def build_dataset():
    lock_path = ARTIFACTS_DIR / ".build_dataset.lock"
    lock_fd = _acquire_file_lock(lock_path)
    atexit.register(_release_file_lock, lock_fd, lock_path)

    if DATASET_DIR.exists():
        print(f"Removing existing {DATASET_DIR}...")
        shutil.rmtree(DATASET_DIR)
    
    (DATASET_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "val" / "images").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "val" / "labels").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "test" / "images").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "test" / "labels").mkdir(parents=True, exist_ok=True)

    min_bbox_norm = float(TRAIN_CONFIG.get("min_bbox_norm", 0.004))
    include_test_in_val = bool(TRAIN_CONFIG.get("include_test_in_val", False))

    print("Starting optimized dataset unification (UAV Optimized)...")

    def _process_image_files(target_split, image_files, split, config, dataset_name, dataset_path, base_oversample_count, smart_sample):
        oversample_count = base_oversample_count if target_split == "train" else 1
        split_smart_sample = smart_sample and target_split == "train"
        sampling_rate = config.get("sampling_rate", 1.0)
        dst_images_dir = DATASET_DIR / target_split / "images"
        try:
            dst_dev = dst_images_dir.stat().st_dev
        except OSError:
            dst_dev = None

        if not split_smart_sample and sampling_rate < 1.0:
            original_count = len(image_files)
            k = int(original_count * sampling_rate)
            if k > 0:
                set_seed(42) # R-01 Fix: Make sampling deterministic
                image_files = random.sample(image_files, k)
            print(f"  Downsampled {split}: {original_count} -> {len(image_files)}")
        else:
            print(f"  Found {len(image_files)} images in {split}")

        class_keep_prob = config.get("smart_sample_keep_prob", DEFAULT_CLASS_KEEP_PROB)
        if not isinstance(class_keep_prob, dict):
            class_keep_prob = DEFAULT_CLASS_KEEP_PROB
        normalized_keep_prob = {}
        for k, v in class_keep_prob.items():
            try:
                cls_id = int(k)
                keep_prob = max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                continue
            normalized_keep_prob[cls_id] = keep_prob
        class_keep_prob = normalized_keep_prob or DEFAULT_CLASS_KEEP_PROB

        if split_smart_sample:
            print(f"  Applying Smart Sampling for {dataset_name}...")

        for i in range(oversample_count):
            kept_by_class = {cls_id: 0 for cls_id in sorted(class_keep_prob.keys())}
            skipped_smart = 0
            out_of_range_cls = 0
            unmapped_cls = 0
            missing_label_count = 0
            out_of_range_bbox_count = 0
            nan_bbox_count = 0
            too_small_bbox_count = 0
            short_line_count = 0  # len(parts) < 5 (segmentation/keypoint format)
            invalid_coords_count = 0
            total_bbox_kept = 0

            desc_suffix = f" (Copy {i+1}/{oversample_count})" if oversample_count > 1 else ""
            file_iter = tqdm(image_files, desc=f"{dataset_name} - {target_split}{desc_suffix}")

            for img_path in file_iter:
                src_split_path = dataset_path / split if split else img_path.parent.parent  # fallback for megaset direct paths
                label_path = None
                possible_label_dirs = [src_split_path / "labels", src_split_path]

                found_params = False
                for ld in possible_label_dirs:
                    if not ld.exists():
                        continue
                    candidate = ld / f"{img_path.stem}.txt"
                    if candidate.exists():
                        label_path = candidate
                        found_params = True
                        break

                if not found_params:
                    candidate = img_path.with_suffix(".txt")
                    if candidate.exists():
                        label_path = candidate

                if not label_path or not label_path.exists():
                    missing_label_count += 1
                    continue

                new_labels = []
                present_target_ids = set()

                try:
                    with open(label_path, "r") as f:
                        lines = f.readlines()

                    has_valid_cls = False
                    source_map = config.get("map", {})
                    id_map = config.get("id_map", {})
                    source_names = config.get("source_names", [])

                    temp_lines = []
                    _seen_lines = set()

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            short_line_count += 1
                            continue

                        try:
                            cls_id = int(parts[0])
                        except ValueError:
                            short_line_count += 1
                            continue

                        target_id = None
                        if id_map and cls_id in id_map:
                            target_id = id_map[cls_id]
                        elif source_names:
                            if cls_id >= len(source_names):
                                out_of_range_cls += 1
                                continue
                            cls_name = source_names[cls_id]
                            if cls_name in source_map:
                                target_id = source_map[cls_name]
                            else:
                                unmapped_cls += 1
                                continue

                        if target_id is None:
                            continue

                        try:
                            coords = list(map(float, parts[1:5]))
                            if len(coords) != 4:
                                invalid_coords_count += 1
                                continue
                            x, y, w, h = coords

                            # NaN/Inf guard
                            if any(v != v or math.isinf(v) for v in (x, y, w, h)):
                                nan_bbox_count += 1
                                continue

                            # Strict range check (eps tolerance for float rounding only)
                            if not all(-_EPS <= v <= 1.0 + _EPS for v in (x, y, w, h)):
                                out_of_range_bbox_count += 1
                                continue

                            # Micro-clamp for float precision artifacts within eps
                            x = min(1.0, max(0.0, x))
                            y = min(1.0, max(0.0, y))
                            w = min(1.0, max(0.0, w))
                            h = min(1.0, max(0.0, h))

                            if not (min_bbox_norm < w <= 1.0 and min_bbox_norm < h <= 1.0):
                                too_small_bbox_count += 1
                                continue

                            # KR-2 Fix: Add to present_target_ids ONLY after all bbox validity checks pass!
                            target_id = int(target_id)
                            present_target_ids.add(target_id)

                            new_line = f"{target_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                            if new_line not in _seen_lines:
                                _seen_lines.add(new_line)
                                temp_lines.append(new_line)
                                has_valid_cls = True
                                total_bbox_kept += 1
                        except ValueError:
                            continue

                    if split_smart_sample:
                        if not present_target_ids:
                            skipped_smart += 1
                            continue
                        keep_prob = max(class_keep_prob.get(cls_id, 0.25) for cls_id in present_target_ids)
                        if random.random() > keep_prob:
                            skipped_smart += 1
                            continue
                        for cls_id in present_target_ids:
                            if cls_id in kept_by_class:
                                kept_by_class[cls_id] += 1

                    if has_valid_cls:
                        new_labels = temp_lines
                        copy_suffix = f"_copy{i}" if oversample_count > 1 else ""
                        unique_name = f"{dataset_name}_{split}{copy_suffix}_{img_path.name}"

                        target_img_path = DATASET_DIR / target_split / "images" / unique_name
                        target_lbl_path = DATASET_DIR / target_split / "labels" / (Path(unique_name).stem + ".txt")

                        if not target_img_path.exists():
                            same_device = dst_dev is not None and img_path.stat().st_dev == dst_dev
                            if same_device:
                                try:
                                    os.link(img_path, target_img_path)
                                except OSError:
                                    shutil.copy2(img_path, target_img_path)
                            else:
                                shutil.copy2(img_path, target_img_path)

                        with open(target_lbl_path, "w") as f:
                            f.writelines(new_labels)

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

            if split_smart_sample:
                print(f"  Smart Sampling Stats ({split}): KeptByClass={kept_by_class}, Skipped={skipped_smart}")
            if out_of_range_cls > 0:
                print(f"  ⚠️ {dataset_name}/{split}: out-of-range class_id count = {out_of_range_cls}")
            if unmapped_cls > 0:
                print(f"  ⚠️ {dataset_name}/{split}: unmapped class count = {unmapped_cls}")
            if missing_label_count > 0:
                print(f"  ⚠️ {dataset_name}/{split}: missing label files = {missing_label_count}")
            rejected = out_of_range_bbox_count + nan_bbox_count + too_small_bbox_count
            excluded = short_line_count + invalid_coords_count
            print(
                f"  [BBOX AUDIT] {dataset_name}/{target_split}{desc_suffix}: "
                f"kept={total_bbox_kept} out_of_range={out_of_range_bbox_count} "
                f"nan={nan_bbox_count} too_small={too_small_bbox_count} "
                f"rejected_total={rejected}"
            )
            if excluded > 0:
                print(
                    f"  [EXCLUDED LABELS] {dataset_name}/{target_split}{desc_suffix}: "
                    f"short_format={short_line_count} invalid_coords={invalid_coords_count} "
                    f"(unsupported segmentation/keypoint or malformed lines)"
                )

    def _execute_megaset_process(synthetic_splits, config, dataset_name, dataset_path, base_oversample_count, smart_sample):
        for target_split, image_files in synthetic_splits:
            _process_image_files(target_split, image_files, None, config, dataset_name, dataset_path, base_oversample_count, smart_sample)
    
    # Actually run the processing here (so it's defined and visible to megaset)
    # Re-apply the earlier block loops since we extracted logic out
    for dataset_name, config in MAPPINGS.items():
        dataset_path = DATASETS_TRAIN_DIR / dataset_name
        if not dataset_path.exists():
            continue
            
        base_oversample_count = config.get("oversample", 1)
        sampling_rate = config.get("sampling_rate", 1.0)
        smart_sample = config.get("smart_sample", False)
        
        source_splits = ["train", "valid", "val", "test"]

        if dataset_name == "megaset":
            # Keep scene-aware split generation for train/val from non-test splits.
            all_megaset_images = []
            for s in ("train", "valid", "val"):
                if (dataset_path / s).exists():
                    images_dir = (dataset_path / s) / "images"
                    if images_dir.exists():
                        all_megaset_images.extend(_list_images(images_dir))
                    else:
                        all_megaset_images.extend(_list_images(dataset_path / s))

            if all_megaset_images:
                scene_groups = {}
                for p in all_megaset_images:
                    stem = Path(p).stem
                    parts = stem.rsplit("_frame_", 1)
                    scene_id = parts[0] if len(parts) > 1 else stem
                    scene_groups.setdefault(scene_id, []).append(p)

                scenes = list(scene_groups.keys())
                set_seed(42)  # deterministic train/val split for megaset
                random.shuffle(scenes)
                val_scene_count = max(1, int(len(scenes) * 0.15))
                val_scenes = set(scenes[:val_scene_count])
                train_imgs, val_imgs = [], []
                for scene_id, imgs in scene_groups.items():
                    if scene_id in val_scenes:
                        val_imgs.extend(imgs)
                    else:
                        train_imgs.extend(imgs)

                synthetic_splits = [("train", train_imgs), ("val", val_imgs)]
                _execute_megaset_process(synthetic_splits, config, dataset_name, dataset_path, base_oversample_count, smart_sample)

            # Process explicit test split with strict mapping policy.
            test_split_path = dataset_path / "test"
            if test_split_path.exists():
                test_images_dir = test_split_path / "images"
                test_images = _list_images(test_images_dir if test_images_dir.exists() else test_split_path)
                if test_images:
                    mapped_split = resolve_target_split("test", include_test_in_val)
                    _process_image_files(
                        mapped_split,
                        test_images,
                        "test",
                        config,
                        dataset_name,
                        dataset_path,
                        base_oversample_count,
                        smart_sample,
                    )
            continue

        for split in source_splits:
            src_split_path = dataset_path / split
            if not src_split_path.exists():
                continue
            target_split = resolve_target_split(split, include_test_in_val)
            images_dir = src_split_path / "images"
            if images_dir.exists():
                image_files = _list_images(images_dir)
            else:
                image_files = _list_images(src_split_path)
            if not image_files:
                continue

            _process_image_files(target_split, image_files, split, config, dataset_name, dataset_path, base_oversample_count, smart_sample)


    # Generate data.yaml
    final_data_yaml = {
        'path': str(DATASET_DIR.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 4,
        'names': {
            0: 'vehicle',
            1: 'human',
            2: 'uap',
            3: 'uai'
        }
    }

    test_images_dir = DATASET_DIR / "test" / "images"
    if test_images_dir.exists() and any(test_images_dir.iterdir()):
        final_data_yaml["test"] = "test/images"
    else:
        final_data_yaml["test"] = "val/images"  # Safe fallback to prevent KeyError in val(split="test")
    
    with open(DATASET_DIR / "dataset.yaml", 'w') as f:
        yaml.dump(final_data_yaml, f)
        
    print(f"Dataset built successfully at {DATASET_DIR}")

    _release_file_lock(lock_fd, lock_path)
    try:
        atexit.unregister(_release_file_lock)
    except AttributeError:
        pass


if __name__ == "__main__":
    build_dataset()
