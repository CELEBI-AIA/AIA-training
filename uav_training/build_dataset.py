
import shutil
import os
from pathlib import Path
from tqdm import tqdm
import yaml
import json
import random
import atexit
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

set_seed(42)

# Use configuration from config.py if needed, or define constants here for standalone utility
from config import PROJECT_ROOT, DATASET_DIR, ARTIFACTS_DIR, TRAIN_CONFIG

# Using the optimized MAPPINGS from the successful unify_datasets.py
# Updated MAPPINGS for "TRAIN" folder
MAPPINGS = {
    # 1. Uap-UaiAlanlariVeriSeti.v2i.yolov8 (Existed before, confirmed in TRAIN)
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
    # SMART SAMPLING: Keep 100% of humans, 10% of vehicles.
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
    # names: ['UAI', 'UAI-', 'UAP', 'UAP-', 'car', 'people']
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

DATASETS_DIR = PROJECT_ROOT / "datasets" / "TRAIN"
MIN_BBOX_NORM = float(TRAIN_CONFIG.get("min_bbox_norm", 0.004))
INCLUDE_TEST_IN_VAL = bool(TRAIN_CONFIG.get("include_test_in_val", False))


def _acquire_file_lock(lock_path: Path) -> int:
    os.makedirs(lock_path.parent, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _release_file_lock(fd: int, lock_path: Path) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
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

    print("Starting optimized dataset unification (UAV Optimized)...")

    def _process_image_files(target_split, image_files, split, config, dataset_name, dataset_path, base_oversample_count, smart_sample):
        oversample_count = base_oversample_count if target_split == "train" else 1
        split_smart_sample = smart_sample and target_split == "train"
        sampling_rate = config.get("sampling_rate", 1.0)
        
        if not split_smart_sample and sampling_rate < 1.0:
            original_count = len(image_files)
            k = int(original_count * sampling_rate)
            if k > 0:
                image_files = random.sample(image_files, k)
            print(f"  Downsampled {split}: {original_count} -> {len(image_files)}")
        else:
            print(f"  Found {len(image_files)} images in {split}")
        
        if split_smart_sample:
            print(f"  Applying Smart Sampling for {dataset_name}...")
        
        kept_humans, kept_vehicles, skipped_smart = 0, 0, 0

        for i in range(oversample_count):
            desc_suffix = f" (Copy {i+1}/{oversample_count})" if oversample_count > 1 else ""
            file_iter = tqdm(image_files, desc=f"{dataset_name} - {target_split}{desc_suffix}")

            for img_path in file_iter:
                src_split_path = dataset_path / split if split else img_path.parent.parent # fallback for megaset direct paths
                label_path = None
                possible_label_dirs = [src_split_path / "labels", src_split_path]
                
                found_params = False
                for ld in possible_label_dirs:
                    if not ld.exists(): continue
                    candidate = ld / f"{img_path.stem}.txt"
                    if candidate.exists():
                        label_path = candidate
                        found_params = True
                        break
                
                if not found_params:
                    candidate = img_path.with_suffix(".txt")
                    if candidate.exists(): label_path = candidate
                
                if not label_path or not label_path.exists(): continue 
                    
                new_labels = []
                has_human, has_vehicle = False, False
                
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        
                    has_valid_cls = False
                    source_map = config.get("map", {})
                    id_map = config.get("id_map", {}) 
                    source_names = config.get("source_names", []) 
                    
                    temp_lines = []
                    _seen_lines = set()

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        
                        cls_id = int(parts[0])
                        target_id = None
                        if id_map and cls_id in id_map: target_id = id_map[cls_id]
                        elif source_names and cls_id < len(source_names):
                            cls_name = source_names[cls_id]
                            if cls_name in source_map: target_id = source_map[cls_name]
                        
                        if target_id is not None:
                            if target_id == 1: has_human = True
                            if target_id == 0: has_vehicle = True

                            try:
                                coords = list(map(float, parts[1:5]))
                                if len(coords) == 4:
                                    x, y, w, h = coords
                                    x, y, w, h = (max(0.0, min(1.0, v)) for v in (x, y, w, h))
                                    if MIN_BBOX_NORM < w <= 1.0 and MIN_BBOX_NORM < h <= 1.0:
                                        new_line = f"{target_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                                        if new_line not in _seen_lines:
                                            _seen_lines.add(new_line)
                                            temp_lines.append(new_line)
                                            has_valid_cls = True
                            except ValueError: continue
                    
                    if split_smart_sample:
                        keep = False
                        if has_human:
                            keep = True
                            kept_humans += 1
                        elif has_vehicle and random.random() < 0.10:
                            keep = True
                            kept_vehicles += 1
                        
                        if not keep:
                            skipped_smart += 1
                            continue
                    
                    if has_valid_cls:
                        new_labels = temp_lines
                        copy_suffix = f"_copy{i}" if oversample_count > 1 else ""
                        unique_name = f"{dataset_name}_{split}{copy_suffix}_{img_path.name}"
                        
                        target_img_path = DATASET_DIR / target_split / "images" / unique_name
                        target_lbl_path = DATASET_DIR / target_split / "labels" / (Path(unique_name).stem + ".txt")
                        
                        if not target_img_path.exists():
                            try: os.link(img_path, target_img_path)
                            except OSError: shutil.copy2(img_path, target_img_path)
                                
                        with open(target_lbl_path, 'w') as f:
                            f.writelines(new_labels)
                            
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        if split_smart_sample:
            print(f"  Smart Sampling Stats ({split}): Kept Humans: {kept_humans}, Kept Vehicles: {kept_vehicles}, Skipped: {skipped_smart}")

    def _execute_megaset_process(synthetic_splits, config, dataset_name, dataset_path, base_oversample_count, smart_sample):
        for target_split, image_files in synthetic_splits:
            _process_image_files(target_split, image_files, target_split, config, dataset_name, dataset_path, base_oversample_count, smart_sample)
    
    # Actually run the processing here (so it's defined and visible to megaset)
    # Re-apply the earlier block loops since we extracted logic out
    for dataset_name, config in MAPPINGS.items():
        dataset_path = DATASETS_DIR / dataset_name
        if not dataset_path.exists():
            continue
            
        base_oversample_count = config.get("oversample", 1)
        sampling_rate = config.get("sampling_rate", 1.0)
        smart_sample = config.get("smart_sample", False)
        
        source_splits = ["train", "valid", "val"]
        if INCLUDE_TEST_IN_VAL: source_splits.append("test")

        if dataset_name == "megaset":
            # Just do what was already setup above
            all_megaset_images = []
            for s in source_splits:
                if (dataset_path / s).exists():
                    images_dir = (dataset_path / s) / "images"
                    if images_dir.exists():
                        all_megaset_images.extend(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg")))
                    else:
                        all_megaset_images.extend(list((dataset_path / s).glob("*.jpg")) + list((dataset_path / s).glob("*.png")) + list((dataset_path / s).glob("*.jpeg")))
                     
            if not INCLUDE_TEST_IN_VAL and (dataset_path / "test").exists():
                images_dir = (dataset_path / "test") / "images"
                test_imgs = set(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg")) if images_dir.exists() else list((dataset_path / "test").glob("*.jpg")) + list((dataset_path / "test").glob("*.png")) + list((dataset_path / "test").glob("*.jpeg")))
                all_megaset_images = [img for img in all_megaset_images if img not in test_imgs]

            if not all_megaset_images: continue
            
            scene_groups = {}
            for p in all_megaset_images:
                stem = Path(p).stem
                parts = stem.rsplit("_frame_", 1)
                scene_id = parts[0] if len(parts) > 1 else stem[:8]
                scene_groups.setdefault(scene_id, []).append(p)

            scenes = list(scene_groups.keys())
            random.shuffle(scenes)
            val_scene_count = max(1, int(len(scenes) * 0.15))
            val_scenes = set(scenes[:val_scene_count])
            train_imgs, val_imgs = [], []
            for scene_id, imgs in scene_groups.items():
                if scene_id in val_scenes: val_imgs.extend(imgs)
                else: train_imgs.extend(imgs)
                
            synthetic_splits = [("train", train_imgs), ("val", val_imgs)]
            _execute_megaset_process(synthetic_splits, config, dataset_name, dataset_path, base_oversample_count, smart_sample)
            continue

        for split in source_splits:
            src_split_path = dataset_path / split
            if not src_split_path.exists(): continue
            target_split = "train" if split == "train" else "val"
            images_dir = src_split_path / "images"
            if images_dir.exists():
                image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
            else:
                image_files = list(src_split_path.glob("*.jpg")) + list(src_split_path.glob("*.png")) + list(src_split_path.glob("*.jpeg"))
            if not image_files: continue
            
            _process_image_files(target_split, image_files, split, config, dataset_name, dataset_path, base_oversample_count, smart_sample)


    # Generate data.yaml
    final_data_yaml = {
        'path': str(DATASET_DIR.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'values': ['vehicle', 'human', 'uap', 'uai'], # Display names
        'names': {
            0: 'vehicle',
            1: 'human',
            2: 'uap',
            3: 'uai'
        }
    }
    
    with open(DATASET_DIR / "dataset.yaml", 'w') as f:
        yaml.dump(final_data_yaml, f)
        
    print(f"Dataset built successfully at {DATASET_DIR}")

if __name__ == "__main__":
    build_dataset()
