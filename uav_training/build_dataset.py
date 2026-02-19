
import shutil
import os
from pathlib import Path
from tqdm import tqdm
import yaml
import json
import random

# Use configuration from config.py if needed, or define constants here for standalone utility
from config import PROJECT_ROOT, DATASET_DIR, ARTIFACTS_DIR

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

def build_dataset():
    if DATASET_DIR.exists():
        print(f"Removing existing {DATASET_DIR}...")
        shutil.rmtree(DATASET_DIR)
    
    (DATASET_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "val" / "images").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "val" / "labels").mkdir(parents=True, exist_ok=True)

    print("Starting optimized dataset unification (UAV Optimized)...")

    for dataset_name, config in MAPPINGS.items():
        dataset_path = DATASETS_DIR / dataset_name
        if not dataset_path.exists():
            print(f"Dataset {dataset_name} not found, skipping.")
            continue
            
        oversample_count = config.get("oversample", 1)
        sampling_rate = config.get("sampling_rate", 1.0)
        smart_sample = config.get("smart_sample", False)
        
        print(f"Processing {dataset_name} (Sample: {sampling_rate*100}%, Oversample: {oversample_count}x, SmartSample: {smart_sample})...")
        
        for split in ["train", "valid", "val", "test"]: 
                src_split_path = dataset_path / split
                
                # Special handling for datasets that might be named differently or missing splits
                if not src_split_path.exists():
                    # specific overrides if needed, or just skip
                    if dataset_name == "megaset" and split == "val":
                         # Megaset only has train and test. Try test as val.
                         src_split_path = dataset_path / "test"
                         if not src_split_path.exists(): continue
                    else:
                        continue

                target_split = "train" if split == "train" else "val"
                
                # Check for images dir vs flat structure
                images_dir = src_split_path / "images"
                image_files = []
                
                if images_dir.exists():
                    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
                else:
                    # Flat structure fallback
                    image_files = list(src_split_path.glob("*.jpg")) + list(src_split_path.glob("*.png")) + list(src_split_path.glob("*.jpeg"))
                
                if not image_files:
                    print(f"  No images found in {src_split_path}")
                    continue

                # Downsampling Logic (Only if NOT smart_sample)
                random.seed(42) 
                if not smart_sample and sampling_rate < 1.0:
                    original_count = len(image_files)
                    k = int(original_count * sampling_rate)
                    if k > 0:
                        image_files = random.sample(image_files, k)
                    print(f"  Downsampled {split}: {original_count} -> {len(image_files)}")
                else:
                    print(f"  Found {len(image_files)} images in {split}")
                
                if smart_sample:
                    print(f"  Applying Smart Sampling for {dataset_name}...")
                
                # Counters for smart stats
                kept_humans = 0
                kept_vehicles = 0
                skipped_smart = 0

                for i in range(oversample_count):
                    desc_suffix = f" (Copy {i+1}/{oversample_count})" if oversample_count > 1 else ""
                    
                    # Use tqdm if possible
                    file_iter = tqdm(image_files, desc=f"{dataset_name} - {split}{desc_suffix}")

                    for img_path in file_iter:
                        # Label Path Logic
                        label_path = None
                        possible_label_dirs = [
                            src_split_path / "labels",
                            src_split_path, # Same dir
                        ]
                        
                        found_params = False
                        for ld in possible_label_dirs:
                            if not ld.exists(): continue
                            candidate = ld / f"{img_path.stem}.txt"
                            if candidate.exists():
                                label_path = candidate
                                found_params = True
                                break
                        
                        if not found_params:
                            # Fallback for IEEECUSBTF
                            candidate = img_path.with_suffix(".txt")
                            if candidate.exists():
                                label_path = candidate
                        
                        if not label_path or not label_path.exists():
                            continue 
                            
                        # MAPPING LOGIC & SMART SAMPLING
                        new_labels = []
                        has_human = False
                        has_vehicle = False
                        
                        try:
                            with open(label_path, 'r') as f:
                                lines = f.readlines()
                                
                            has_valid_cls = False
                            
                            source_map = config.get("map", {})
                            id_map = config.get("id_map", {}) 
                            source_names = config.get("source_names", []) 
                            
                            temp_lines = []

                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) < 5: continue
                                
                                cls_id = int(parts[0])
                                target_id = None
                                
                                # Determine target ID
                                if id_map and cls_id in id_map:
                                    target_id = id_map[cls_id]
                                elif source_names and cls_id < len(source_names):
                                    cls_name = source_names[cls_id]
                                    if cls_name in source_map:
                                        target_id = source_map[cls_name]
                                
                                if target_id is not None:
                                    # Track what we found
                                    if target_id == 1: has_human = True
                                    if target_id == 0: has_vehicle = True

                                    # Parse coordinates
                                    try:
                                        coords = list(map(float, parts[1:5]))
                                        if len(coords) == 4:
                                            x, y, w, h = coords
                                            
                                            # Clamp to [0, 1]
                                            x = max(0.0, min(1.0, x))
                                            y = max(0.0, min(1.0, y))
                                            w = max(0.0, min(1.0, w))
                                            h = max(0.0, min(1.0, h))
                                            
                                            if w > 0.001 and h > 0.001:
                                                new_line = f"{target_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                                                temp_lines.append(new_line)
                                                has_valid_cls = True
                                    except ValueError:
                                        continue
                            
                            # SMART SAMPLING DECISION
                            if smart_sample:
                                keep = False
                                if has_human:
                                    keep = True # Keep 100% of humans
                                    kept_humans += 1
                                elif has_vehicle:
                                    if random.random() < 0.10: # Keep 10% of vehicle-only
                                        keep = True
                                        kept_vehicles += 1
                                
                                if not keep:
                                    skipped_smart += 1
                                    continue # Skip this image
                            
                            # Write output if valid
                            if has_valid_cls:
                                new_labels = temp_lines
                                
                                copy_suffix = f"_copy{i}" if oversample_count > 1 else ""
                                unique_name = f"{dataset_name}_{split}{copy_suffix}_{img_path.name}"
                                
                                target_img_path = DATASET_DIR / target_split / "images" / unique_name
                                target_lbl_path = DATASET_DIR / target_split / "labels" / (Path(unique_name).stem + ".txt")
                                
                                # Symlink or Copy image
                                if not target_img_path.exists():
                                    try:
                                        os.symlink(img_path.resolve(), target_img_path)
                                    except OSError:
                                        shutil.copy2(img_path, target_img_path)
                                        
                                with open(target_lbl_path, 'w') as f:
                                    f.writelines(new_labels)
                                    
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")
                            continue
                
                if smart_sample:
                    print(f"  Smart Sampling Stats ({split}): Kept Humans: {kept_humans}, Kept Vehicles: {kept_vehicles}, Skipped: {skipped_smart}")

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
