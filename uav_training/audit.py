import os
import yaml
import json
import argparse
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    # simple fallback if tqdm missing
    def tqdm(x, **kwargs): return x

from config import PROJECT_ROOT, DATASETS_ROOT, AUDIT_REPORT, ARTIFACTS_DIR, TARGET_CLASSES

# Define output path using config
OUTPUT_REPORT = ARTIFACTS_DIR / "audit_report.json"


def get_subdirs(path):
    try:
        return [d.name for d in path.iterdir() if d.is_dir()]
    except Exception as e:
        print(f"Error accessing subdirs: {e}")
        return []

def read_yaml(path):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        return None

def read_txt_classes(path):
    try:
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"Error reading classes file: {e}")
        return []


def _list_images(path: Path):
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(path.glob(ext))
    return images


def _compute_split_overlap(split_stems: dict):
    train_val = split_stems["train"] & split_stems["val"]
    train_test = split_stems["train"] & split_stems["test"]
    val_test = split_stems["val"] & split_stems["test"]
    sample_names = sorted(list(train_val | train_test | val_test))[:20]
    return {
        "train_val_overlap": len(train_val),
        "train_test_overlap": len(train_test),
        "val_test_overlap": len(val_test),
        "has_overlap": bool(train_val or train_test or val_test),
        "sample_names": sample_names,
    }


def scan_and_audit():
    if not ARTIFACTS_DIR.exists():
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        
    results = []
    print(f"Scanning datasets in {DATASETS_ROOT}...")
    
    # We want to find subdirectories in DATASETS_ROOT that look like datasets
    # A simple heuristic: check immediate subdirectories. 
    # Or recursive? The user said "datasetleri ... içine koydum" implies they are direct children or one level deep.
    # Let's check direct children first to avoid scanning too deep and finding subfolders of datasets.
    
    if not DATASETS_ROOT.exists():
        print(f"Error: {DATASETS_ROOT} does not exist.")
        return

    # iterate over directories in DATASETS_ROOT
    dirs_to_audit = [d for d in DATASETS_ROOT.iterdir() if d.is_dir()]
    
    print(f"Found {len(dirs_to_audit)} candidates.")
    
    for d in dirs_to_audit:
        print(f"Auditing {d.name}...")
        # We need to pass full path or handle it in audit_directory
        # audit_directory currently takes name and uses BASE_PATH. 
        # We should update audit_directory to take full Path object.
        r = audit_directory(d) 
        results.append(r)
        print(f"  -> {r['status']} ({r['format']}): {r['reason']}")
    
    # Save results
    with open(AUDIT_REPORT, 'w') as f:
        json.dump(results, f, indent=2)
    
    valid_datasets = [r for r in results if r["status"] == "INCLUDE"]
    print(f"\nAudit complete. Found {len(valid_datasets)} valid datasets.")
    print(f"Report saved to {AUDIT_REPORT}")

# We need to accept Path object in audit_directory
def audit_directory(dir_path):
    # dir_path is a Path object now
    result = {
        "name": dir_path.name,
        "path": str(dir_path),
        "format": "unknown",
        "status": "SKIP",
        "reason": "Unknown structure",
        "classes": [],
        "uap_count": 0,
        "uai_count": 0,
        "image_count": 0,
        "label_count": 0,
        "split_counts": {
            "train": {"images": 0, "labels": 0},
            "val": {"images": 0, "labels": 0},
            "test": {"images": 0, "labels": 0},
        },
        "split_overlap": {
            "train_val_overlap": 0,
            "train_test_overlap": 0,
            "val_test_overlap": 0,
            "has_overlap": False,
            "sample_names": [],
        },
    }
    # ... rest of logic ...
    # but wait, the original audit_directory used line 47: dir_path = BASE_PATH / dir_name
    # we need to skip that line or modify it.
    
    # Let's verify if dir_path exists (it should since we listed it)
    if not dir_path.exists():
        result["reason"] = "Directory not found"
        return result

    # Check for README SKIP indicators
    readme_files = list(dir_path.glob("README*")) + list(dir_path.glob("*.txt")) + list(dir_path.glob("*.md"))
    is_sample = False
    for r in readme_files:
        try:
            content = r.read_text(errors='ignore').lower()
            if any(x in content for x in ["test only", "inference only", "sample", "ornek", "örnek"]):
                if "örnek" in r.name.lower() or "sample" in r.name.lower():
                     is_sample = True
        except Exception as e:
            # Silently pass errors from reading unstructured README files
            pass
            
    # Check for YOLO format (data.yaml)
    data_yaml = dir_path / "data.yaml"
    if data_yaml.exists():
        result["format"] = "YOLO"
        y = read_yaml(data_yaml)
        if y and 'names' in y:
            result["classes"] = y['names']
            
        # Count images/labels
        img_count = 0
        lbl_count = 0
        split_stems = {"train": set(), "val": set(), "test": set()}
        
        for sub in ['train', 'valid', 'val', 'test']:
            canonical_split = "val" if sub in {"valid", "val"} else sub
            img_dir = dir_path / sub / 'images'
            lbl_dir = dir_path / sub / 'labels'
            
            if img_dir.exists():
                imgs = _list_images(img_dir)
                img_count += len(imgs)
                result["split_counts"][canonical_split]["images"] += len(imgs)
                split_stems[canonical_split].update({p.stem for p in imgs})
            
            if lbl_dir.exists():
                lbls = list(lbl_dir.glob("*.txt"))
                lbl_count += len(lbls)
                result["split_counts"][canonical_split]["labels"] += len(lbls)
                
        result["image_count"] = img_count
        result["label_count"] = lbl_count
        result["split_overlap"] = _compute_split_overlap(split_stems)

    # Check for flat YOLO (classes.txt + images/labels in same dir)
    elif (dir_path / "classes.txt").exists() or (dir_path / "images&labels" / "classes.txt").exists():
        result["format"] = "YOLO_FLAT"
        if (dir_path / "classes.txt").exists():
             c_path = dir_path / "classes.txt"
             search_root = dir_path
        else:
             c_path = dir_path / "images&labels" / "classes.txt"
             search_root = dir_path / "images&labels"

        result["classes"] = read_txt_classes(c_path)
        
        imgs = list(search_root.glob("*.jpg")) + list(search_root.glob("*.png"))
        lbls = list(search_root.glob("*.txt"))
        # remove classes.txt from label count
        lbls = [l for l in lbls if l.name != "classes.txt"]
        
        result["image_count"] = len(imgs)
        result["label_count"] = len(lbls)
        
    elif "video" in str(dir_path).lower() or list(dir_path.glob("*.mp4")) or list(dir_path.glob("*.MP4")):
         result["format"] = "VIDEO"
         result["reason"] = "Video dataset (requires preprocessing)"
         result["status"] = "SKIP"
         return result
         
    else:
        result["reason"] = "No data.yaml or classes.txt found"
        result["status"] = "SKIP"
        imgs = list(dir_path.rglob("*.jpg")) + list(dir_path.rglob("*.png"))
        result["image_count"] = len(imgs)
        if len(imgs) > 0:
             result["reason"] = "Raw images found but no standard labeling detected"
        return result

    # Class filtering logic
    if isinstance(result["classes"], list):
        class_map = {i: n for i, n in enumerate(result["classes"])}
        class_values = result["classes"]
    elif isinstance(result["classes"], dict):
        class_map = result["classes"]
        class_values = list(result["classes"].values())
    else:
        class_map = {}
        class_values = []
        
    for idx, name in class_map.items():
        n = str(name).lower()
        for target_name in TARGET_CLASSES:
             if target_name in n:
                 key = f"{target_name}_count"
                 result[key] = 1
            
    # Include Decision
    if result["image_count"] < 10:
         result["status"] = "SKIP"
         result["reason"] = "Too few images"
    elif result["format"] == "YOLO" or result["format"] == "YOLO_FLAT":
         has_relevant = False
         for name in class_values:
             n = str(name).lower()
             # Updated relevant check using config
             for target_name in TARGET_CLASSES:
                 if target_name in n:
                     has_relevant = True
         
         if has_relevant:
             result["status"] = "INCLUDE"
             result["reason"] = "Valid YOLO format with target classes"
             if result["split_overlap"]["has_overlap"]:
                 result["reason"] += " | Split overlap risk detected"
             
             if is_sample:
                 pass 
         else:
             result["status"] = "SKIP" 
             result["reason"] = "No target classes found in names"
    
    return result

if __name__ == "__main__":
    scan_and_audit()
