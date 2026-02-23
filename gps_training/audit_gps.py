import os
import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np

# Adjust path to import config
import sys
sys.path.append(str(Path(__file__).parent))
from config import DATASETS_ROOT, ARTIFACTS_DIR, is_colab

if is_colab() and Path("/content/datasets_local").exists():
    EFFECTIVE_DATASETS_ROOT = Path("/content/datasets_local")
else:
    EFFECTIVE_DATASETS_ROOT = DATASETS_ROOT

def recursive_scan(root_dir):
    candidates = []
    print(f"Scanning {root_dir}...")
    
    for root, dirs, files in os.walk(root_dir):
        root_path = Path(root)
        
        # Look for translation CSVs
        for file in files:
            if file.endswith(".csv"):
                # Potential candidate
                csv_path = root_path / file
                try:
                    # Quick check of headers
                    df_head = pd.read_csv(csv_path, nrows=5)
                    required_cols = {"translation_x", "translation_y", "frame_numbers"}
                    if required_cols.issubset(df_head.columns):
                        # Found a valid trace file
                        candidates.append({
                            "csv_path": csv_path,
                            "root": root_path,
                            "filename": file
                        })
                except Exception as e:
                    # Not a csv or unreadable
                    pass
                    
    return candidates

def find_media(candidate):
    """
    Find corresponding video or image directory.
    Heuristic: Remove 'Translation', 'GT', related suffixes from CSV name and look for match.
    """
    csv_name = candidate["filename"]
    root = candidate["root"]
    
    # Common patterns:
    # Name_Translation.csv -> Name.mp4 or Name/
    # Name-translation.csv -> Name.mp4
    # GT_Translations.csv -> [Scanning for .mp4 in folder]
    
    base_name = csv_name.replace("_Translation", "").replace("-translation", "").replace(".csv", "")
    
    # Check 1: Directory with base_name
    dir_match = root / base_name
    if dir_match.exists() and dir_match.is_dir():
        # Check if it has images
        images = list(dir_match.glob("*.jpg")) + list(dir_match.glob("*.png")) + list(dir_match.glob("*.webp"))
        if len(images) > 10:
            return {"type": "images", "path": dir_match, "count": len(images)}
            
    # Check 2: Video file with base_name
    for ext in [".mp4", ".MP4", ".avi", ".AVI"]:
        vid_match = root / (base_name + ext)
        if vid_match.exists():
            return {"type": "video", "path": vid_match}
            
    # Check 3: If CSV is "GT_Translations", look for ANY video or matching images folder in dir
    if "GT_Translations" in csv_name:
        # Look for video
        videos = list(root.glob("*.mp4")) + list(root.glob("*.MP4"))
        if len(videos) == 1:
            return {"type": "video", "path": videos[0]}
            
    # Check 4: Check if 'images' folder exists
    img_dir = root / "images"
    if img_dir.exists():
         images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.webp"))
         if len(images) > 10:
             return {"type": "images", "path": img_dir, "count": len(images)}

    return None

def find_intrinsics(candidate):
    """
    Search for calibration txt file.
    """
    root = candidate["root"]
    
    # Look for files with "Kalibrasyon", "Calibration", "Intrinsics"
    # Check current dir and parent dir
    search_dirs = [root, root.parent]
    
    for d in search_dirs:
        for file in d.glob("*.txt"):
            if any(k in file.name.lower() for k in ["kalibrasyon", "calibration", "intrinsic"]):
                return file
                
    return None

def audit():
    candidates = recursive_scan(EFFECTIVE_DATASETS_ROOT)
    report = []
    
    print(f"Found {len(candidates)} potential trajectory CSVs.")
    
    for cand in candidates:
        print(f"Checking {cand['filename']}...")
        
        # 1. Media
        media = find_media(cand)
        
        # 2. Intrinsics
        intrinsics = find_intrinsics(cand)
        
        # 3. Validation
        status = "excluded"
        reason = []
        
        if not media:
            reason.append("No media found (video/images)")
        
        # Check CSV content validity
        try:
            df = pd.read_csv(cand['csv_path'])
            if len(df) < 10:
                reason.append("Too few rows")
            
            # Check delta vs absolute
            # We assume INPUT is absolute, we calculate deltas. 
            # If standard deviation is huge, maybe it's weird.
            # But we generally accept it if cols exist.
            
            # Check for Z
            if "translation_z" not in df.columns:
                reason.append("Missing Z axis (only 2D)") # We might optionally allow this if we want to padding
                
        except Exception as e:
            reason.append(f"CSV Read Error: {e}")
            
        if not reason:
            status = "included"
            
        entry = {
            "csv_path": str(cand['csv_path'].relative_to(EFFECTIVE_DATASETS_ROOT)),
            "media_path": str(media['path'].relative_to(EFFECTIVE_DATASETS_ROOT)) if media else None,
            "media_type": media['type'] if media else None,
            "intrinsics_path": str(intrinsics.relative_to(EFFECTIVE_DATASETS_ROOT)) if intrinsics else None,
            "status": status,
            "reasons": reason,
            "num_frames": len(df) if not reason else 0
        }
        report.append(entry)
        
    # Stats
    included = [r for r in report if r['status'] == 'included']
    print(f"Audit Complete. Included: {len(included)}, Excluded: {len(report) - len(included)}")
    
    # Save
    out_path = ARTIFACTS_DIR / "gps_audit_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Report saved to {out_path}")

if __name__ == "__main__":
    audit()
