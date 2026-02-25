import glob
import os
import random
import argparse
import sys
from pathlib import Path

# Ensure project root on path when run as script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ultralytics import YOLO
from uav_training.config import DATASET_DIR, ARTIFACTS_DIR, IMAGE_EXTENSIONS

DEFAULT_INFER_SOURCE = str(DATASET_DIR / "val" / "images")

def smoke_infer(model_path, source=DEFAULT_INFER_SOURCE, num_images=5):
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Select random images (jpg, png, webp, bmp, tiff, gif, etc.)
    all_images = []
    for ext in IMAGE_EXTENSIONS:
        all_images.extend(glob.glob(os.path.join(source, f"*{ext}")))
    
    if not all_images:
        print(f"No images found in {source}")
        return
        
    selected_images = random.sample(all_images, min(len(all_images), num_images))
    print(f"Running inference on {len(selected_images)} images...")
    
    for img in selected_images:
        print(f" - {img}")
        
    save_dir = ARTIFACTS_DIR / "inference_smoke_test"
    
    try:
        results = model.predict(
            selected_images, 
            save=True, 
            project=str(save_dir.parent), 
            name=save_dir.name, 
            exist_ok=True
        )
        print(f"\nInference completed. Results saved to {save_dir}")
        
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Path to model file")
    parser.add_argument("--source", type=str, default=DEFAULT_INFER_SOURCE, help="Source directory (default: built dataset val/images)")
    parser.add_argument("--num", type=int, default=5, help="Number of images to test")
    
    args = parser.parse_args()
    smoke_infer(args.model, args.source, args.num)
