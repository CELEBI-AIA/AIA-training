import glob
import os
import random
import argparse
from ultralytics import YOLO
from pathlib import Path

from config import DATASET_DIR, ARTIFACTS_DIR, DATASETS_ROOT

def smoke_infer(model_path, source=str(DATASET_DIR / "val" / "images"), num_images=5):
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Select random images
    all_images = glob.glob(os.path.join(source, "*.jpg")) + glob.glob(os.path.join(source, "*.png"))
    
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
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to model file")
    parser.add_argument("--source", type=str, default=str(DATASETS_ROOT / "TEST_DATA"), help="Source directory")
    parser.add_argument("--num", type=int, default=5, help="Number of images to test")
    
    args = parser.parse_args()
    smoke_infer(args.model, args.source, args.num)
