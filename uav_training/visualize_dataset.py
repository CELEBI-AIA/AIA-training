import argparse
import random
import sys
import cv2
from pathlib import Path
from tqdm import tqdm

# Ensure project root on path when run as script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from uav_training.config import DATASET_DIR, ARTIFACTS_DIR, IMAGE_EXTENSIONS  # noqa: E402
OUTPUT_DIR = ARTIFACTS_DIR / "verification_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define class colors for visualization
CLASS_COLORS = {
    0: (0, 0, 255),    # Vehicle: Red
    1: (0, 255, 0),    # Human: Green
    2: (255, 165, 0),  # UAP: Orange
    3: (255, 0, 0)     # UAI: Blue
}

CLASS_NAMES = {
    0: 'vehicle',
    1: 'human',
    2: 'uap',
    3: 'uai'
}

def verify_dataset(num_samples=20, split="train"):
    """Visualize labeled images from the specified split (train, val, or test)."""
    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"

    print(f"Verifying dataset in {DATASET_DIR} (split={split})...")

    if not images_dir.exists():
        print(f"Error: {images_dir} does not exist. Run build_dataset.py first.")
        return

    # Get all image files (jpg, png, webp, bmp, tiff, gif, etc.)
    ext_set = set(IMAGE_EXTENSIONS)
    image_files = [
        p for p in images_dir.glob("*")
        if p.is_file() and p.suffix.lower() in ext_set
    ]

    if not image_files:
        print("No images found.")
        return

    print(f"Found {len(image_files)} images. Sampling {num_samples}...")

    # Sample images
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    for img_path in tqdm(samples):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Determine label path
        label_path = labels_dir / f"{img_path.stem}.txt"

        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    # Denormalize
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)

                    color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                    label = CLASS_NAMES.get(cls_id, str(cls_id))

                    # Draw box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save output
        out_path = OUTPUT_DIR / f"vis_{split}_{img_path.name}"
        cv2.imwrite(str(out_path), img)

    print(f"Verification samples saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize dataset labels")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to visualize (default: train)",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=20,
        help="Number of samples to visualize (default: 20)",
    )
    args = parser.parse_args()
    verify_dataset(num_samples=args.num, split=args.split)

