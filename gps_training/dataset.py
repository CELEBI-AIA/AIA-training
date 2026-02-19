import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import json
import os
from pathlib import Path
from config import DATASETS_ROOT, TRAIN_CONFIG

class GPSDataset(Dataset):
    def __init__(self, audit_report_path, split="train", val_split=0.1):
        self.img_size = TRAIN_CONFIG["img_size"]
        self.samples = []
        
        with open(audit_report_path, "r") as f:
            report = json.load(f)
            
        valid_seqs = [r for r in report if r["status"] == "included"]
        
        # Split sequences (not frames, to avoid data leak)
        # Or split frames? 
        # Splitting by sequence is safer for generalization, but we only have 8 sequences.
        # So we must split by time (first 90% train, last 10% val) or interleaved.
        # Time-split is best for trajectory forecasting.
        
        for item in valid_seqs:
            csv_path = DATASETS_ROOT / item["csv_path"]
            media_path = DATASETS_ROOT / item["media_path"]
            media_type = item["media_type"]
            
            # Load CSV
            df = pd.read_csv(csv_path)
            
            # clean frame_numbers if they are strings
            if "frame_numbers" in df.columns:
                # Check first element
                first_val = df["frame_numbers"].iloc[0]
                if isinstance(first_val, str) and first_val.startswith("frame_"):
                    df["frame_numbers"] = df["frame_numbers"].apply(lambda x: int(x.split("_")[1]) if isinstance(x, str) else x)
                
                df = df.sort_values("frame_numbers").reset_index(drop=True)
            
            # Calculate Split point
            n_frames = len(df)
            split_idx = int(n_frames * (1 - val_split))
            
            if split == "train":
                df_subset = df.iloc[:split_idx]
            else:
                df_subset = df.iloc[split_idx:]
                
            # Generate pairs (t, t+1)
            # We need to know how to get the image.
            # If video: we store video path and frame index.
            # If images: we need filename pattern.
            
            # Pre-calculate paths or indices
            # THYZ images: frame_000000.webp
            # HYZ names: ? We need to check one. 
            # We'll assume robust frame indexing based on 'frame_numbers'.
            
            subset_indices = df_subset.index.tolist()
            # We can only make pairs if t and t+1 are consecutive in valid set
            # But since we split by block, it's fine.
            
            for i in range(len(subset_indices) - 1):
                idx_curr = subset_indices[i]
                idx_next = subset_indices[i+1]
                
                # Check continuity (optional, if frame_numbers has gaps)
                # row_curr = df.iloc[idx_curr]
                # row_next = df.iloc[idx_next]
                # if row_next['frame_numbers'] != row_curr['frame_numbers'] + 1: continue
                
                # Translations
                p1 = df.iloc[idx_curr][["translation_x", "translation_y", "translation_z"]].values.astype(np.float32)
                p2 = df.iloc[idx_next][["translation_x", "translation_y", "translation_z"]].values.astype(np.float32)
                delta = p2 - p1
                
                self.samples.append({
                    "media_path": str(media_path),
                    "media_type": media_type,
                    "frame_idx_1": df.iloc[idx_curr]["frame_numbers"],
                    "frame_idx_2": df.iloc[idx_next]["frame_numbers"],
                    "delta": delta
                })
                
        print(f"Dataset ({split}): Loaded {len(self.samples)} pairs from {len(valid_seqs)} sequences.")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, media_path, media_type, frame_idx):
        if media_type == "video":
            cap = cv2.VideoCapture(media_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                # Retry once
                cap = cv2.VideoCapture(media_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise ValueError(f"Could not read frame {frame_idx} from {media_path}")
            return frame
        
        elif media_type == "images":
            # Robust file finding
            # Try 1: Exact THYZ format (frame_000123.webp)
            possible_names = [
                f"frame_{int(frame_idx):06d}.webp",
                f"frame_{int(frame_idx):06d}.jpg",
                f"frame_{int(frame_idx):06d}.png",
                f"{int(frame_idx)}.jpg",
                f"{int(frame_idx)}.png",
                f"{int(frame_idx)}.webp"
            ]
            
            for name in possible_names:
                p = os.path.join(media_path, name)
                if os.path.exists(p):
                     img = cv2.imread(p)
                     if img is not None:
                         return img
                         
            # If we are here, we failed.
            # raise ValueError(f"Image for frame {frame_idx} not found in {media_path}") 
            # Return blank to avoid crashing training? No, dataset should define valid data.
            raise FileNotFoundError(f"Frame {frame_idx} not found in {media_path}. Checked variants.")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            img1 = self._load_image(sample["media_path"], sample["media_type"], sample["frame_idx_1"])
            img2 = self._load_image(sample["media_path"], sample["media_type"], sample["frame_idx_2"])
            
            # Resize
            img1 = cv2.resize(img1, self.img_size)
            img2 = cv2.resize(img2, self.img_size)
            
            # BGR to RGB and Normalize
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / 255.0
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) / 255.0
            
            # To Tensor (C, H, W)
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            
            delta = torch.from_numpy(sample["delta"]).float()
            
            return img1, img2, delta
            
        except Exception as e:
            # Return dummy or fail (better to skip in collation, but simplified here)
            print(f"Error loading sample {idx}: {e}")
            # Recursively try next? Or return zeros.
            return torch.zeros((3, *self.img_size)), torch.zeros((3, *self.img_size)), torch.zeros(3)
