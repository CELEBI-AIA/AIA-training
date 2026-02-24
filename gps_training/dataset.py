import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import json
import os
import atexit
import logging
import random
from collections import OrderedDict
from pathlib import Path
from config import DATASETS_ROOT, TRAIN_CONFIG, is_colab

logger = logging.getLogger(__name__)

# On Colab, if local extraction exists, use it to avoid Drive FUSE bottlenecks
if is_colab() and Path("/content/datasets_local").exists():
    EFFECTIVE_DATASETS_ROOT = Path("/content/datasets_local")
else:
    EFFECTIVE_DATASETS_ROOT = DATASETS_ROOT

class GPSDataset(Dataset):
    def __init__(self, audit_report_path, split="train", val_split=0.1):
        self.img_size = TRAIN_CONFIG["img_size"]
        self.split = split
        self.samples = []
        self._video_caps = {}
        self._frame_cache = OrderedDict()
        self._frame_cache_size = int(TRAIN_CONFIG.get("frame_cache_size", 256))
        atexit.register(self._close_video_caps)
        
        with open(audit_report_path, "r") as f:
            report = json.load(f)
            
        valid_seqs = [r for r in report if r["status"] == "included"]
        
        # Split sequences (not frames, to avoid data leak)
        # Or split frames? 
        # Splitting by sequence is safer for generalization, but we only have 8 sequences.
        # So we must split by time (first 90% train, last 10% val) or interleaved.
        # Time-split is best for trajectory forecasting.
        
        for item in valid_seqs:
            csv_path = EFFECTIVE_DATASETS_ROOT / item["csv_path"]
            media_path = EFFECTIVE_DATASETS_ROOT / item["media_path"]
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

    def _cache_get(self, media_path, frame_idx):
        key = (media_path, int(frame_idx))
        cached = self._frame_cache.get(key)
        if cached is None:
            return None
        # LRU touch
        self._frame_cache.move_to_end(key)
        return cached.copy()

    def _cache_put(self, media_path, frame_idx, frame):
        key = (media_path, int(frame_idx))
        self._frame_cache[key] = frame.copy()
        self._frame_cache.move_to_end(key)
        while len(self._frame_cache) > self._frame_cache_size:
            self._frame_cache.popitem(last=False)

    def _get_video_cap(self, media_path):
        cap = self._video_caps.get(media_path)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(media_path)
            self._video_caps[media_path] = cap
        return cap

    def _read_video_frame(self, media_path, frame_idx):
        cached = self._cache_get(media_path, frame_idx)
        if cached is not None:
            return cached

        cap = self._get_video_cap(media_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            # Re-open once in case decoder state is stale
            try:
                cap.release()
            except Exception:
                pass
            finally:
                self._video_caps.pop(media_path, None)
            cap = self._get_video_cap(media_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Could not read frame {frame_idx} from {media_path}")

        self._cache_put(media_path, frame_idx, frame)
        return frame

    def _read_video_pair(self, media_path, frame_idx_1, frame_idx_2):
        cached_1 = self._cache_get(media_path, frame_idx_1)
        cached_2 = self._cache_get(media_path, frame_idx_2)
        if cached_1 is not None and cached_2 is not None:
            return cached_1, cached_2

        # Fast path for consecutive frames: one seek + two reads
        if int(frame_idx_2) == int(frame_idx_1) + 1:
            cap = self._get_video_cap(media_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx_1))
            ret1, frame1 = cap.read()
            ret2, frame2 = cap.read()
            if ret1 and ret2:
                self._cache_put(media_path, frame_idx_1, frame1)
                self._cache_put(media_path, frame_idx_2, frame2)
                return frame1, frame2

            # Decoder state fallback
            try:
                cap.release()
            except Exception:
                pass
            finally:
                self._video_caps.pop(media_path, None)

        frame1 = cached_1 if cached_1 is not None else self._read_video_frame(media_path, frame_idx_1)
        frame2 = cached_2 if cached_2 is not None else self._read_video_frame(media_path, frame_idx_2)
        return frame1, frame2

    def _load_image(self, media_path, media_type, frame_idx):
        if media_type == "video":
            return self._read_video_frame(media_path, frame_idx)
        
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

    def _augment_pair(self, img1, img2, delta):
        """Apply identical augmentations to both Siamese frames.

        Only active during training. Geometric transforms that flip the
        spatial axis also negate the corresponding translation delta.
        """
        # Horizontal flip (50%)
        if random.random() < 0.5:
            img1 = cv2.flip(img1, 1)
            img2 = cv2.flip(img2, 1)
            delta[0] = -delta[0]  # negate translation_x

        # Color jitter (brightness / contrast) — same params for both
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.uniform(-20, 20)     # brightness
            img1 = np.clip(img1.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            img2 = np.clip(img2.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Gaussian blur (30%)
        if random.random() < 0.3:
            ksize = random.choice([3, 5])
            img1 = cv2.GaussianBlur(img1, (ksize, ksize), 0)
            img2 = cv2.GaussianBlur(img2, (ksize, ksize), 0)

        return img1, img2, delta

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            media_path = sample["media_path"]
            media_type = sample["media_type"]
            frame_idx_1 = sample["frame_idx_1"]
            frame_idx_2 = sample["frame_idx_2"]

            if media_type == "video":
                img1, img2 = self._read_video_pair(media_path, frame_idx_1, frame_idx_2)
            else:
                img1 = self._load_image(media_path, media_type, frame_idx_1)
                img2 = self._load_image(media_path, media_type, frame_idx_2)
            
            # Resize
            img1 = cv2.resize(img1, self.img_size)
            img2 = cv2.resize(img2, self.img_size)

            delta = sample["delta"].copy()

            # Augmentation (train only)
            if self.split == "train":
                img1, img2, delta = self._augment_pair(img1, img2, delta)
            
            # BGR to RGB (Keep as uint8 to save RAM/PCI-e bandwidth)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            # To Tensor (C, H, W)
            img1 = torch.from_numpy(img1).permute(2, 0, 1)
            img2 = torch.from_numpy(img2).permute(2, 0, 1)
            
            delta = torch.from_numpy(delta).float()
            
            return img1, img2, delta
            
        except Exception as e:
            logger.error(
                "Dataset sample failed idx=%s media=%s err=%s",
                idx,
                sample.get("media_path"),
                e,
            )
            return None

    def _close_video_caps(self):
        for _, cap in list(self._video_caps.items()):
            try:
                cap.release()
            except Exception:
                pass
        self._video_caps.clear()
        self._frame_cache.clear()

    def __del__(self):
        self._close_video_caps()
