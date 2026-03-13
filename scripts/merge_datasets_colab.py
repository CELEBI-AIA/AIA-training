#!/usr/bin/env python3
"""
Colab Merge Script — TRAIN_DATA + uaiuapdataset → TRAIN_DATA_COMBINED.tar.gz
=============================================================================
Parallel download (Drive→SSD) + pigz extraction + merge + repackage.

Kullanım (Colab hücresinde):
    !python /content/repo/scripts/merge_datasets_colab.py
"""

import os
import sys
import time
import shutil
import subprocess
import threading
import concurrent.futures

# ─── Paths ───────────────────────────────────────────────────────
DRIVE_DATASETS = "/content/drive/MyDrive/AIA/datasets"

OLD_ARCHIVE = os.path.join(DRIVE_DATASETS, "TRAIN_DATA.tar.gz")
NEW_ARCHIVE = os.path.join(DRIVE_DATASETS, "uaiuapdataset.tar.gz")

OUTPUT_ARCHIVE = os.path.join(DRIVE_DATASETS, "TRAIN_DATA_COMBINED.tar.gz")

WORK_DIR = "/content/merge_workspace"
MERGED_DIR = os.path.join(WORK_DIR, "TRAIN_DATA")
DOWNLOAD_DIR = os.path.join(WORK_DIR, "downloads")

# ─── Config ──────────────────────────────────────────────────────
DOWNLOAD_WORKERS = 8
CHUNK_SIZE = 32 * 1024 * 1024  # 32 MB


# ─── Helpers ─────────────────────────────────────────────────────
def _run(cmd, desc=""):
    if desc:
        print(f"  ⏳ {desc}...", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"  ❌ HATA ({elapsed:.0f}s): {r.stderr[:500]}", flush=True)
        sys.exit(1)
    print(f"  ✅ ({elapsed:.0f}s)", flush=True)
    return r


def _disk_free_gb():
    st = os.statvfs("/content")
    return (st.f_bavail * st.f_frsize) / (1024**3)


def _size_gb(path):
    return os.path.getsize(path) / (1024**3)


def _count_files(path):
    total = 0
    for _, _, files in os.walk(path):
        total += len(files)
    return total


# ─── Parallel Download ──────────────────────────────────────────
def _parallel_download(src_path, dst_path, workers=DOWNLOAD_WORKERS, chunk=CHUNK_SIZE):
    """Multi-threaded byte-range copy from Drive FUSE to local SSD."""
    file_size = os.path.getsize(src_path)
    name = os.path.basename(src_path)
    print(f"  📥 {name}: {file_size / (1024**3):.2f} GB → local SSD ({workers} workers, {chunk // (1024**1024)}MB chunks)", flush=True)

    # Pre-allocate output file
    with open(dst_path, "wb") as f:
        f.truncate(file_size)

    progress_lock = threading.Lock()
    progress = {"bytes": 0}
    t0 = time.time()

    def _copy_range(start, end):
        """Copy byte range [start, end) using pread/pwrite."""
        src_fd = os.open(src_path, os.O_RDONLY)
        dst_fd = os.open(dst_path, os.O_WRONLY)
        try:
            offset = start
            while offset < end:
                read_size = min(chunk, end - offset)
                data = os.pread(src_fd, read_size, offset)
                if not data:
                    break
                os.pwrite(dst_fd, data, offset)
                offset += len(data)
                with progress_lock:
                    progress["bytes"] += len(data)
        finally:
            os.close(src_fd)
            os.close(dst_fd)

    # Progress printer
    def _print_progress():
        while not done_event.is_set():
            with progress_lock:
                done_bytes = progress["bytes"]
            elapsed = time.time() - t0
            pct = done_bytes / file_size * 100
            speed = done_bytes / elapsed / (1024**2) if elapsed > 0 else 0
            eta = (file_size - done_bytes) / (done_bytes / elapsed) if done_bytes > 0 else 0
            print(
                f"\r    {pct:5.1f}%  |  {done_bytes / (1024**3):.1f}/{file_size / (1024**3):.1f} GB"
                f"  |  {speed:.0f} MB/s  |  ETA {eta:.0f}s   ",
                end="", flush=True,
            )
            done_event.wait(1.0)
        print(f"\r    100.0%  |  {file_size / (1024**3):.1f} GB  |  done{'':30s}", flush=True)

    done_event = threading.Event()
    progress_thread = threading.Thread(target=_print_progress, daemon=True)
    progress_thread.start()

    # Split file into ranges for workers
    ranges = []
    for i in range(0, file_size, file_size // workers + 1):
        end = min(i + file_size // workers + 1, file_size)
        ranges.append((i, end))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_copy_range, s, e) for s, e in ranges]
        concurrent.futures.wait(futures)
        for f in futures:
            f.result()  # Raise if any failed

    done_event.set()
    progress_thread.join()
    elapsed = time.time() - t0
    speed = file_size / elapsed / (1024**2)
    print(f"  ✅ İndirildi: {elapsed:.0f}s ({speed:.0f} MB/s)", flush=True)
    return dst_path


# ─── Main ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DATASET BİRLEŞTİRME")
    print("  TRAIN_DATA + uaiuapdataset → TRAIN_DATA_COMBINED")
    print("=" * 60)

    # Pre-checks
    for archive, name in [(OLD_ARCHIVE, "TRAIN_DATA.tar.gz"), (NEW_ARCHIVE, "uaiuapdataset.tar.gz")]:
        if not os.path.isfile(archive):
            print(f"  ❌ HATA: {name} bulunamadı: {archive}")
            sys.exit(1)
        print(f"  📦 {name}: {_size_gb(archive):.2f} GB")

    print(f"  💾 Boş disk: {_disk_free_gb():.1f} GB")

    needed = _size_gb(OLD_ARCHIVE) + _size_gb(NEW_ARCHIVE) + 50  # extracted + tar output
    if _disk_free_gb() < needed:
        print(f"  ⚠️  En az {needed:.0f} GB boş alan gerekli!")
        sys.exit(1)

    # 1) Workspace
    print(f"\n{'─'*60}")
    print("  1/6 — Workspace hazırlanıyor")
    print(f"{'─'*60}")
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(MERGED_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Install pigz
    subprocess.run("which pigz >/dev/null 2>&1 || apt-get install -y -qq pigz >/dev/null 2>&1", shell=True)

    # 2) Parallel download BOTH archives to local SSD
    print(f"\n{'─'*60}")
    print("  2/6 — Arşivler local SSD'ye indiriliyor (paralel)")
    print(f"{'─'*60}")

    local_old = os.path.join(DOWNLOAD_DIR, "TRAIN_DATA.tar.gz")
    local_new = os.path.join(DOWNLOAD_DIR, "uaiuapdataset.tar.gz")

    # Download sequentially (Drive FUSE doesn't like concurrent large reads on different files)
    _parallel_download(OLD_ARCHIVE, local_old)
    _parallel_download(NEW_ARCHIVE, local_new)

    print(f"  💾 Boş disk: {_disk_free_gb():.1f} GB")

    # 3) Extract TRAIN_DATA.tar.gz
    print(f"\n{'─'*60}")
    print("  3/6 — TRAIN_DATA.tar.gz çıkartılıyor")
    print(f"{'─'*60}")
    _run(
        f"tar -I 'pigz -p {os.cpu_count()}' -xf {local_old} -C {MERGED_DIR}",
        desc="TRAIN_DATA.tar.gz extract"
    )
    os.remove(local_old)
    print(f"  🗑️  local tar.gz silindi - {_disk_free_gb():.1f} GB boş")

    # Flatten if wrapped in a subdirectory
    subdirs = [d for d in os.listdir(MERGED_DIR) if os.path.isdir(os.path.join(MERGED_DIR, d))]
    if len(subdirs) == 1 and subdirs[0] in ("TRAIN_DATA", "mega", "datasets"):
        wrapper = os.path.join(MERGED_DIR, subdirs[0])
        print(f"  🔄 Wrapper klasör ({subdirs[0]}/) düzleştiriliyor...")
        for item in os.listdir(wrapper):
            shutil.move(os.path.join(wrapper, item), os.path.join(MERGED_DIR, item))
        os.rmdir(wrapper)

    print(f"  📁 İçerik: {sorted(os.listdir(MERGED_DIR))}")

    # 4) Extract uaiuapdataset.tar.gz
    print(f"\n{'─'*60}")
    print("  4/6 — uaiuapdataset.tar.gz çıkartılıyor")
    print(f"{'─'*60}")
    uai_temp = os.path.join(WORK_DIR, "_uai_temp")
    os.makedirs(uai_temp, exist_ok=True)
    _run(
        f"tar -I 'pigz -p {os.cpu_count()}' -xf {local_new} -C {uai_temp}",
        desc="uaiuapdataset.tar.gz extract"
    )
    os.remove(local_new)
    print(f"  🗑️  local tar.gz silindi - {_disk_free_gb():.1f} GB boş")

    # Find teknofest root
    uai_root = uai_temp
    candidates = [d for d in os.listdir(uai_temp) if os.path.isdir(os.path.join(uai_temp, d))]
    if len(candidates) == 1 and candidates[0].startswith("uai"):
        uai_root = os.path.join(uai_temp, candidates[0])

    # Move teknofest_XX folders
    teknofest_folders = sorted([
        d for d in os.listdir(uai_root)
        if os.path.isdir(os.path.join(uai_root, d)) and d.startswith("teknofest_")
    ])
    print(f"  📁 {len(teknofest_folders)} teknofest klasörü taşınıyor...")
    for folder in teknofest_folders:
        src = os.path.join(uai_root, folder)
        dst = os.path.join(MERGED_DIR, folder)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)

    shutil.rmtree(uai_temp)

    # 5) Verify
    print(f"\n{'─'*60}")
    print("  5/6 — Doğrulama")
    print(f"{'─'*60}")
    final_dirs = sorted([d for d in os.listdir(MERGED_DIR) if os.path.isdir(os.path.join(MERGED_DIR, d))])
    total_files = _count_files(MERGED_DIR)
    print(f"  📁 {len(final_dirs)} klasör, {total_files} dosya:")
    for d in final_dirs:
        cnt = _count_files(os.path.join(MERGED_DIR, d))
        print(f"       {d}/ ({cnt} dosya)")

    expected_old = {"UAI_UAP", "drone-vision-project", "megaset"}
    expected_new = {f"teknofest_{i:02d}" for i in range(1, 18) if i != 13}
    found = set(final_dirs)
    missing = (expected_old | expected_new) - found
    if missing:
        print(f"  ⚠️  Eksik: {sorted(missing)}")
    else:
        print(f"  ✅ Tüm beklenen klasörler mevcut!")

    # 6) Compress + upload
    print(f"\n{'─'*60}")
    print("  6/6 — Sıkıştırma ve Drive'a yükleme")
    print(f"{'─'*60}")

    local_tar = os.path.join(WORK_DIR, "TRAIN_DATA_COMBINED.tar.gz")
    _run(
        f"cd {WORK_DIR} && tar -cf - TRAIN_DATA | pigz -p {os.cpu_count()} > {local_tar}",
        desc=f"Sıkıştırma (pigz, {os.cpu_count()} çekirdek)"
    )

    tar_size = _size_gb(local_tar)
    print(f"  📦 Arşiv: {tar_size:.2f} GB  |  💾 Boş disk: {_disk_free_gb():.1f} GB")

    # Upload
    if os.path.exists(OUTPUT_ARCHIVE):
        print(f"  🔄 Eski dosya yedekleniyor (.bak)")
        os.rename(OUTPUT_ARCHIVE, OUTPUT_ARCHIVE + ".bak")

    print(f"  ☁️  Drive'a kopyalanıyor ({tar_size:.2f} GB)...")
    t0 = time.time()
    shutil.copy2(local_tar, OUTPUT_ARCHIVE)
    elapsed = time.time() - t0
    speed = tar_size * 1024 / elapsed if elapsed > 0 else 0
    print(f"  ✅ Yüklendi: {elapsed:.0f}s ({speed:.1f} MB/s)")

    # Cleanup
    shutil.rmtree(WORK_DIR)

    print(f"\n{'='*60}")
    print(f"  ✅ BİRLEŞTİRME TAMAMLANDI!")
    print(f"  📦 {OUTPUT_ARCHIVE}")
    print(f"  📊 {tar_size:.2f} GB | {total_files} dosya | {len(final_dirs)} klasör")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
