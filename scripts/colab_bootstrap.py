##############################################################################
# 🚀 YOLO UAV Training Bootstrap — Google Colab
# Paste this entire cell into Colab and run.
##############################################################################

VERSION = "0.7.0"

# ── Configuration ───────────────────────────────────────────────────────────
REPO_URL       = "https://github.com/CELEBI-AIA/AIA-training.git"
REPO_BRANCH    = "main"
DRIVE_DATASET  = "/content/drive/MyDrive/AIA/datasets.tar.gz"
LOCAL_CACHE    = "/content/datasets_local"
DRIVE_RUNS     = "/content/drive/MyDrive/AIA/runs"
DRIVE_UPLOAD   = "/content/drive/MyDrive/AIA"     # best.pt upload destination
TRAIN_SCRIPT   = "uav_training/train.py"
# ────────────────────────────────────────────────────────────────────────────

import subprocess, sys, os, glob, time
from datetime import datetime

# ── Force unbuffered output → ALL print/log lines appear in Colab INSTANTLY ──
os.environ["PYTHONUNBUFFERED"] = "1"

def _run(cmd: str, *, check: bool = True, **kw):
    """Run a shell command — capture output and print to Colab cell explicitly.

    Colab's IPython stdout is NOT a real file descriptor, so subprocess.run()
    without capture sends output to /dev/null. We capture + print instead.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    r = subprocess.run(
        cmd, shell=True, check=check,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, **kw
    )
    if r.stdout:
        sys.stdout.write(r.stdout)
        sys.stdout.flush()
    return r

def _banner(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}", flush=True)

print(f"\n🛰️  UAV Training Bootstrap v{VERSION}", flush=True)
print(f"    Repo: {REPO_URL} ({REPO_BRANCH})\n", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
# 0. Pre-Flight Cleanup
# ═══════════════════════════════════════════════════════════════════════════
_banner("0/8 — Pre-flight cleanup")

import gc

# Kill leftover training processes
print("  🧹 Killing stale processes …", flush=True)
_run("pkill -9 -f 'uav_training/train.py' 2>/dev/null || true", check=False)
_run("pkill -9 -f 'yolo' 2>/dev/null || true", check=False)
_run("pkill -9 -f 'build_dataset.py' 2>/dev/null || true", check=False)

# Free GPU memory
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("  🧹 GPU VRAM cache cleared", flush=True)
except Exception:
    pass

# Python garbage collect
gc.collect()

# Clear __pycache__ in repo if it exists
REPO_DIR = "/content/repo"
if os.path.isdir(REPO_DIR):
    _run(f'find {REPO_DIR} -type d -name __pycache__ -exec rm -rf {{}} + 2>/dev/null || true', check=False)
    # NOTE: Do NOT delete *.cache files — these are YOLO's label caches
    # that store pre-scanned 30k+ labels. Deleting them costs 35+ seconds per run.
    print("  🧹 Stale __pycache__ cleared", flush=True)

# Clear /tmp junk
_run("rm -rf /tmp/pip-* /tmp/torch_* 2>/dev/null || true", check=False)
print("  ✓ Cleanup done\n", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Mount Google Drive
# ═══════════════════════════════════════════════════════════════════════════
_banner("1/8 — Mounting Google Drive")
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

if not os.path.isfile(DRIVE_DATASET):
    raise FileNotFoundError(
        f"Dataset archive not found at {DRIVE_DATASET}. "
        "Upload your datasets.tar.gz to Google Drive first."
    )

os.makedirs(DRIVE_RUNS, exist_ok=True)
os.makedirs(DRIVE_UPLOAD, exist_ok=True)
print(f"  ✓ Drive mounted", flush=True)
print(f"  ✓ Dataset archive: {DRIVE_DATASET}")
# Print archive size
tar_size = os.path.getsize(DRIVE_DATASET) / (1024**3)
print(f"  ✓ Archive size: {tar_size:.2f} GB")
print(f"  ✓ Runs dir: {DRIVE_RUNS}")
print(f"  ✓ Upload dir: {DRIVE_UPLOAD}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Clone or Pull Repository
# ═══════════════════════════════════════════════════════════════════════════
_banner("2/8 — Setting up repository")
REPO_DIR = "/content/repo"

if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    print("  ↻ Repo exists — pulling latest changes …", flush=True)
    _run(f"git -C {REPO_DIR} fetch --all")
    _run(f"git -C {REPO_DIR} reset --hard origin/{REPO_BRANCH}")
else:
    print(f"  ↓ Cloning {REPO_URL} …", flush=True)
    _run(f"git clone --depth 1 -b {REPO_BRANCH} {REPO_URL} {REPO_DIR}")

# Show repo info
_run(f"git -C {REPO_DIR} log --oneline -1")
print("  ✓ Repo ready", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
# 3. Install Requirements
# ═══════════════════════════════════════════════════════════════════════════
_banner("3/8 — Installing dependencies")

req_file = os.path.join(REPO_DIR, "requirements.txt")
if os.path.isfile(req_file):
    print("  📦 Installing requirements.txt …", flush=True)
    _run(f"{sys.executable} -m pip install --progress-bar on -r {req_file}")
else:
    print("  ⚠ No requirements.txt found")

# Ensure ultralytics + psutil are available
print("  📦 Verifying ultralytics & psutil …", flush=True)
_run(f"{sys.executable} -m pip install --progress-bar on ultralytics psutil", check=False)

# Print key package versions
print("\n  📊 Package versions:", flush=True)
_run(f'{sys.executable} -c "import ultralytics; print(f\'     ultralytics: {{ultralytics.__version__}}\')"', check=False)
_run(f'{sys.executable} -c "import torch; print(f\'     torch: {{torch.__version__}}, CUDA: {{torch.version.cuda}}\')"', check=False)
print("  ✓ Dependencies installed", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
# 4. Dataset: Download tar.gz → Local SSD → Extract → Verify → Cleanup
# ═══════════════════════════════════════════════════════════════════════════
_banner("4/8 — Dataset to local SSD (DOWNLOAD → EXTRACT → VERIFY)")

os.makedirs(LOCAL_CACHE, exist_ok=True)

# Install tools
print("  📦 Installing extraction tools …", flush=True)
_run("apt-get update -qq && apt-get install -y pigz pv 2>&1 | tail -3", check=False)

CACHE_MARKER = os.path.join(LOCAL_CACHE, ".done")
NCPU = os.cpu_count() or 2
LOCAL_TAR = "/content/datasets.tar.gz"  # Local SSD copy of the archive

existing = sum(len(f) for _, _, f in os.walk(LOCAL_CACHE))
t0 = time.time()

if os.path.isfile(CACHE_MARKER):
    # ═══ Already extracted — skip everything ═══
    print(f"  ⚡ Cache complete ({existing} files) — SKIP", flush=True)

elif existing > 5000:
    print(f"  ⚡ Dataset detected ({existing} files) — SKIP", flush=True)
    with open(CACHE_MARKER, 'w') as f:
        f.write("1")

else:
    # ── PHASE 1: Download tar.gz from Drive FUSE → Local SSD ──
    # Drive FUSE streaming is slow (~200 MB/s). Copying the whole file
    # to local SSD first lets us extract at full NVMe speed (~3 GB/s).
    tar_size_gb = os.path.getsize(DRIVE_DATASET) / (1024**3)

    if os.path.isfile(LOCAL_TAR):
        local_size = os.path.getsize(LOCAL_TAR)
        drive_size = os.path.getsize(DRIVE_DATASET)
        if local_size == drive_size:
            print(f"  ⚡ tar.gz already on local SSD ({tar_size_gb:.1f} GB) — skip download", flush=True)
        else:
            print(f"  ♻️  Partial download ({local_size/(1024**3):.1f}/{tar_size_gb:.1f} GB) — re-downloading", flush=True)
            os.remove(LOCAL_TAR)
            print(f"  📥 Downloading {tar_size_gb:.1f} GB → local SSD …", flush=True)
            # Use dd with large block size for max Drive FUSE throughput
            _run(f'dd if="{DRIVE_DATASET}" of="{LOCAL_TAR}" bs=64M status=progress 2>&1 | tail -5')
    else:
        print(f"  📥 Downloading {tar_size_gb:.1f} GB → local SSD …", flush=True)
        print(f"     Source: {DRIVE_DATASET}", flush=True)
        print(f"     Target: {LOCAL_TAR}", flush=True)
        # dd with 64M blocks saturates Drive FUSE bandwidth
        _run(f'dd if="{DRIVE_DATASET}" of="{LOCAL_TAR}" bs=64M status=progress 2>&1 | tail -5')

    dl_elapsed = time.time() - t0
    print(f"  ✓ Download done in {dl_elapsed:.0f}s ({tar_size_gb/max(dl_elapsed,1)*1024:.0f} MB/s)", flush=True)

    # ── PHASE 2: Extract from LOCAL copy (full NVMe speed) ──
    t1 = time.time()
    TAR_FAST = "--no-same-owner --no-same-permissions -b 512"

    if existing > 100:
        print(f"  ♻️  Incremental extraction ({existing} existing files) …", flush=True)
        _run(
            f'pigz -d -c -p {NCPU} "{LOCAL_TAR}" '
            f'| tar -xf - -C "{LOCAL_CACHE}" '
            f'{TAR_FAST} --skip-old-files '
            f'--checkpoint=10000 '
            f'--checkpoint-action=echo="  %u files checked…"'
        )
    else:
        print(f"  🚀 Full extraction → {LOCAL_CACHE} (pigz {NCPU} cores, local SSD)", flush=True)
        _run(
            f'pigz -d -c -p {NCPU} "{LOCAL_TAR}" '
            f'| tar -xf - -C "{LOCAL_CACHE}" '
            f'{TAR_FAST} '
            f'--checkpoint=10000 '
            f'--checkpoint-action=echo="  %u files extracted…"'
        )

    ext_elapsed = time.time() - t1
    final_count = sum(len(f) for _, _, f in os.walk(LOCAL_CACHE))
    print(f"  ✓ Extracted in {ext_elapsed:.0f}s — {final_count} files", flush=True)

    # ── PHASE 3: Verify & Cleanup ──
    if final_count > 5000:
        print(f"  ✅ Verification passed: {final_count} files on local SSD", flush=True)
        with open(CACHE_MARKER, 'w') as f:
            f.write("1")
        # Delete local tar.gz to free ~78 GB of SSD space
        if os.path.isfile(LOCAL_TAR):
            os.remove(LOCAL_TAR)
            print(f"  🗑️  Deleted local tar.gz — freed {tar_size_gb:.1f} GB", flush=True)
    else:
        print(f"  ⚠️  Only {final_count} files extracted — keeping tar.gz for retry", flush=True)

elapsed = time.time() - t0
final_count = sum(len(f) for _, _, f in os.walk(LOCAL_CACHE))
print(f"\n  ✓ Total time: {elapsed:.0f}s — {final_count} files ready on SSD", flush=True)

# Symlink repo's datasets/ → local cache so training scripts find it
repo_datasets = os.path.join(REPO_DIR, "datasets")
if os.path.islink(repo_datasets) or os.path.isdir(repo_datasets):
    _run(f'rm -rf "{repo_datasets}"')
os.symlink(LOCAL_CACHE, repo_datasets)
print(f"  🔗 Symlinked {repo_datasets} → {LOCAL_CACHE}")

# Disk usage report
print("\n  📊 Disk usage:", flush=True)
_run(f"df -h /content | tail -1 | awk '{{print \"     /content  — used: \"$3\"  free: \"$4\"  (\"$5\" full)\"}}'")
_run(f"du -sh {LOCAL_CACHE} | awk '{{print \"     Dataset   — \"$1}}'")

# ═══════════════════════════════════════════════════════════════════════════
# 5. Auto Hardware Detection + Configure
# ═══════════════════════════════════════════════════════════════════════════
_banner("5/8 — Auto-detecting hardware & configuring")

# Set environment variables BEFORE training script imports config.py
# UAV_PROJECT_DIR = where to SYNC results on Drive after training
# Training itself runs on local SSD (/content/runs/) for max I/O speed
os.environ["UAV_PROJECT_DIR"] = DRIVE_RUNS
os.environ["DRIVE_UPLOAD_DIR"] = DRIVE_UPLOAD

# --- CRITICAL TWEAK FOR A100 / DATALOADER BOTTLENECKS ---
# Prevent OpenCV and OpenMP from spawning threads inside DataLoader multiprocessing workers.
# 8 workers * 12 default threads = 96 threads fighting for 12 CPU cores = 1.3 it/s thrashing.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENCV_FOR_THREADS_NUM"] = "1"
# ---------------------------------------------------------

# Point YOLO's runs dir AND datasets dir to local SSD (NOT Drive)
_run('yolo settings runs_dir="/content/runs"', check=False)
_run('yolo settings datasets_dir="/content/datasets_local"', check=False)
print(f"  ✓ Training output → /content/runs/ (local SSD, max speed)", flush=True)
print(f"  ✓ Datasets dir → /content/datasets_local/ (local SSD)", flush=True)
print(f"  ✓ Post-training sync → {DRIVE_RUNS} (Google Drive)", flush=True)
print(f"  ✓ Model export → {DRIVE_UPLOAD}", flush=True)

# Full nvidia-smi output (visible in Colab)
print("\n  📊 GPU Status:", flush=True)
_run("nvidia-smi", check=False)

# Quick torch probe for verification
print("\n  📊 PyTorch GPU Probe:", flush=True)
try:
    import torch as _t
    print(f"  CUDA available: {_t.cuda.is_available()}", flush=True)
    if _t.cuda.is_available():
        _p = _t.cuda.get_device_properties(0)
        print(f"  GPU: {_p.name}  |  VRAM: {_p.total_memory / 1024**3:.1f} GB", flush=True)
    del _t, _p
except Exception as _e:
    print(f"  ⚠️ PyTorch probe failed: {_e}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Launch Training (with live log tee)
# ═══════════════════════════════════════════════════════════════════════════
_banner("6/8 — Starting training")

# Search for the latest checkpoint (Local SSD first, then Drive)
def find_latest_checkpoint(*dirs) -> str | None:
    for d in dirs:
        candidates = glob.glob(os.path.join(d, "**", "weights", "last.pt"), recursive=True)
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
    return None

checkpoint = find_latest_checkpoint("/content/runs", DRIVE_RUNS)

train_script_path = os.path.join(REPO_DIR, TRAIN_SCRIPT)
if not os.path.isfile(train_script_path):
    raise FileNotFoundError(
        f"Training script not found at {train_script_path}."
    )

# ── Prepare log file (Local SSD for max speed) ──
log_dir = "/content/logs"
os.makedirs(log_dir, exist_ok=True)
log_name = datetime.now().strftime("log_%Y-%m-%d_%H-%M.txt")
log_path = os.path.join(log_dir, log_name)

if checkpoint:
    print(f"  🔄 Resuming from checkpoint: {checkpoint}", flush=True)
    train_cmd = f'{sys.executable} -u "{train_script_path}" --resume'
else:
    print("  🆕 No checkpoint found — starting fresh training", flush=True)
    train_cmd = f'{sys.executable} -u "{train_script_path}"'

print(f"  ▶ Command: {train_cmd}", flush=True)
print(f"  📝 Live log: {log_path}", flush=True)
print(f"\n{'─'*60}", flush=True)

os.chdir(REPO_DIR)

# ── Real-time output forwarding with tee to log file ──
# Uses subprocess.Popen + os.read for TRUE real-time output in Colab.
# Every byte from train.py appears instantly in the cell AND in the log file.
proc = subprocess.Popen(
    train_cmd, shell=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    env={**os.environ, "PYTHONUNBUFFERED": "1"}
)

with open(log_path, 'wb') as lf:
    fd = proc.stdout.fileno()
    while True:
        data = os.read(fd, 8192)  # os.read returns partial reads → real-time
        if not data:
            break
        # Colab's sys.stdout is IPython OutStream (no .buffer attribute)
        # Decode bytes → text and use .write() which works everywhere
        sys.stdout.write(data.decode('utf-8', errors='replace'))
        sys.stdout.flush()
        lf.write(data)
        lf.flush()

exit_code = proc.wait()
print(f"\n{'─'*60}", flush=True)

if exit_code != 0:
    print(f"  ❌ Training exited with code {exit_code}", flush=True)
else:
    print(f"  ✅ Training completed successfully", flush=True)

print(f"  📝 Full log saved: {log_path}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
# 7. Post-Training Sync to Drive
# ═══════════════════════════════════════════════════════════════════════════
_banner("7/8 — Syncing outputs to Google Drive")

# Sync logs
drive_log_dir = os.path.join(DRIVE_UPLOAD, "logs")
os.makedirs(drive_log_dir, exist_ok=True)
print(f"  ☁️  Syncing logs to: {drive_log_dir}", flush=True)
_run(f'rsync -a "{log_dir}/" "{drive_log_dir}/"', check=False)

# Results are synced by train.py itself, but we can do a fallback sync here just in case
runs_dir = "/content/runs"
if os.path.exists(runs_dir):
    print(f"  ☁️  Syncing runs to: {DRIVE_RUNS}", flush=True)
    _run(f'rsync -a "{runs_dir}/" "{DRIVE_RUNS}/"', check=False)

# ═══════════════════════════════════════════════════════════════════════════
# 8. Post-Training Summary
# ═══════════════════════════════════════════════════════════════════════════
_banner("8/8 — Post-training summary")

print(f"\n  📁 Training outputs: {DRIVE_RUNS}", flush=True)
_run(f'du -sh "{DRIVE_RUNS}" 2>/dev/null | awk \'{{print "     Size: "$1}}\'', check=False)

# List exported model folders
models_dir = os.path.join(DRIVE_UPLOAD, "models")
if os.path.isdir(models_dir):
    model_folders = sorted([d for d in os.listdir(models_dir)
                            if os.path.isdir(os.path.join(models_dir, d))])
    if model_folders:
        print(f"\n  🏆 Exported models in {models_dir}:")
        for mf in model_folders[-5:]:  # Show last 5
            mf_path = os.path.join(models_dir, mf)
            n_files = len(os.listdir(mf_path))
            print(f"     📂 {mf} ({n_files} files)")

# List any best_*.pt files in upload dir (quick access)
best_files = glob.glob(os.path.join(DRIVE_UPLOAD, "best_*.pt"))
if best_files:
    print(f"\n  🎯 Quick access models in {DRIVE_UPLOAD}:")
    for bf in sorted(best_files):
        size_mb = os.path.getsize(bf) / (1024 * 1024)
        print(f"     → {os.path.basename(bf)} ({size_mb:.1f} MB)")

# List log files from Drive
log_files = sorted(glob.glob(os.path.join(drive_log_dir, "log_*.txt")))
if log_files:
    print(f"\n  📝 Training logs in {drive_log_dir}:")
    for lf in log_files[-5:]:  # Show last 5 logs
        size_kb = os.path.getsize(lf) / 1024
        print(f"     → {os.path.basename(lf)} ({size_kb:.0f} KB)")

_banner("✅ Training complete — all outputs saved to Google Drive")
