##############################################################################
# 🚀 YOLO UAV Training Bootstrap — Google Colab
# Paste this entire cell into Colab and run.
##############################################################################

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
    """Run a shell command with real-time output streaming to Colab cell."""
    sys.stdout.flush()
    sys.stderr.flush()
    return subprocess.run(cmd, shell=True, check=check, **kw)

def _banner(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Mount Google Drive
# ═══════════════════════════════════════════════════════════════════════════
_banner("1/7 — Mounting Google Drive")
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
_banner("2/7 — Setting up repository")
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
_banner("3/7 — Installing dependencies")

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
# 4. Dataset Extraction (Drive .tar.gz → Local SSD Cache)
# ═══════════════════════════════════════════════════════════════════════════
_banner("4/7 — Extracting dataset to local SSD (MAX SPEED)")

os.makedirs(LOCAL_CACHE, exist_ok=True)

# Install tools with visible output
print("  📦 Installing extraction tools …", flush=True)
_run("apt-get update -qq && apt-get install -y pigz mbuffer 2>&1 | tail -3", check=False)

# Marker = extraction fully completed previously
CACHE_MARKER = os.path.join(LOCAL_CACHE, ".done")
NCPU = os.cpu_count() or 2
HAS_MBUFFER = os.system("which mbuffer >/dev/null 2>&1") == 0

# Smart cache detection:
# 1. If .done marker exists → instant skip
# 2. If no marker but directory has >5000 files → treat as complete (user put data there manually)
# 3. If partial (100-5000 files) → incremental extraction
# 4. If empty → full speed extraction

existing = sum(len(f) for _, _, f in os.walk(LOCAL_CACHE))

t0 = time.time()

if os.path.isfile(CACHE_MARKER):
    # ═══ TIER 1: Fully cached — instant skip ═══
    print(f"  ⚡ Cache complete ({existing} files) — skipping extraction entirely", flush=True)

elif existing > 5000:
    # ═══ TIER 1b: No marker but lots of files — assume complete, create marker ═══
    print(f"  ⚡ Dataset already in local cache ({existing} files) — skipping extraction", flush=True)
    with open(CACHE_MARKER, 'w') as f:
        f.write("1")

else:
    # Common tar flags for maximum speed:
    TAR_FAST = "--no-same-owner --no-same-permissions -b 512"

    if existing > 100:
        # ═══ TIER 2: Partial cache — incremental, skip existing files ═══
        print(f"  ♻️  Partial cache ({existing} files) — incremental mode …", flush=True)
        extract_cmd = (
            f'pigz -d -c -p {NCPU} "{DRIVE_DATASET}" '
            f'| tar -xf - -C "{LOCAL_CACHE}" '
            f'{TAR_FAST} '
            f'--skip-old-files '
            f'--checkpoint=10000 '
            f'--checkpoint-action=echo="  %u dosya kontrol edildi…"'
        )
        _run(extract_cmd)
    else:
        # ═══ TIER 3: Fresh extraction — maximum speed pipeline ═══
        if HAS_MBUFFER:
            print(f"  🚀 Max-speed extraction → {LOCAL_CACHE}", flush=True)
            print(f"     Pipeline: Drive → mbuffer(64MB) → pigz({NCPU} cores) → tar(fast)")
            extract_cmd = (
                f'mbuffer -i "{DRIVE_DATASET}" -m 64M -q '
                f'| pigz -d -c -p {NCPU} '
                f'| tar -xf - -C "{LOCAL_CACHE}" '
                f'{TAR_FAST} '
                f'--checkpoint=10000 '
                f'--checkpoint-action=echo="  %u dosya çıkarıldı…"'
            )
        else:
            print(f"  🚀 Fast extraction → {LOCAL_CACHE} (pigz {NCPU} cores)", flush=True)
            extract_cmd = (
                f'pigz -d -c -p {NCPU} "{DRIVE_DATASET}" '
                f'| tar -xf - -C "{LOCAL_CACHE}" '
                f'{TAR_FAST} '
                f'--checkpoint=10000 '
                f'--checkpoint-action=echo="  %u dosya çıkarıldı…"'
            )
        _run(extract_cmd)

    # Write marker = fully extracted
    with open(CACHE_MARKER, 'w') as f:
        f.write("1")

elapsed = time.time() - t0
final_count = sum(len(f) for _, _, f in os.walk(LOCAL_CACHE))
print(f"\n  ✓ Done in {elapsed:.1f}s — {final_count} files ready", flush=True)

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
_banner("5/7 — Auto-detecting hardware & configuring")

# Set environment variables BEFORE training script imports config.py
os.environ["UAV_PROJECT_DIR"] = DRIVE_RUNS
os.environ["DRIVE_UPLOAD_DIR"] = DRIVE_UPLOAD

_run(f'yolo settings runs_dir="{DRIVE_RUNS}"', check=False)
print(f"  ✓ UAV_PROJECT_DIR → {DRIVE_RUNS}", flush=True)
print(f"  ✓ DRIVE_UPLOAD_DIR → {DRIVE_UPLOAD}")

# Full nvidia-smi output (visible in Colab)
print("\n  📊 GPU Status:", flush=True)
_run("nvidia-smi", check=False)

# Quick torch probe for verification
print("\n  📊 PyTorch GPU Probe:", flush=True)
_run(f'{sys.executable} -c "'
     'import torch; '
     'print(f\"  CUDA available: {{torch.cuda.is_available()}}\"); '
     'print(f\"  GPU: {{torch.cuda.get_device_name(0)}}\") if torch.cuda.is_available() else None; '
     'print(f\"  VRAM: {{torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f}} GB\") if torch.cuda.is_available() else None'
     '"', check=False)

# ═══════════════════════════════════════════════════════════════════════════
# 6. Launch Training (with live log tee)
# ═══════════════════════════════════════════════════════════════════════════
_banner("6/7 — Starting training")

# Search for the latest checkpoint in Drive runs
def find_latest_checkpoint(runs_dir: str) -> str | None:
    candidates = glob.glob(os.path.join(runs_dir, "**", "weights", "last.pt"), recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

checkpoint = find_latest_checkpoint(DRIVE_RUNS)

train_script_path = os.path.join(REPO_DIR, TRAIN_SCRIPT)
if not os.path.isfile(train_script_path):
    raise FileNotFoundError(
        f"Training script not found at {train_script_path}."
    )

# ── Prepare log file ──
log_dir = os.path.join(DRIVE_UPLOAD, "logs")
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
# 7. Post-Training Summary
# ═══════════════════════════════════════════════════════════════════════════
_banner("7/7 — Post-training summary")

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

# List log files
log_files = sorted(glob.glob(os.path.join(log_dir, "log_*.txt")))
if log_files:
    print(f"\n  📝 Training logs in {log_dir}:")
    for lf in log_files[-5:]:  # Show last 5 logs
        size_kb = os.path.getsize(lf) / 1024
        print(f"     → {os.path.basename(lf)} ({size_kb:.0f} KB)")

_banner("✅ Training complete — outputs saved to Google Drive")
