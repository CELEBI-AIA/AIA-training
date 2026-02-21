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

# ── Force unbuffered output so ALL print/log lines appear in Colab instantly ──
os.environ["PYTHONUNBUFFERED"] = "1"

def _run(cmd: str, *, check: bool = True, **kw):
    """Run a shell command with real-time output streaming to Colab cell."""
    # Flush Python's own output first so banners/prints appear BEFORE subprocess
    sys.stdout.flush()
    sys.stderr.flush()
    r = subprocess.run(cmd, shell=True, check=check, **kw)
    return r

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
print(f"  ✓ Drive mounted — dataset archive found at {DRIVE_DATASET}", flush=True)
print(f"  ✓ Runs directory ready at {DRIVE_RUNS}")
print(f"  ✓ Upload directory ready at {DRIVE_UPLOAD}")

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

print("  ✓ Repo ready")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Install Requirements
# ═══════════════════════════════════════════════════════════════════════════
_banner("3/7 — Installing dependencies")

# Show pip output in real-time (removed -q flag for visibility)
req_file = os.path.join(REPO_DIR, "requirements.txt")
if os.path.isfile(req_file):
    print("  📦 Installing requirements.txt …", flush=True)
    _run(f"{sys.executable} -m pip install --progress-bar on -r {req_file}")
    print("  ✓ requirements.txt installed")
else:
    print("  ⚠ No requirements.txt found — skipping")

# Ensure ultralytics + psutil are available
print("  📦 Verifying ultralytics & psutil …", flush=True)
_run(f"{sys.executable} -m pip install --progress-bar on ultralytics psutil", check=False)
print("  ✓ ultralytics & psutil verified")

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

# Get CPU count for pigz thread tuning
NCPU = os.cpu_count() or 2

# Check if mbuffer is actually available
HAS_MBUFFER = os.system("which mbuffer >/dev/null 2>&1") == 0

t0 = time.time()

if os.path.isfile(CACHE_MARKER):
    # ═══ TIER 1: Fully cached — instant skip (0 seconds) ═══
    existing = sum(len(f) for _, _, f in os.walk(LOCAL_CACHE))
    print(f"  ⚡ Cache complete ({existing} files) — skipping extraction entirely", flush=True)

else:
    # Count existing files
    existing = sum(len(f) for _, _, f in os.walk(LOCAL_CACHE))

    # Common tar flags for maximum speed:
    #   --no-same-owner        → skip chown() per file (saves 1 syscall/file)
    #   --no-same-permissions  → skip chmod() per file (saves 1 syscall/file)
    #   -b 512                 → 256KB read blocks (fewer read syscalls)
    #   For 100K+ tiny files these syscall savings add up to minutes
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
        # Pipeline: Drive → mbuffer(64MB) → pigz(all cores) → tar(fast flags)
        #
        # mbuffer smooths Drive's bursty network I/O with a 64MB RAM buffer
        # so pigz never stalls waiting for data. This alone can save 20-30%.
        #
        # pigz -p N uses all N cores for parallel decompression.
        # tar with fast flags skips per-file chown/chmod syscalls.
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
            # Fallback: pigz without mbuffer (still fast)
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

# Quick count after extraction
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
print(f"  ✓ UAV_PROJECT_DIR → {DRIVE_RUNS}")
print(f"  ✓ DRIVE_UPLOAD_DIR → {DRIVE_UPLOAD}")

# Print GPU info (visible in Colab)
print("\n  📊 GPU Info:", flush=True)
_run("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader", check=False)

# ═══════════════════════════════════════════════════════════════════════════
# 6. Launch Training (with automatic resume)
# ═══════════════════════════════════════════════════════════════════════════
_banner("6/7 — Starting training")

# Search for the latest checkpoint in Drive runs
def find_latest_checkpoint(runs_dir: str) -> str | None:
    """Walk the runs directory and return the most recent last.pt by mtime."""
    candidates = glob.glob(os.path.join(runs_dir, "**", "weights", "last.pt"), recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

checkpoint = find_latest_checkpoint(DRIVE_RUNS)

train_script_path = os.path.join(REPO_DIR, TRAIN_SCRIPT)
if not os.path.isfile(train_script_path):
    raise FileNotFoundError(
        f"Training script not found at {train_script_path}. "
        f"Check TRAIN_SCRIPT config (currently: '{TRAIN_SCRIPT}')."
    )

if checkpoint:
    print(f"  🔄 Resuming from checkpoint: {checkpoint}", flush=True)
    train_cmd = f'{sys.executable} -u "{train_script_path}" --resume'
else:
    print("  🆕 No checkpoint found — starting fresh training", flush=True)
    train_cmd = f'{sys.executable} -u "{train_script_path}"'

print(f"  ▶ Command: {train_cmd}\n")
print("─" * 60, flush=True)

os.chdir(REPO_DIR)
_run(train_cmd)

# ═══════════════════════════════════════════════════════════════════════════
# 7. Post-Training Summary
# ═══════════════════════════════════════════════════════════════════════════
_banner("7/7 — Post-training summary")

# The train.py script handles best.pt renaming and Drive upload automatically
# via DRIVE_UPLOAD_DIR environment variable.

print(f"\n  📁 Training outputs: {DRIVE_RUNS}", flush=True)
_run(f'du -sh "{DRIVE_RUNS}" | awk \'{{print "     Size: "$1}}\'')

# List any best_*.pt files in upload dir
best_files = glob.glob(os.path.join(DRIVE_UPLOAD, "best_*.pt"))
if best_files:
    print(f"\n  🏆 Uploaded models in {DRIVE_UPLOAD}:")
    for bf in sorted(best_files):
        size_mb = os.path.getsize(bf) / (1024 * 1024)
        print(f"     → {os.path.basename(bf)} ({size_mb:.1f} MB)")
else:
    print(f"\n  ⚠️ No renamed best.pt found in {DRIVE_UPLOAD}")

_banner("✅ Training complete — outputs saved to Google Drive")
