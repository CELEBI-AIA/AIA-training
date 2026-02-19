##############################################################################
# 🚀 YOLO UAV Training Bootstrap — Google Colab
# Paste this entire cell into Colab and run.
##############################################################################

# ── Configuration ───────────────────────────────────────────────────────────
REPO_URL      = "https://github.com/YOUR_USER/YOUR_REPO.git"   # ← set this
REPO_BRANCH   = "main"                                          # ← set this
DRIVE_DATASET = "/content/drive/MyDrive/AIA/datasets"
LOCAL_CACHE   = "/content/datasets_local"
DRIVE_RUNS    = "/content/drive/MyDrive/AIA/runs"
TRAIN_SCRIPT  = "uav_training/train.py"
# ────────────────────────────────────────────────────────────────────────────

import subprocess, sys, os, pathlib, glob, textwrap

def _run(cmd: str, *, check: bool = True, shell: bool = True, **kw):
    """Run a shell command, stream output, and optionally check return code."""
    r = subprocess.run(cmd, shell=shell, check=check, **kw)
    return r

def _banner(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. Mount Google Drive
# ═══════════════════════════════════════════════════════════════════════════
_banner("1/6 — Mounting Google Drive")
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

if not os.path.isdir(DRIVE_DATASET):
    raise FileNotFoundError(
        f"Dataset not found at {DRIVE_DATASET}. "
        "Upload your dataset to Google Drive first."
    )

os.makedirs(DRIVE_RUNS, exist_ok=True)
print(f"  ✓ Drive mounted — dataset found at {DRIVE_DATASET}")
print(f"  ✓ Runs directory ready at {DRIVE_RUNS}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Clone or Pull Repository
# ═══════════════════════════════════════════════════════════════════════════
_banner("2/6 — Setting up repository")
REPO_DIR = "/content/repo"

if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    print("  ↻ Repo exists — pulling latest changes …")
    _run(f"git -C {REPO_DIR} fetch --all --quiet")
    _run(f"git -C {REPO_DIR} reset --hard origin/{REPO_BRANCH}")
else:
    print(f"  ↓ Cloning {REPO_URL} …")
    _run(f"git clone --depth 1 -b {REPO_BRANCH} {REPO_URL} {REPO_DIR}")

print("  ✓ Repo ready")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Install Requirements
# ═══════════════════════════════════════════════════════════════════════════
_banner("3/6 — Installing dependencies")
req_file = os.path.join(REPO_DIR, "requirements.txt")
if os.path.isfile(req_file):
    _run(f"{sys.executable} -m pip install -q -r {req_file}")
    print("  ✓ requirements.txt installed")
else:
    print("  ⚠ No requirements.txt found — skipping")

# Ensure ultralytics is available (needed for runs_dir config)
_run(f"{sys.executable} -m pip install -q ultralytics", check=False)
print("  ✓ ultralytics verified")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Dataset Streaming (Drive → Local SSD Cache)
# ═══════════════════════════════════════════════════════════════════════════
_banner("4/6 — Preparing dataset cache")

# Install pv if missing (needed for progress display)
_run("which pv >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq pv > /dev/null 2>&1)")

if os.path.isdir(LOCAL_CACHE) and os.listdir(LOCAL_CACHE):
    print(f"  ✓ Local cache already exists at {LOCAL_CACHE} — skipping copy")
else:
    os.makedirs(LOCAL_CACHE, exist_ok=True)

    # Measure source size for progress bar
    print("  📏 Measuring dataset size …")
    du_result = subprocess.run(
        f'du -sb "{DRIVE_DATASET}" | cut -f1',
        shell=True, capture_output=True, text=True
    )
    total_bytes = du_result.stdout.strip()

    print(f"  📦 Dataset size: {int(total_bytes) / (1024**3):.2f} GB")
    print(f"  🚀 Streaming dataset → {LOCAL_CACHE} …\n")

    # Stream-copy using tar + pv (single-line progress, no rsync)
    pv_size_flag = f"-s {total_bytes}" if total_bytes.isdigit() else ""
    stream_cmd = (
        f'tar -C "{DRIVE_DATASET}" -cf - . '
        f'| pv -f -pterb {pv_size_flag} '
        f'| tar -C "{LOCAL_CACHE}" -xf -'
    )
    _run(stream_cmd)
    print("\n  ✓ Dataset streaming complete")

# Symlink repo's datasets/ → local cache so training scripts find it
repo_datasets = os.path.join(REPO_DIR, "datasets")
if os.path.islink(repo_datasets) or os.path.isdir(repo_datasets):
    _run(f'rm -rf "{repo_datasets}"')
os.symlink(LOCAL_CACHE, repo_datasets)
print(f"  🔗 Symlinked {repo_datasets} → {LOCAL_CACHE}")

# Disk usage report
print("\n  📊 Disk usage:")
_run(f'df -h /content | tail -1 | awk \'{{print "     /content  — used: "$3"  free: "$4"  ("$5" full)"}}\'')
_run(f'du -sh "{LOCAL_CACHE}" 2>/dev/null | awk \'{{print "     Cache     — "$1}}\'')

# ═══════════════════════════════════════════════════════════════════════════
# 5. Configure Ultralytics Output → Drive
# ═══════════════════════════════════════════════════════════════════════════
_banner("5/6 — Configuring output directories")

_run(f'yolo settings runs_dir="{DRIVE_RUNS}"', check=False)
os.environ["UAV_PROJECT_DIR"] = DRIVE_RUNS
print(f"  ✓ Ultralytics runs_dir → {DRIVE_RUNS}")
print(f"  ✓ UAV_PROJECT_DIR → {DRIVE_RUNS}")

# ═══════════════════════════════════════════════════════════════════════════
# 6. Launch Training (with automatic resume)
# ═══════════════════════════════════════════════════════════════════════════
_banner("6/6 — Starting training")

# Search for the latest checkpoint in Drive runs
def find_latest_checkpoint(runs_dir: str) -> str | None:
    """Walk the runs directory and return the most recent last.pt by mtime."""
    candidates = glob.glob(os.path.join(runs_dir, "**", "weights", "last.pt"), recursive=True)
    if not candidates:
        return None
    # Sort by modification time, newest first
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
    print(f"  🔄 Resuming from checkpoint: {checkpoint}")
    train_cmd = f'{sys.executable} "{train_script_path}" --resume'
else:
    print("  🆕 No checkpoint found — starting fresh training")
    train_cmd = f'{sys.executable} "{train_script_path}"'

print(f"  ▶ Command: {train_cmd}\n")
print("─" * 60)

os.chdir(REPO_DIR)
_run(train_cmd)

_banner("✅ Training complete — outputs saved to Google Drive")
_run(f'du -sh "{DRIVE_RUNS}" | awk \'{{print "  📁 Runs directory: "$1}}\'')
