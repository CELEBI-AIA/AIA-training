##############################################################################
# 🚀 YOLO UAV Training Bootstrap — Google Colab
# Paste this entire cell into Colab and run.
##############################################################################

VERSION = "dev"

# ── Configuration ───────────────────────────────────────────────────────────
REPO_URL       = "https://github.com/CELEBI-AIA/AIA-training.git"
REPO_BRANCH    = "main"
DRIVE_DATASET  = "/content/drive/MyDrive/AIA/datasets.tar.gz"
LOCAL_CACHE    = "/content/datasets_local"
DRIVE_RUNS     = "/content/drive/MyDrive/AIA/runs"
DRIVE_UPLOAD   = "/content/drive/MyDrive/AIA"     # best.pt upload destination
TRAIN_SCRIPT   = "uav_training/train.py"
TWO_PHASE_TRAINING = True  # Run 50+15 optimized profile by default (M-01)
FORCE_FRESH_START  = False # Ignore existing checkpoints and start fresh (B-05)
# ────────────────────────────────────────────────────────────────────────────

import subprocess, sys, os, glob, shutil, time, threading, shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


def _read_repo_version(repo_dir: str) -> str:
    """Read module version from uav_training/__init__.py if available."""
    version_file = os.path.join(repo_dir, "uav_training", "__init__.py")
    if not os.path.isfile(version_file):
        return "dev"

    try:
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("__version__"):
                    _, value = line.split("=", 1)
                    return value.strip().strip("\"'")
    except Exception:
        pass

    return "dev"

# ── Force unbuffered output → ALL print/log lines appear in Colab INSTANTLY ──
os.environ["PYTHONUNBUFFERED"] = "1"

def _run(cmd: str, *, check: bool = True, print_output: bool = True, **kw):
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
    if print_output and r.stdout:
        sys.stdout.write(r.stdout)
        sys.stdout.flush()
    return r


def _resolve_drive_path(path: str) -> str:
    """Resolve dangling MyDrive shortcuts to actual target path when available."""
    if os.path.exists(path):
        return path
    if os.path.islink(path):
        target = os.path.realpath(path)
        if os.path.exists(target):
            print(f"  🔗 Resolved shortcut path: {path} -> {target}", flush=True)
            return target
    return path

def _banner(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}", flush=True)

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

DRIVE_UPLOAD = _resolve_drive_path(DRIVE_UPLOAD)
DRIVE_RUNS = _resolve_drive_path(DRIVE_RUNS)
DRIVE_DATASET = _resolve_drive_path(DRIVE_DATASET)

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
    _run(f"git -C {shlex.quote(REPO_DIR)} fetch --all")
    _run(f"git -C {shlex.quote(REPO_DIR)} reset --hard origin/{shlex.quote(REPO_BRANCH)}")
else:
    print(f"  ↓ Cloning {REPO_URL} …", flush=True)
    _run(f"git clone --depth 1 -b {shlex.quote(REPO_BRANCH)} {shlex.quote(REPO_URL)} {shlex.quote(REPO_DIR)}")

# Show repo info
_run(f"git -C {REPO_DIR} log --oneline -1")
VERSION = _read_repo_version(REPO_DIR)
print(f"\n🛰️  UAV Training Bootstrap v{VERSION}", flush=True)
print(f"    Repo: {REPO_URL} ({REPO_BRANCH})", flush=True)
print(f"  ✓ Bootstrap version synced from repo: v{VERSION}", flush=True)
print("  ✓ Repo ready", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
# 3. Install Requirements
# ═══════════════════════════════════════════════════════════════════════════
_banner("3/8 — Installing dependencies")

req_file = os.path.join(REPO_DIR, "requirements.txt")
if os.path.isfile(req_file):
    print("  📦 Installing requirements.txt …", flush=True)
    install_result = _run(
        f"{sys.executable} -m pip install --disable-pip-version-check --progress-bar off -r {shlex.quote(req_file)}",
        print_output=False,
    )

    install_out = install_result.stdout or ""
    updated_line = ""
    for line in install_out.splitlines():
        if line.startswith("Successfully installed "):
            updated_line = line.replace("Successfully installed ", "").strip()
            break

    already_satisfied_count = sum(
        1 for line in install_out.splitlines() if "Requirement already satisfied:" in line
    )

    if updated_line:
        print(f"  ✓ Updated packages: {updated_line}", flush=True)
    elif already_satisfied_count > 0:
        print(f"  ✓ No package updates needed ({already_satisfied_count} requirements already satisfied)", flush=True)
    else:
        print("  ✓ Dependency check completed", flush=True)
else:
    print("  ⚠ No requirements.txt found")

# Ensure ultralytics + psutil are available
print("  📦 Verifying ultralytics & psutil …", flush=True)
# ultralytics & psutil already in requirements.txt — no duplicate install needed

# Print key package versions
print("\n  📊 Package versions:", flush=True)
_run(f'{sys.executable} -c "import ultralytics; print(f\'     ultralytics: {{ultralytics.__version__}}\')"', check=False)
_run(f'{sys.executable} -c "import torch; print(f\'     torch: {{torch.__version__}}, CUDA: {{torch.version.cuda}}\')"', check=False)
print("  ✓ Dependencies installed", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
# 4. Auto Hardware Detection + Configure (BEFORE dataset download!)
# ═══════════════════════════════════════════════════════════════════════════
_banner("4/8 — Auto-detecting hardware & configuring")

# ── GPU/TPU Check — abort early if hardware is incompatible ──
# YOLO requires CUDA GPU. If TPU is selected, stop NOW before wasting
# time downloading 78 GB of data that can't be used.
print("\n  📊 GPU/TPU Probe:", flush=True)
_is_tpu = os.environ.get("TPU_NAME") is not None or \
          os.environ.get("COLAB_TPU_ADDR") is not None or \
          os.path.exists("/dev/accel0")

try:
    import torch as _t
    _has_cuda = _t.cuda.is_available()
    print(f"  CUDA available: {_has_cuda}", flush=True)
    if _has_cuda:
        _p = _t.cuda.get_device_properties(0)
        print(f"  GPU: {_p.name}  |  VRAM: {_p.total_memory / 1024**3:.1f} GB", flush=True)
        _cc = _t.cuda.get_device_capability(0)
        print(f"  Compute capability: {_cc[0]}.{_cc[1]}", flush=True)
        if _cc[0] >= 8:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            print(f"  Ampere+ (sm_{_cc[0]}{_cc[1]}): native AMP BF16, expandable VRAM segments enabled", flush=True)
        del _p, _cc
    del _t
except Exception as _e:
    _has_cuda = False
    print(f"  ⚠️ PyTorch probe failed: {_e}", flush=True)

if _is_tpu and not _has_cuda:
    print("\n" + "!" * 60, flush=True)
    print("  ❌  TPU RUNTIME — YOLO EĞİTİMİ İÇİN UYGUN DEĞİL!")
    print("  YOLO/Ultralytics CUDA (GPU) gerektirir, TPU/XLA desteklemez.")
    print("  Lütfen runtime'ı GPU'ya çevirin:")
    print("     Runtime → Change runtime type → GPU (T4/L4/A100/H100)")
    print("!" * 60, flush=True)
    raise RuntimeError(
        "TPU runtime detected. YOLO requires a CUDA GPU. "
        "Please change to: Runtime → Change runtime type → GPU"
    )

if not _has_cuda:
    print("\n  ⚠️  CUDA GPU bulunamadı — CPU ile eğitim çok yavaş olacak!", flush=True)

# Full nvidia-smi output
print("\n  📊 GPU Status:", flush=True)
_run("nvidia-smi", check=False)

# Set environment variables BEFORE training script imports config.py
os.environ["UAV_PROJECT_DIR"] = DRIVE_RUNS
os.environ["DRIVE_UPLOAD_DIR"] = DRIVE_UPLOAD

# --- DataLoader thread limiter ---
cpu_count = os.cpu_count() or 8
data_threads = max(8, min(12, cpu_count))
intra_threads = max(4, data_threads // 2)
interop_threads = 2

os.environ["OMP_NUM_THREADS"] = str(intra_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(intra_threads)
os.environ["MKL_NUM_THREADS"] = str(max(2, data_threads // 4))
os.environ["VECLIB_MAXIMUM_THREADS"] = str(max(2, data_threads // 4))
os.environ["NUMEXPR_NUM_THREADS"] = str(max(2, data_threads // 4))
os.environ["OPENCV_FOR_THREADS_NUM"] = str(max(2, data_threads // 4))

# Force inter/intra op threads for PyTorch 
_run(
    f'{sys.executable} -c "import torch; torch.set_num_threads({intra_threads}); torch.set_num_interop_threads({interop_threads})"',
    check=False,
)

# Point YOLO settings to local SSD
_run('yolo settings runs_dir="/content/runs"', check=False)
_run('yolo settings datasets_dir="/content/datasets_local"', check=False)
print(f"  ✓ Training output → /content/runs/ (local SSD)", flush=True)
print(f"  ✓ Datasets dir → /content/datasets_local/ (local SSD)", flush=True)
print(f"  ✓ Post-training sync → {DRIVE_RUNS} (Google Drive)", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
# 5. Dataset: Download tar.gz → Local SSD → Extract → Verify → Cleanup
# ═══════════════════════════════════════════════════════════════════════════
_banner("5/8 — Dataset to local SSD (DOWNLOAD → EXTRACT → VERIFY)")

os.makedirs(LOCAL_CACHE, exist_ok=True)

CACHE_MARKER = os.path.join(LOCAL_CACHE, ".done")
NCPU = os.cpu_count() or 2
LOCAL_TAR = "/content/datasets.tar.gz"
t0 = time.time()
existing = sum(len(f) for _, _, f in os.walk(LOCAL_CACHE))
dataset_yaml = os.path.join(LOCAL_CACHE, "dataset.yaml")

# Fast path: if dataset is already materialized on local SSD, skip tool install
# and all Drive→SSD copy/extract work.
if os.path.isfile(dataset_yaml) and existing > 5000 and os.path.isfile(CACHE_MARKER):
    print(f"  ⚡ Local SSD dataset already ready ({existing} files) — SKIP copy/extract", flush=True)
else:
    # Install extraction tools only when download/extract may be needed.
    print("  📦 Installing extraction tools …", flush=True)
    _run("apt-get update -qq && apt-get install -y pigz pv eatmydata 2>&1 | tail -3", check=False)

    if os.path.isfile(CACHE_MARKER) and existing > 5000:
        print(f"  ⚡ Cache complete ({existing} files) — SKIP", flush=True)

    else:
        # ── Disk cleanup before download ──
        # Colab SSD is ~200GB. We need ~78GB for tar.gz + ~78GB for extracted files.
        # Clean up leftovers from previous runs to prevent disk overflow.
        if os.path.isfile(LOCAL_TAR):
            _old_tar_gb = os.path.getsize(LOCAL_TAR) / (1024**3)
            os.remove(LOCAL_TAR)
            print(f"  🗑️  Removed old tar.gz — freed {_old_tar_gb:.1f} GB", flush=True)
        if existing > 0 and existing <= 5000:
            # Partial/broken old extraction — remove it
            import shutil as _shutil
            _shutil.rmtree(LOCAL_CACHE, ignore_errors=True)
            os.makedirs(LOCAL_CACHE, exist_ok=True)
            existing = 0
            print(f"  🗑️  Removed incomplete old dataset", flush=True)

        # Show disk space before we start
        _run(f"df -h /content | tail -1 | awk '{{print \"  📊 Disk: used \"$3\"  free \"$4\"  (\"$5\" full)\"}}'")

        # ── PHASE 1: Download tar.gz from Drive → Local SSD ──
        # Uses Python file I/O with real-time progress — dd/pv output is
        # invisible in Colab because it goes to stderr which Colab swallows.
        tar_size_bytes = os.path.getsize(DRIVE_DATASET)
        tar_size_gb = tar_size_bytes / (1024**3)

        _need_download = True
        if os.path.isfile(LOCAL_TAR):
            if os.path.getsize(LOCAL_TAR) == tar_size_bytes:
                print(f"  ⚡ tar.gz already on local SSD ({tar_size_gb:.1f} GB) — skip download", flush=True)
                _need_download = False
            else:
                print(f"  ♻️  Partial file — re-downloading", flush=True)
                os.remove(LOCAL_TAR)

        if _need_download:
            N_WORKERS = max(1, min(8, int(os.environ.get("UAV_DOWNLOAD_WORKERS", "4"))))
            print(f"  📥 Downloading {tar_size_gb:.1f} GB → local SSD ({N_WORKERS} workers) …", flush=True)
            print(f"     {DRIVE_DATASET} → {LOCAL_TAR}", flush=True)
            CHUNK = 8 * 1024 * 1024  # 8 MB — smaller to reduce lock contention & memory
            chunk_size = (tar_size_bytes + N_WORKERS - 1) // N_WORKERS
            _dl_start = time.time()
            _copied_lock = threading.Lock()
            _copied = [0]
            _done = threading.Event()

            def _copy_range(start: int, end: int) -> int:
                n = 0
                fd_dst = os.open(LOCAL_TAR, os.O_RDWR)
                try:
                    with open(DRIVE_DATASET, "rb") as src:
                        src.seek(start)
                        remaining = end - start
                        current_offset = start
                        while remaining > 0:
                            to_read = min(CHUNK, remaining)
                            buf = src.read(to_read)
                            if not buf:
                                break
                            os.pwrite(fd_dst, buf, current_offset)
                            n += len(buf)
                            remaining -= len(buf)
                            current_offset += len(buf)
                            with _copied_lock:
                                _copied[0] += len(buf)
                finally:
                    os.close(fd_dst)
                return n

            def _progress_loop():
                while not _done.is_set():
                    time.sleep(2.0)
                    with _copied_lock:
                        c = _copied[0]
                    if c >= tar_size_bytes:
                        break
                    elapsed = time.time() - _dl_start
                    pct = c / tar_size_bytes * 100
                    speed_mb = (c / (1024**2)) / max(elapsed, 0.1)
                    remaining_sec = (tar_size_bytes - c) / max(c / max(elapsed, 0.1), 1)
                    mins, secs = divmod(int(remaining_sec), 60)
                    sys.stdout.write(
                        f"\r  📥  {pct:5.1f}%  |  "
                        f"{c/(1024**3):.1f}/{tar_size_gb:.1f} GB  |  "
                        f"{speed_mb:.0f} MB/s  |  "
                        f"ETA {mins}m {secs}s   "
                    )
                    sys.stdout.flush()

            # Pre-allocate output file
            with open(LOCAL_TAR, "wb") as f:
                f.truncate(tar_size_bytes)

            _prog_thread = threading.Thread(target=_progress_loop, daemon=True)
            _prog_thread.start()

            try:
                with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
                    futures = [
                        ex.submit(_copy_range, i * chunk_size, min((i + 1) * chunk_size, tar_size_bytes))
                        for i in range(N_WORKERS)
                        if i * chunk_size < tar_size_bytes
                    ]
                    for _ in as_completed(futures):
                        pass
            finally:
                _done.set()
                _prog_thread.join(timeout=3)

            # Final progress line
            with _copied_lock:
                c = _copied[0]
            elapsed = time.time() - _dl_start
            pct = c / tar_size_bytes * 100
            speed_mb = (c / (1024**2)) / max(elapsed, 0.1)
            sys.stdout.write(
                f"\r  📥  {pct:5.1f}%  |  "
                f"{c/(1024**3):.1f}/{tar_size_gb:.1f} GB  |  "
                f"{speed_mb:.0f} MB/s  |  done      \n"
            )
            sys.stdout.flush()

        dl_elapsed = time.time() - t0
        print(f"  ✓ Download done in {dl_elapsed:.0f}s ({tar_size_gb/max(dl_elapsed,1)*1024:.0f} MB/s)", flush=True)

        # ── PHASE 2: Extract from local copy (full NVMe speed) ──
        # stdbuf: 64MB pipe buffer for pigz→tar throughput
        # eatmydata: skip fsync for faster writes (safe to re-extract on crash)
        # pigz -p NCPU: parallel decompression
        t1 = time.time()
        TAR_FAST = "--no-same-owner --no-same-permissions -b 512"
        PIPE_BUF = "64M"
        CKPT = 50000  # less frequent progress = less overhead
        _eatmydata = "eatmydata " if shutil.which("eatmydata") else ""

        if existing > 100:
            print(f"  ♻️  Incremental extraction ({existing} existing files) …", flush=True)
            _ext_cmd = (
                f'stdbuf -o {PIPE_BUF} pigz -d -c -p {NCPU} {shlex.quote(LOCAL_TAR)} '
                f'| stdbuf -i {PIPE_BUF} {_eatmydata}tar -xf - -C {shlex.quote(LOCAL_CACHE)} '
                f'{TAR_FAST} --skip-old-files '
                f'--checkpoint={CKPT} --checkpoint-action=echo="  %u files checked…"'
            )
        else:
            _ed = "eatmydata, " if _eatmydata else ""
            print(f"  🚀 Full extraction → {LOCAL_CACHE} ({_ed}pigz {NCPU} cores, 64MB pipe)", flush=True)
            _ext_cmd = (
                f'stdbuf -o {PIPE_BUF} pigz -d -c -p {NCPU} {shlex.quote(LOCAL_TAR)} '
                f'| stdbuf -i {PIPE_BUF} {_eatmydata}tar -xf - -C {shlex.quote(LOCAL_CACHE)} '
                f'{TAR_FAST} '
                f'--checkpoint={CKPT} --checkpoint-action=echo="  %u files extracted…"'
            )

        # Stream extraction output in real-time (not buffered like _run)
        _ext_proc = subprocess.Popen(
            _ext_cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        _ext_fd = _ext_proc.stdout.fileno()
        while True:
            _chunk = os.read(_ext_fd, 4096)
            if not _chunk:
                break
            sys.stdout.write(_chunk.decode('utf-8', errors='replace'))
            sys.stdout.flush()
        _ext_proc.wait()

        ext_elapsed = time.time() - t1
        final_count = sum(len(f) for _, _, f in os.walk(LOCAL_CACHE))
        print(f"  ✓ Extracted in {ext_elapsed:.0f}s — {final_count} files", flush=True)

        # ── PHASE 3: Always delete tar.gz, then verify ──
        # Must free ~78GB immediately — disk can't hold tar.gz + extracted files.
        # Drive still has the original if we need to retry.
        if os.path.isfile(LOCAL_TAR):
            os.remove(LOCAL_TAR)
            print(f"  🗑️  Deleted local tar.gz — freed {tar_size_gb:.1f} GB", flush=True)

        if final_count > 5000:
            print(f"  ✅ Verification passed: {final_count} files on local SSD", flush=True)
            with open(CACHE_MARKER, 'w') as f:
                f.write("1")
        else:
            print(f"  ⚠️  Only {final_count} files extracted — re-run to retry from Drive", flush=True)

elapsed = time.time() - t0
final_count = sum(len(f) for _, _, f in os.walk(LOCAL_CACHE))
print(f"\n  ✓ Total time: {elapsed:.0f}s — {final_count} files ready on SSD", flush=True)

# Symlink repo's datasets/ → local cache so training scripts find it
repo_datasets = os.path.join(REPO_DIR, "datasets")
if os.path.islink(repo_datasets) or os.path.isdir(repo_datasets):
    _run(f'rm -rf {shlex.quote(repo_datasets)}')
os.symlink(LOCAL_CACHE, repo_datasets)
print(f"  🔗 Symlinked {repo_datasets} → {LOCAL_CACHE}")

# Disk usage report
print("\n  📊 Disk usage:", flush=True)
_run(f"df -h /content | tail -1 | awk '{{print \"     /content  — used: \"$3\"  free: \"$4\"  (\"$5\" full)\"}}'")
_run(f"du -sh {LOCAL_CACHE} | awk '{{print \"     Dataset   — \"$1}}'")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Launch Training (with live log tee)
# ═══════════════════════════════════════════════════════════════════════════
_banner("6/8 — Starting training")

# Search for the latest checkpoint (Local SSD first, then Drive)
def find_latest_checkpoint(*dirs) -> str | None:
    for d in dirs:
        candidates = glob.glob(os.path.join(d, "**", "weights", "last.pt"), recursive=True)
        # Filter for the exact version
        candidates = [c for c in candidates if f"v{VERSION}" in c]
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
log_name = f"log_v{VERSION}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
log_path = os.path.join(log_dir, log_name)

if FORCE_FRESH_START:
    print("  ⚠️ FORCE_FRESH_START enabled - ignoring existing checkpoints.", flush=True)
    checkpoint = None

if checkpoint:
    print(f"  🔄 Resuming from checkpoint: {checkpoint}", flush=True)
    train_cmd = [
        sys.executable, "-u", train_script_path,
        "--model", checkpoint,
        "--resume",
    ]
else:
    print("  🆕 No checkpoint found or fresh start forced — starting fresh training", flush=True)
    train_cmd = [sys.executable, "-u", train_script_path]
    if TWO_PHASE_TRAINING:
        train_cmd.append("--two-phase")

print(f"  ▶ Command: {' '.join(train_cmd)}", flush=True)
print(f"  📝 Live log: {log_path}", flush=True)
print(f"\n{'─'*60}", flush=True)

os.chdir(REPO_DIR)

# ── Real-time output forwarding with tee to log file ──
# Uses subprocess.Popen + os.read for TRUE real-time output in Colab.
# Every byte from train.py appears instantly in the cell AND in the log file.
proc = subprocess.Popen(
    train_cmd, shell=False,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    env={**os.environ, "PYTHONUNBUFFERED": "1"}
)


def _periodic_runs_sync(stop_event: threading.Event, interval_sec: int = 300):
    """
    Periodically sync /content/runs to Drive during training to avoid
    checkpoint loss when Colab runtime disconnects/restarts.
    """
    runs_dir = "/content/runs"
    if not os.path.isdir(DRIVE_RUNS):
        print(f"  ⚠️  Periodic sync disabled (Drive runs path unavailable): {DRIVE_RUNS}", flush=True)
        return

    last_synced_ckpt_mtime = None
    quiet_window_sec = max(0, int(os.environ.get("UAV_SYNC_QUIET_WINDOW_SEC", "0")))
    warned_no_ckpt = False
    while not stop_event.wait(interval_sec):
        if not os.path.isdir(runs_dir):
            continue

        ckpt_candidates = glob.glob(os.path.join(runs_dir, "**", "weights", "last.pt"), recursive=True)
        if not ckpt_candidates:
            if not warned_no_ckpt:
                print("  ℹ️  Periodic sync waiting for first checkpoint...", flush=True)
                warned_no_ckpt = True
            continue

        warned_no_ckpt = False
        latest_ckpt = max(ckpt_candidates, key=os.path.getmtime)
        latest_mtime = os.path.getmtime(latest_ckpt)
        if last_synced_ckpt_mtime is not None and latest_mtime <= last_synced_ckpt_mtime:
            continue
        if quiet_window_sec > 0 and (time.time() - latest_mtime) < quiet_window_sec:
            continue

        _run(f"rsync -am --include='*/' --include='weights/***' --exclude='*' {shlex.quote(runs_dir + '/')} {shlex.quote(DRIVE_RUNS + '/')}", check=False, print_output=False)
        last_synced_ckpt_mtime = latest_mtime
        now = datetime.now().strftime("%H:%M:%S")
        print(
            f"\n  ☁️  [{now}] Periodic checkpoint sync completed "
            f"({os.path.basename(latest_ckpt)} mtime={int(latest_mtime)})",
            flush=True,
        )


_sync_stop = threading.Event()
_sync_thread = None
sync_interval = int(os.environ.get("UAV_SYNC_INTERVAL_SEC", "300"))
if sync_interval > 0:
    _sync_thread = threading.Thread(target=_periodic_runs_sync, args=(_sync_stop, sync_interval), daemon=True)
    _sync_thread.start()
else:
    print("  ℹ️  Periodic runs sync disabled (UAV_SYNC_INTERVAL_SEC<=0). Using epoch-end Drive sync.", flush=True)

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
_sync_stop.set()
if _sync_thread is not None:
    _sync_thread.join(timeout=5)
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
_run(f'rsync -a {shlex.quote(log_dir + "/")} {shlex.quote(drive_log_dir + "/")}', check=False)

# Results are synced by train.py itself, but we can do a fallback sync here just in case
runs_dir = "/content/runs"
if os.path.exists(runs_dir):
    print(f"  ☁️  Syncing runs to: {DRIVE_RUNS}", flush=True)
    _run(f'rsync -a {shlex.quote(runs_dir + "/")} {shlex.quote(DRIVE_RUNS + "/")}', check=False)

# ═══════════════════════════════════════════════════════════════════════════
# 8. Post-Training Summary
# ═══════════════════════════════════════════════════════════════════════════
_banner("8/8 — Post-training summary")

print(f"\n  📁 Training outputs: {DRIVE_RUNS}", flush=True)
_run(f'du -sh {shlex.quote(DRIVE_RUNS)} 2>/dev/null | awk \'{{print "     Size: "$1}}\'', check=False)

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
