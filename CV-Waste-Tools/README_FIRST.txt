CV WASTE TOOLS - WINDOWS + LINUX
=================================

Three applications in one launcher:
  (1) Split & Train: make no-copy YOLO train/val/test manifests and train models.
  (2) Annotation Audit: check labeled images using a YOLO model and export a
      CSV for the CV Waste Web REVIEW task. It does not edit annotation files.
  (3) Precision / Recall Test: select a YOLO model, dataset YAML or image folder,
      and target Food precision. Runs one batched model.predict() pass (conf=0.01),
      stores detections in a reusable cache, then computes exact TP/FP/FN and
      Recall@Precision offline. Re-evaluating does not rerun inference.

QUICK START - WINDOWS 10/11
---------------------------
1. Extract the entire archive (keep all files in the same folder).
2. Install Miniforge, Miniconda or Anaconda if Conda is not installed.
3. Double-click START_CV_WASTE_TOOLS.vbs (no command windows).
   For troubleshooting, START_GREENPLAST.bat shows one diagnostic terminal.
4. First run: environment installation/verification may take time.
   Later runs: the verified environment is loaded from launcher_config.json
   without repeating the heavy imports. Use Recheck / repair when needed.

QUICK START - LINUX (DESKTOP / X11 OR XWAYLAND)
----------------------------------------------
1. Extract the entire archive to a writable directory.
2. Install Miniforge, Miniconda or Anaconda if Conda is not installed.
3. In a terminal, change to the extracted directory and run:

      chmod +x START_GREENPLAST.sh
      ./START_GREENPLAST.sh

   Or: bash START_GREENPLAST.sh

4. After dependency checks, choose Split & Train, Annotation Audit or Precision / Recall Test.

The Linux shell script works even if the Conda base environment does not
have tkinter, torch, numpy, or ultralytics. The script locates Conda and
runs greenplast_bootstrap.py using Python from the Conda base environment;
the bootstrap then starts the GUI in the verified environment.


PERSISTENT STARTUP CONFIGURATION (FAST START)
---------------------------------------------
After the first successful environment verification, the launcher saves:
  Windows: %LOCALAPPDATA%\CVWasteTools\launcher_config.json
  Linux:   ~/.config/cv-waste-tools/launcher_config.json
           (or $XDG_CONFIG_HOME/cv-waste-tools/launcher_config.json)

Subsequent starts use this JSON to reuse the chosen Conda interpreter without
scanning every Conda environment, importing PyTorch/Ultralytics, or performing
CUDA operations. No installation or dependency check runs on the fast path.
Only a lightweight environment-filesystem check and NVIDIA GPU/driver identity
query are performed. Both Windows and Linux shell startup also attempt to
launch Conda base Python directly instead of reactivating base each time. The saved environment is automatically reverified when
its Python or installed packages change, Conda or environment paths disappear,
the GPU/driver changes, or this package's requirements change.

Use the launcher button "Recheck / repair environment" to bypass the cache
and run the complete dependency/CUDA test. You can also edit the JSON:
  "always_recheck": true
but leave this false for fast startup. If problems persist after manual edits
to the environment, use Recheck / repair. Do not hand-edit "probe" or
"environment_signature"; they are automatically generated.

The config lives in your user profile, not in the program ZIP, so replacing
program files will not delete it. Each Windows/Linux machine has its own cache.

CONDA ENVIRONMENT NAME
----------------------
The new managed environment is named cv-waste-tools in Conda and
is displayed as "CV Waste Tools" in the application. Conda does not
permit spaces in environment names. Existing suitable environments,
including an older greenplast-tools installation, can still be reused.
The launcher does not rename or delete an existing environment.

GPU SUPPORT (WINDOWS AND LINUX)
-------------------------------
Ultralytics does not have a separate "GPU module". It uses PyTorch for GPU
training and inference. The launcher installs Ultralytics and, when possible,
a prebuilt CUDA-enabled torch + torchvision rather than CPU-only PyTorch.

- The launcher detects BOTH the NVIDIA GPU's compute capability (SM version)
  and the NVIDIA driver's maximum CUDA version. These are different values:
  a driver reporting CUDA 13.0 does NOT imply that a GTX 1060 (sm_61) can
  run PyTorch's CUDA 13.0 kernels.
- For Pascal/older hardware (compute capability below 7.5), it pins
  torch==2.5.1 and torchvision==0.20.1 with CUDA 12.1 (cu121) on drivers
  that support CUDA 12.1, or CUDA 11.8 (cu118) on older supported drivers.
  This retains CUDA kernel support for the GTX 1060 6GB (sm_61).
- For newer NVIDIA cards the launcher keeps the modern CUDA wheel selection:
  driver maximum >=13.2 -> cu132; >=13.0 -> cu130; >=12.8 -> cu128;
  >=12.6 -> cu126; >=11.8 -> cu118.
- IMPORTANT: If an NVIDIA GPU is detected but nvidia-smi omits the CUDA
  version, do not silently choose a CPU-only environment. Test existing
  CUDA-capable environments first; when setup is necessary, attempt a GPU
  wheel selected for the compute capability (e.g. cu128 for an RTX 5090),
  then verify real CUDA execution. If the driver cannot run the selected
  wheel, setup fails with a CUDA error rather than quietly using CPU.
- The PyTorch wheel includes CUDA runtime libraries. Normally you do NOT
  need to separately install the full CUDA Toolkit; you DO need a working
  NVIDIA GPU driver on the host. Conda does not install/update GPU drivers.
- The environment check imports Ultralytics, PyTorch, torchvision, NumPy,
  OpenCV, PyYAML, tkinter and checks torch.cuda.is_available(). It also
  compares the GPU's compute capability with torch.cuda.get_arch_list(),
  then performs an actual tensor operation on the GPU. A PyTorch install
  reporting cuda.is_available() = True is NOT enough if it lacks sm_61 kernels.
- Validation uses the selected environment's Python directly (without a slow
  Conda-run wrapper); its individual stages appear in the status area.
  A timed-out check is not mistaken for missing packages and will not trigger
  a blind PyTorch reinstall.
- If it finds an existing suitable Conda environment, it reuses that
  environment without changing it. Otherwise it creates or repairs a
  persistent environment named cv-waste-tools (display name: CV Waste Tools)
  and installs dependencies.
- The package does not automatically configure AMD ROCm GPU support.
- If the NVIDIA driver is incompatible or CUDA installation fails, the
  launcher reports an error or CPU-only environment; it does NOT silently
  claim GPU support.

TO TRAIN ON THE GPU
-------------------
Open "Split & Train", switch to "2. Model & training", and explicitly
select "GPU 0 (CUDA)" under Training device or keep "Auto (prefer GPU)".
Auto now uses CUDA on supported NVIDIA hardware and raises an error if an
NVIDIA GPU exists but the active environment cannot access CUDA. Select CPU
explicitly only when CPU training is intended.
The launcher displays "CUDA: available" (and the GPU name) once the setup
has been verified. The first installation needs internet access and may
download multiple gigabytes.

WORKFLOW / FILES
----------------
- START_CV_WASTE_TOOLS.vbs: Windows one-click GUI start (no console).
- START_GREENPLAST.bat: Windows diagnostic start (one console).
- START_GREENPLAST.sh: Linux shell start.
- greenplast_launcher.py: common GUI with buttons for all three applications.
- greenplast_bootstrap.py: standard-library environment bootstrap for Linux,
  also used by Windows when Conda base lacks tkinter.
- greenplast_environment.py: shared Conda discovery, validation and installer.
- train_and_split_tool_GUI.py: no-copy dataset splitting and YOLO training.
- audit_dataset_review_gui.py: non-recursive annotation audit; reads YOLO
  detection BBOX labels with 5 fields or saved-confidence labels with 6
  fields. Unannotated/background images are skipped by default.
- test_model_recall_at_precision.py: Tkinter confidence-threshold and recall evaluation.
  Supports data.yaml/autosplit_data.yaml with test/val/train selection, or an
  image folder in standard YOLO labels layout. Exports operating_point_summary.json,
  food_precision_recall_by_confidence.csv and Ultralytics validation plots.
  A model prediction precision target is not a physical mass-based purity target.
- run_tool.py: starts the selected tool in the chosen environment.
- requirements.txt: Ultralytics and other Python dependencies.

Original images and annotations are not moved or copied by splitting;
annotation audit never writes to YOLO label files. Train and audit outputs
are written to paths selected inside the respective tools.

GTX 1060 / PASCAL GPU - CUDA ERROR REPAIR
--------------------------------------
If an earlier launcher installed torch 2.14.1+cu130, it may report:
  "no kernel image is available for execution on the device"
The corrected launcher detects sm_61 and repairs cv-waste-tools in place,
installing torch 2.5.1 + torchvision 0.20.1 from the CUDA 12.1 PyTorch index.
It does not delete the Conda environment. Internet access is required for
this one-time wheel replacement. Restart START_GREENPLAST after upgrading.

For a manual repair from Anaconda Prompt (Windows):
  conda activate cv-waste-tools
  python -m pip install --force-reinstall "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" --index-url https://download.pytorch.org/whl/cu121
  python -c "import torch; print(torch.cuda.get_arch_list()); print(torch.ones(2, device='cuda').sum().item())"
Expected: sm_61 in the architecture list and tensor result 2.0.

Official archived PyTorch wheels: https://pytorch.org/get-started/previous-versions/

TROUBLESHOOTING
---------------
- Conda not found on Linux: set CONDA_EXE=/absolute/path/to/bin/conda
  before launching, or install Conda in a standard location.
- No GUI on Linux: start from an X11/XWayland graphical desktop, with DISPLAY
  and appropriate display permissions configured. Headless servers need a
  desktop, remote X forwarding or an appropriate virtual display.
- GPU absent: check "nvidia-smi" on the host. If it fails, repair/install
  the NVIDIA driver, not just the CUDA Python packages.
- Training runs on CPU: Auto now prefers CUDA and will not silently select
  CPU when an NVIDIA driver is detected. Check that the launcher reports
  "CUDA: available" and use Recheck / repair environment when needed.
- Environment repair: use the launcher "Recheck / repair environment" button.
- If upgrading from a package that reported "timed out after 100 seconds",
  replace the extracted program files with this version and double-click the
  usual launcher. Leave your existing cv-waste-tools environment installed:
  this version will check and reuse it when the dependencies are healthy.
- If a real import stalls for more than eight minutes, the status/log identifies
  the stage (e.g., PyTorch or CUDA). Then check the NVIDIA driver, antivirus,
  and environment without assuming the package installation failed.
- Background images: intentionally missing .txt files are skipped during
  annotation audit, but the selected label directory must still be correct.

This package was syntax-tested, shell-checked and tested with simulated Conda
and CUDA environments. No actual download/installation or NVIDIA GPU test
was run on your hardware. GPU compatibility ultimately depends on the host
GPU, driver, platform and available PyTorch wheels.

CONSOLE-FREE TOOL LAUNCHES
--------------------------
Windows: double-click START_CV_WASTE_TOOLS.vbs. You will see the main
launcher and the GUI of whichever tool you open, with no command prompts.
START_GREENPLAST.bat remains available for debugging (one console window).
Linux: ./START_GREENPLAST.sh uses the existing terminal for environment setup
and starts tools without additional terminals.

Training output, audit tracebacks, and runtime messages appear in the main
launcher's activity log and are saved completely in the logs/ folder.
The Open logs folder button opens that directory. Logs stay available even
if the main launcher is closed while training continues.
Windows bootstrap diagnostics are saved to logs/windows_startup.log when
launching with the .vbs shortcut.


BATCHED ANNOTATION AUDIT AND DEVICE SELECTION (UPDATED)
------------------------------------------------------
The Annotation Audit GUI now accepts:
- Image size (imgsz): 640 / 960 / 1280 / 1536, or any positive valid value.
- Batch size: choose according to available VRAM (e.g. 2 on a 6GB GTX 1060;
  8 or more on a sufficiently equipped RTX 5090).
- Device: Auto (prefer GPU), GPU 0, or CPU.

The audit scans only images directly inside the chosen image folder, checks
whether each has a readable YOLO label, and creates a temporary .txt manifest
containing only the eligible images. It runs one Ultralytics model.predict()
call with stream=True, batch=<selected>, imgsz=<selected>, and device=<selected>.
Ultralytics handles image loading/preprocessing in batches. Prediction results
are joined to each annotation by full resolved filename (not list index) and
written into the same web-REVIEW-compatible CSV schema.

The configured audit confidence cutoff (default 0.80) is deliberately distinct
from the low confidence floor in the Precision/Recall test. The latter uses
Ultralytics model.val()'s batched validation dataloader and P/R curves.

On an NVIDIA machine, a cached Conda environment reporting CUDA=False is
invalidated rather than reused. Rechecking discovers a CUDA-capable
existing environment or repairs the dedicated cv-waste-tools environment.
The software never installs GPU drivers.


FAST RECALL@PRECISION TEST — ONE-TIME INFERENCE + CACHE
------------------------------------------------------
Open Precision / Recall Test from the CV Waste Tools startup window.
Choose best.pt, a dataset YAML (data.yaml / autosplit_data.yaml) or an image
folder, and the desired test/val/train split. Set image size, batch size,
Food class ID, target precision, matching IoU, NMS IoU and GPU/CPU selection.

FIRST RUN:
- Use "Run / reuse cached predictions" with "Reuse existing prediction cache"
  enabled (default).
- Ultralytics predict() processes all selected images ONCE at conf=0.01
  using a .txt image manifest, stream=True, and your batch setting.
- Bounding boxes, class IDs, image indices and confidence scores are saved
  in output/prediction_cache/detections_<key>.npz, with adjacent JSON metadata.
- Label files are read and matched to Food predictions at the configured IoU.
- Matching uses a confidence-ordered, one-to-one greedy class-aware rule.
  Each correct Food prediction counts once (TP); another Food prediction on
  the same Food item, or on Non_Food/background, counts as FP. Unmatched
  Food ground truth counts as FN.
- The confidence sweep is computed from these raw detections; it does not
  run the network again. It reports the maximum Food recall at or above the
  requested minimum Food precision.

LATER RUNS:
- Change target precision or matching IoU and click
  "Recalculate from cache only" for a guaranteed inference-free calculation.
- You can correct ground-truth labels without rerunning inference: labels are
  re-read fresh each time. The model and images must stay unchanged.
- The cache is automatically invalidated if model file, image files, image
  size, NMS IoU or max detections change. Changing batch size does NOT
  invalidate it, since batch size should not change predictions.
- Uncheck "Reuse existing prediction cache" to force another inference pass.
- Changing the model or dataset requires one new pass for that combination.

OUTPUTS:
- output/prediction_cache/detections_<key>.npz (reusable predictions)
- output/prediction_cache/detections_<key>.json (cache metadata)
- output/precision_recall_<timestamp>/food_precision_recall_by_confidence.csv
- output/precision_recall_<timestamp>/operating_point_summary.json
- Optional precision and recall vs confidence PNG (off by default for speed)

NOTES:
- Predicted confidences below 0.01 do not exist in the cache. The tool does
  not reproduce Ultralytics' smoothed/interpolated model.val() PR curves.
  Its values are empirical instance-level TP/FP/FN at the chosen IoU.
- Unlabeled images are treated as annotated backgrounds. Check the logs for
  the count of missing annotation files to spot misconfigured label paths.
- The tool deliberately refuses silent CPU inference if an NVIDIA GPU is
  detectable but torch.cuda.is_available() is False. It asks you to repair
  the launcher environment instead. CPU can be explicitly chosen.
- This is instance-level precision, NOT mass-weighted purity from sorting
  trials. Heavy contaminants may have disproportionate economic impact.
