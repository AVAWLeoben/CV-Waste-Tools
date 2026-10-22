GREENPLAST TOOLS - WINDOWS + LINUX
=================================

Two applications in one launcher:
  (1) Split & Train: make no-copy YOLO train/val/test manifests and train models.
  (2) Annotation Audit: check labeled images using a YOLO model and export a
      CSV for the CV Waste Web REVIEW task. It does not edit annotation files.

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

4. After dependency checks, choose Split & Train or Annotation Audit.

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
select "GPU 0 (CUDA)" under Training device. "Auto" in this training GUI
currently defaults to CPU; installing CUDA PyTorch does not change that.
The launcher displays "CUDA: available" (and the GPU name) once the setup
has been verified. The first installation needs internet access and may
download multiple gigabytes.

WORKFLOW / FILES
----------------
- START_CV_WASTE_TOOLS.vbs: Windows one-click GUI start (no console).
- START_GREENPLAST.bat: Windows diagnostic start (one console).
- START_GREENPLAST.sh: Linux shell start.
- greenplast_launcher.py: common GUI with buttons for both applications.
- greenplast_bootstrap.py: standard-library environment bootstrap for Linux,
  also used by Windows when Conda base lacks tkinter.
- greenplast_environment.py: shared Conda discovery, validation and installer.
- train_and_split_tool_GUI.py: no-copy dataset splitting and YOLO training.
- audit_dataset_review_gui.py: non-recursive annotation audit; reads YOLO
  detection BBOX labels with 5 fields or saved-confidence labels with 6
  fields. Unannotated/background images are skipped by default.
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
- Training runs on CPU: select "GPU 0 (CUDA)" in the training GUI; check
  that launcher status reports "CUDA: available" first.
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
