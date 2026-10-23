CV WASTE STUDIO — SINGLE-WINDOW MODEL TOOLS
===========================================

This edition replaces the old launcher + three independent GUI windows with
one desktop application. Select a tool from the left sidebar; its settings
remain available when you switch tools. The bottom activity console displays
messages for the currently selected tool, and can be hidden or cleared.

THE THREE TOOLS ARE RETAINED
-----------------------------------------

01  DATASETS & TRAINING
    - Use multiple source dataset folders (mixed images/labels or YOLO layouts).
    - Produce train/val/test path lists, with configurable percentages.
    - Optionally filter images with empty annotations.
    - Optionally override YOLO class names.
    - Choose one of three independent workflows: Split + train, Split only,
      or Train existing split (no re-splitting).
    - For Train existing split, browse to a YOLO data.yaml / dataset.yaml /
      autosplit_data.yaml or its containing folder. Train/val and class names
      must be defined. The selected YAML is validated but never changed.
    - Train/fine-tune a YOLO 8 / 11 / 26 model with the same training options.
    - Existing training settings: model size, custom weights, duration in
      epochs/hours, batch, image size, CLS loss weight, device, output and name.
    - Files are referenced by paths, not copied or moved.
    - Selecting sibling images/ and labels/ folders treats them as one dataset.
    - Splitting/training runs in a background *process*, not another window;
      the UI remains responsive. Stop running job is available in this view.
      Training output is saved to a complete timestamped file in logs/.

02  ANNOTATION AUDIT
    - Check model predictions against existing YOLO boxes/polygons.
    - Choose model, image, label and output directories.
    - Retains confidence, IoU, batch, image size, device and coverage controls.
    - Reports progress and supports cancellation.
    - Exports the same web-review-compatible suspicious annotations CSV.
    - Does not modify original annotations.

03  PRECISION / RECALL
    - Choose model + dataset YAML or an image folder.
    - Choose test/val/train split, target precision, Food class, IoU thresholds,
      batch, image size, device, max detections and optional plot output.
    - Cache model predictions so another precision threshold can be evaluated
      without rerunning inference.
    - Writes detailed operating-point summaries and optional plots.

The processing functions in the original Python modules are retained.
Training from existing splits bypasses the splitter entirely.
The old standalone entry points are also included for compatibility, but
normal startup uses only the new single-window CV Waste Studio.

WINDOWS QUICK START
-------------------
1. Extract the entire Model_Training directory, preserving its files.
2. Install Miniforge, Miniconda or Anaconda if Conda is not installed.
3. For the branded Windows launcher, double-click
   CREATE_CV_WASTE_STUDIO_SHORTCUTS.vbs once, then launch via the new
   "CV Waste Studio" shortcut on your desktop or in this folder.
   The original START_CV_WASTE_TOOLS.vbs also works (with Windows' generic
   VBS file icon). To troubleshoot startup, run START_GREENPLAST.bat instead.
4. Initial startup may install/recheck the Conda environment. The initial
   setup screen is then replaced by the main Studio interface automatically.
   It is not a second tool window and does not require user interaction.
5. Choose Datasets & training, Annotation audit or Precision / recall in the sidebar.

LINUX DESKTOP QUICK START
-------------------------
1. Extract Model_Training to a writable folder.
2. Install Miniforge, Miniconda or Anaconda if required.
3. Run: bash START_GREENPLAST.sh
4. The launcher verifies/reuses the managed environment, then opens the
   single desktop interface. An X11 or XWayland display is required.

Using Python directly in an already configured Conda environment:
    python greenplast_launcher.py
The GUI verifies the Conda environment unless launched with a previously
verified environment prepared by the supplied bootstrap.

LAYOUT & CONTROLS
-----------------
- Left sidebar: switch instantly between the three workspaces; no new GUI
  window is opened when switching.
- Environment: view active interpreter, PyTorch/CUDA status, configuration,
  and run Recheck / repair. Environment repair is blocked during active jobs.
- Activity dock: automatic per-workspace logs; Hide log to give settings more
  space, and scroll inside each long settings page where necessary.
- Model operations keep running if you switch workspaces.
- Closing the app during training offers to leave the headless training
  process running, with its saved output log.
- Standard system file/folder dialogs may still open transiently for browsing.

ENVIRONMENT AND OUTPUT FILES
----------------------------
- Managed environment: cv-waste-tools (or another compatible Conda env).
- Saved environment config on Windows:
      %LOCALAPPDATA%\CVWasteTools\launcher_config.json
- Saved environment config on Linux:
      ~/.config/cv-waste-tools/launcher_config.json
- Complete split/training logs: Model_Training/logs/train_<timestamp>.log
- Saved job settings: Model_Training/logs/train_<timestamp>_settings.json
- Session messages: Model_Training/logs/studio_<tool>_<timestamp>.log
- Startup diagnostics on Windows: logs/windows_startup.log
- Existing results (CSV, splits, checkpoints, metrics, prediction caches)
  remain in the chosen destination directories.

Conda, an appropriate Python/Tk installation, and the requirements in
requirements.txt are needed. GPU support depends on the installed NVIDIA
driver, PyTorch wheel, and hardware. Training/inference were not exercised
on GPU as part of this interface refactor; check on your machine before a
long production run. This project does not install GPU drivers.

TECHNICAL NOTES / BACKWARD COMPATIBILITY
----------------------------------------
- GUI shell: greenplast_launcher.py
- App branding: studio_branding.py, assets/cv_waste_studio_windows_v5.ico and assets/cv_waste_studio.png
  (original user-supplied logo; small GUI variants are derived from it)
- Windows branded shortcut installer: CREATE_CV_WASTE_STUDIO_SHORTCUTS.vbs
- Branding and shortcut instructions: ICON_AND_SHORTCUTS.txt
- Headless split/train worker: run_training_job.py
- Original split/train module: train_and_split_tool_GUI.py
- Original annotation audit: audit_dataset_review_gui.py
- Original cached evaluation: test_model_recall_at_precision.py
- Original optional old GUI dispatcher: run_tool.py
- Core environment detection/installation: greenplast_environment.py
- Initial Linux setup: greenplast_bootstrap.py

The GUI remains built on Python Tkinter/ttk; no browser, cloud service, or
additional GUI framework is required. Training runs in a child Python
process in the verified environment. Audit and recall run on background
threads using the original implementations.

When developing without an installed Ultralytics/PyTorch environment, the
GUI can be smoke-tested with mocked environment details, but actual inference
and training require the full packages and a suitable model/dataset.

TRAINING AN EXISTING SPLIT WITHOUT CREATING NEW SPLITS
----------------------------------------------------
1. In Datasets & training, choose "Train existing split" at the top.
2. Browse to your existing YOLO dataset YAML (or enter its containing folder).
   A standard dataset typically has these keys:

      path: D:/datasets/my_dataset
      train: images/train
      val: images/val
      test: images/test       # optional
      names: [class_one, class_two]

   Alternatively, 'train' and 'val' can point to text manifests of images.
   Existing CV Waste Studio splits have an autosplit_data.yaml file.
3. Click "Check dataset" to verify required YAML keys and class names.
   File availability and annotation validity are fully checked by Ultralytics
   when training begins; Check dataset is intentionally a lightweight check.
4. Set the model/weights, epochs, batch, image size, CLS weight, device, results
   folder, and run name in "2. Model & training".
5. Click "Train existing dataset". No split generation runs. The input YAML,
   split lists, image files and labels are not rewritten by this workflow.
   Training outputs go to the selected results folder.

If you have separate images/ and labels/ directories but no train/val split,
use Split + train. If you already split images into train/val/test but have
no YAML, add a standard Ultralytics data.yaml first.

ICON TROUBLESHOOTING (V5): See ICON_AND_SHORTCUTS.txt. Unpin old Windows
taskbar icons before recreating shortcuts with the supplied VBS installer.
