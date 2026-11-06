<p align="center">
  <img src="https://github.com/AVAWLeoben/CV-Waste-Tools/blob/b763ed31e1203009e2bbc1a6eb8f0adad1f974ba/CV_Waste_Studio/assets/cv_waste_studio.ico" width="112" alt="CV Waste Studio logo">
</p>

<h1 align="center">CV Waste Studio</h1>

<p align="center">
  <strong>A desktop workspace for YOLO dataset preparation, training, evaluation, tuning, and model export.</strong><br>
  Built for practical computer-vision workflows in waste sorting, recycling research, and other custom object-detection projects.
</p>

<p align="center">
  <a href="https://github.com/AVAWLeoben/CV-Waste-Tools/releases"><strong>Download from Releases</strong></a>
  · <a href="https://github.com/AVAWLeoben/CV-Waste-Tools/issues">Report an issue</a>
  · <a href="#quick-start">Quick start</a>
</p>

---

## Overview

**CV Waste Studio** brings the model-development workflow into one graphical application. Instead of switching between several scripts and windows, you can prepare YOLO datasets, train models, examine results, evaluate detection performance, compare architectures, tune hyperparameters, and export deployment models from a consistent interface.

Studio uses the **Ultralytics YOLO APIs** for training, prediction, validation, tuning, and export. It does **not** replace the underlying Ultralytics training loop or dataloader. Long-running jobs execute in separate processes to keep the desktop interface responsive.

**New here?** Go to the **[Releases page](https://github.com/AVAWLeoben/CV-Waste-Tools/releases)** to find the available download and its release notes. The application is Python-based and uses **Conda** for environment setup; it is not a standalone executable requiring no dependencies.

## Screenshots

| Training monitor | Hyperparameter tuning |
|:---:|:---:|
| ![Live epoch progress and training graphs](docs/images/studio-training.png) | ![Hyperparameter tuning dashboard](docs/images/studio-tuning.png) |

![Model export with platform and dependency checks](docs/images/studio-export.png)

*Screenshots illustrate the interface; plotted example values may be simulated rather than benchmark measurements.*

## Features

| Section | What you can do |
| --- | --- |
| **01 · Datasets & Training** | Combine dataset folders; create train/validation/test splits; train from a new or an existing `data.yaml`; use pretrained or custom YOLO weights; import training settings from `args.yaml` or `best_hyperparameters.yaml`; track iterations, losses, validation metrics, and best epoch. |
| **02 · Annotation Audit** | Inspect YOLO annotations for potential issues and generate review-friendly findings. |
| **03 · Precision / Recall** | Measure recall at a target precision for a selected class; reuse cached predictions; configure NMS; stop evaluations; reopen previous tests; save a summary graphic and numerical results. |
| **04 · Training Results** | Browse previous Ultralytics runs, inspect interactive epoch curves, view validation plots, and examine the recorded metrics table. |
| **05 · Model Comparison** | Queue different YOLO generations, architectures and sizes for sequential training; screen on validation data or compare using a held-out test split; inspect accuracy and efficiency summaries. |
| **06 · Hyperparameter Tuning** | Use Ultralytics' native genetic tuner or optional Bayesian TPE + ASHA through Ray Tune; edit search spaces; import/refine saved best parameters; monitor trials; continue supported saved runs. |
| **07 · Model Export** | Export `.pt` checkpoints to ONNX, TensorRT Engine, or LiteRT/TFLite with format-specific settings and environment checks. |

Other conveniences include a unified activity log, start/stop controls, responsive training layouts, an environment status page, GPU detection, and branded Windows shortcuts.

## Quick start

### 1. Get the application

Open **[GitHub Releases](https://github.com/AVAWLeoben/CV-Waste-Tools/releases)** and download the **CV Waste Studio application ZIP** from the release you want to use. Extract it to a permanent directory and keep the files inside `CV_Waste_Studio/` together, including `assets/` and the launcher scripts.

> **Important:** GitHub's automatically generated *Source code (zip)* archive is not necessarily the same thing as an application release asset. Follow the instructions and asset names for the specific release.

### 2. Windows

**Prerequisite:** Install [Miniforge](https://github.com/conda-forge/miniforge), Miniconda, or Anaconda if Conda is not already available.

1. Open the extracted `CV_Waste_Studio` folder.
2. Double-click **`START_CV_WASTE_TOOLS.vbs`** to start Studio without a command window, or run **`START_GREENPLAST.bat`** to see startup diagnostics.
3. On first launch, allow the launcher to check or prepare a compatible Conda environment. This may download Python/PyTorch/Ultralytics dependencies.
4. Optional: run **`CREATE_CV_WASTE_STUDIO_SHORTCUTS.vbs`** to create branded desktop shortcuts.

The regular shortcut reuses an existing Studio window. A separate **New Window** shortcut allows another instance. Avoid simultaneous GPU jobs using the same dataset cache or results directory.

### 3. Linux

**Prerequisites:** A graphical Linux desktop, Conda (Miniforge/Miniconda/Anaconda), and Tkinter support. NVIDIA drivers and a compatible CUDA-enabled PyTorch build are needed for GPU acceleration.

From a terminal in the extracted application directory:

```bash
cd CV_Waste_Studio
bash START_GREENPLAST.sh
```

The Linux launcher looks for Conda in the current environment and common installation directories, then checks or prepares a suitable environment.

To make an application-menu shortcut, create a `.desktop` entry that runs the **absolute path** to `START_GREENPLAST.sh` and uses the icon at `CV_Waste_Studio/assets/cv_waste_studio.png`. Some desktop environments require you to mark the shortcut as trusted or select **Allow Launching**.

### 4. First launch

- Open **Environment** in the sidebar and confirm the selected Python, PyTorch and device.
- Configure your dataset and output locations before starting a training job.
- Optional export and Bayesian-tuning dependencies are checked when the corresponding tool is opened; they are not all required for ordinary training.

**Supported Python environments:** The launcher can select compatible environments and can create its managed `cv-waste-tools` Conda environment (Python 3.11). Use the bundled launchers for the easiest setup.

## Typical workflow

1. **Prepare data.** Add dataset directories, review labels, and produce a training split—or load an existing YOLO `data.yaml` without re-splitting.
2. **Choose an architecture.** Compare candidate YOLO models and sizes using a consistent dataset and training budget.
3. **Tune hyperparameters.** Start with native genetic tuning or optional Bayesian TPE + ASHA; optionally narrow the search space around a previous winner.
4. **Fine-tune.** Import `best_hyperparameters.yaml` or a previous `args.yaml` into the training tab and run a longer training job.
5. **Inspect results.** Follow live epoch progress, best checkpoint metrics and interactive graphs; reopen the saved training results later.
6. **Evaluate.** Use Precision / Recall to examine an operating point at your required precision, and save the resulting report image.
7. **Export.** Convert the selected checkpoint for your intended deployment target.

### Dataset format

Studio works with YOLO datasets, including conventional sibling `images/` and `labels/` directories and existing YAML-defined splits. One standard arrangement is:

```text
dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

An example `data.yaml`:

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
test: images/test
names:
  0: target_class_0
  1: target_class_1
```

An existing split may also use image-list text files, as supported by Ultralytics. Make sure class IDs and label files match the dataset definition. **Pre-annotation can write new label files**; review predictions before using them as ground truth.

## Training, tuning and evaluation details

- **Training:** Uses native `YOLO(...).train(...)`. Image size, batch size, workers, device, cache and other parameters are configurable. `args.yaml` import can pass supported advanced parameters not otherwise exposed in the form.
- **Caching:** Disk caching can create `.npy` files alongside source images. Automatic cleanup is available for **newly created** image caches; it preserves cache files that predate the job. RAM caching requires enough system memory, especially with multiple Windows dataloader workers.
- **Live monitoring:** Displays iteration progress within epochs, the best epoch so far, and interactive charts for training/validation loss, precision, recall and mAP.
- **Model screening:** The validation-first mode is intended for choosing architecture and size before final testing. The full comparison mode can evaluate held-out test data.
- **Hyperparameter tuning:** Native Ultralytics evolution is available without Ray. Bayesian TPE + ASHA uses optional Ray Tune / Optuna dependencies and checks compatibility when you open the tuning page. Installation requires explicit confirmation.
- **Reproducibility:** Save your dataset YAML, model checkpoint, training arguments, tuning YAML and result files together. Use a genuinely independent test split for the final reported performance.

## Export formats and platform support

| Format | Typical use | Environment notes |
| --- | --- | --- |
| **ONNX** | Portable inference across runtimes | Requires the corresponding Ultralytics export dependencies. |
| **TensorRT (`.engine`)** | NVIDIA GPU deployment | Requires a compatible CUDA GPU and TensorRT builder; optional dependency installation is offered after an environment check. Engines are sensitive to GPU and runtime compatibility. |
| **LiteRT (`.tflite`)** | Lightweight/edge deployment, including CV-Waste Everywhere workflows | Runs natively on compatible Linux systems or through **WSL2 on Windows**. Studio can guide WSL setup and check a separate Linux export environment. |

Export parameters depend on the format and may include square or rectangular image size, batch, FP32/FP16/INT8, dynamic shapes, NMS, ONNX simplification and TensorRT workspace. **INT8 requires representative calibration data.** Validate exported models on the target runtime before deployment.

## Where results and logs are saved

Studio displays the output directory for completed operations and provides **Open Results Folder** where supported.

- **Training:** Native Ultralytics run folders, typically containing `weights/best.pt`, `weights/last.pt`, `results.csv`, `results.png` and `args.yaml`.
- **Tuning:** Trial histories and reusable `best_hyperparameters.yaml`, with checkpoints and other artifacts as supported by the selected tuner.
- **Precision / Recall:** A timestamped `precision_recall_*` directory containing `operating_point_summary.json`, a confidence-sweep CSV, and `evaluation_summary.png` when report generation succeeds.
- **Model comparison:** A comparison CSV/JSON, per-model logs and the respective training/evaluation artifacts.
- **Model export:** A timestamped export directory containing the converted model and `export_summary.json`.
- **Studio diagnostics:** Startup, task and optional dependency-install logs are stored under `CV_Waste_Studio/logs/`.

Do not commit real datasets, proprietary model weights, temporary caches, or private logs to a public repository unless you intend to publish them.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Studio doesn't open | Run `START_GREENPLAST.bat` on Windows or `bash START_GREENPLAST.sh` on Linux to see the actual error. Check the `logs/` directory next to the launcher. |
| Conda not found | Install Miniforge/Miniconda/Anaconda; on Linux, make sure `conda` is on your PATH or set `CONDA_EXE`. |
| CUDA not detected | Verify NVIDIA drivers and the PyTorch/CUDA build inside the Conda environment selected by Studio. |
| Training runs out of RAM | Reduce image cache pressure or batch size; `cache=ram` can be expensive with Windows multiprocessing workers. |
| Disk image caches consume space | Use disk-cache cleanup, or choose cache off; pre-existing `.npy` files are deliberately not deleted automatically. |
| Bayesian tuner unavailable | Open **06 Hyperparameter Tuning**, run the compatibility check, and optionally approve installation of Ray/Optuna in the selected environment. Native tuning remains available. |
| TensorRT export unavailable | Open **07 Model Export** and inspect its CUDA/TensorRT dependency report. Installation requires a compatible GPU, drivers and the full builder package. |
| LiteRT unavailable on Windows | Verify WSL2, your Linux distribution and the WSL export environment; recheck from Model Export. The modern LiteRT-Torch path does not necessarily require the full TensorFlow package. |

For reproducible issues, file a report under **[GitHub Issues](https://github.com/AVAWLeoben/CV-Waste-Tools/issues)** with your OS, Python/Ultralytics/PyTorch versions, model type, the steps to reproduce, and a **redacted** activity or startup log.

## Project layout

```text
CV_Waste_Studio/
├── assets/                        # Application icons
├── START_CV_WASTE_TOOLS.vbs       # Windows quiet launcher
├── START_GREENPLAST.bat           # Windows diagnostic launcher
├── START_GREENPLAST.sh            # Linux launcher
├── greenplast_launcher.py         # Unified desktop interface
├── train_and_split_tool_GUI.py    # Dataset splitting and training
├── hyperparameter_tuning.py       # Native/Ray tuning integration
├── model_comparison_worker.py     # Sequential model comparisons
├── model_export.py                # Native export integration
├── test_model_recall_at_precision.py
├── tests/                          # Regression tests
└── requirements.txt               # Core Python dependencies
```

To run the bundled regression tests from the application directory:

```bash
python -m unittest discover -s tests -q
```

Additional optional packages are required by some tools; `requirements.txt` is not a complete lockfile for every export/tuning backend.

## Contributing and feedback

Bug reports, compatibility findings, documentation corrections and feature suggestions are welcome via **[Issues](https://github.com/AVAWLeoben/CV-Waste-Tools/issues)**. For code contributions, open a pull request with a clear description and relevant test coverage.

**Please keep the distinction between validation and test data clear when reporting model results.** Comparative results depend on the dataset split, hardware, library versions and training budget.

---

<p align="center">
  <strong>CV Waste Studio</strong> · <a href="https://github.com/AVAWLeoben/CV-Waste-Tools/releases">Releases</a> · <a href="https://github.com/AVAWLeoben/CV-Waste-Tools/issues">Issues</a>
</p>
