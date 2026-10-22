# -*- coding: utf-8 -*-
"""Create no-copy Ultralytics dataset splits and optionally train YOLO.

Default: integrated dataset/training GUI. Run with --autosplit for split-only GUI.
"""

import os
import re
import subprocess
import sys

# Workaround for possible OpenMP library conflicts (Spyder / Windows).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# --- OPTIONAL DEPENDENCIES ---
def _load_dependency(import_name, pip_name):
    """Import a dependency; install it into the running Python environment if needed."""
    try:
        return __import__(import_name)
    except ImportError:
        print(f"Installing missing dependency: {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        return __import__(import_name)


def _build_model_filename(version, size):
    """Return a model name such as yolov8n.pt, yolo11s.pt, or yolo26m.pt."""
    version_key = str(version).lower().strip()
    size_key = str(size).lower().strip()

    if size_key not in ("n", "s", "m", "l", "x"):
        raise ValueError(f"Invalid YOLO model size: {size!r}")

    versions = {
        "8": "yolov8", "v8": "yolov8", "yolov8": "yolov8",
        "11": "yolo11", "v11": "yolo11", "yolo11": "yolo11",
        "26": "yolo26", "v26": "yolo26", "yolo26": "yolo26",
    }
    if version_key not in versions:
        raise ValueError(f"Unsupported YOLO version: {version!r}")
    return f"{versions[version_key]}{size_key}.pt"


def _normalize_device_choice(device):
    """Normalize supported device settings without probing hardware."""
    if device is None:
        return "auto"
    value = str(device).strip().lower()
    if value in ("auto", "automatic", "auto (cpu default)", "auto (prefer gpu)"):
        return "auto"
    if value in ("0", "gpu", "gpu 0", "cuda:0", "gpu 0 (cuda/rocm)"):
        return "gpu"
    if value == "cpu":
        return "cpu"
    raise ValueError(f"Invalid training device: {device!r}. Choose 'auto', 0, or 'cpu'.")


def _validate_cls_weight(value):
    """Parse a non-negative, finite YOLO classification-loss weight."""
    import math

    try:
        weight = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError) as exc:
        raise ValueError("CLS loss weight must be a number (e.g. 0.5).") from exc
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("CLS loss weight must be finite and at least 0.")
    return weight


# Verified wheel families: CUDA 12.6, 13.0, and 13.2.
# Pair torch 2.14.1 with its matching torchvision 0.29.1 release.
_TORCH_VERSION = "2.14.1"
_TORCHVISION_VERSION = "0.29.1"


def _nvidia_gpu_info():
    """Return (GPU model, maximum supported CUDA version) reported by nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True,
            timeout=15, check=False, errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout
    version_match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
    # Use a second, machine-readable query for the GPU name.
    try:
        name_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
            check=False, errors="replace",
        )
        name = name_result.stdout.strip().splitlines()[0] if name_result.returncode == 0 else "NVIDIA GPU"
    except (OSError, subprocess.TimeoutExpired, IndexError):
        name = "NVIDIA GPU"
    version = (int(version_match[1]), int(version_match[2])) if version_match else None
    return name, version


def _cuda_index_for_driver(max_cuda):
    """Choose a documented CUDA wheel that the installed driver can support."""
    if max_cuda is None:
        raise RuntimeError(
            "nvidia-smi did not report the driver's CUDA version. "
            "Check the NVIDIA driver and install PyTorch manually from "
            "https://pytorch.org/get-started/locally/."
        )
    if max_cuda >= (13, 2):
        return "cu132"
    if max_cuda >= (13, 0):
        return "cu130"
    if max_cuda >= (12, 6):
        return "cu126"
    raise RuntimeError(
        f"NVIDIA driver reports CUDA {max_cuda[0]}.{max_cuda[1]}, "
        "but this script's supported PyTorch wheels require CUDA 12.6 or later. "
        "Update the NVIDIA driver, then try again."
    )


def _probe_pytorch():
    """Inspect PyTorch in a fresh process (important when replacing installed wheels)."""
    probe_code = (
        "import torch; "
        "print('TORCH_VERSION=' + str(torch.__version__)); "
        "print('CUDA_BUILD=' + str(torch.version.cuda)); "
        "print('HIP_BUILD=' + str(getattr(torch.version, 'hip', None))); "
        "print('GPU_AVAILABLE=' + str(torch.cuda.is_available())); "
        "print('GPU_NAME=' + (torch.cuda.get_device_name(0) "
        "if torch.cuda.is_available() else 'none'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe_code],
        capture_output=True, text=True, check=False, timeout=90,
        errors="replace",
    )
    if result.returncode != 0:
        return {"ok": False, "error": (result.stderr or result.stdout).strip()}
    details = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            details[key] = value.strip()
    return {"ok": "GPU_AVAILABLE" in details, **details}


def _ensure_gpu_pytorch(device, auto_install_cuda):
    """Optionally repair a CPU-only PyTorch install, before importing Ultralytics."""
    preference = _normalize_device_choice(device)
    print(f"Python executable: {sys.executable}")
    existing = _probe_pytorch()
    if existing["ok"]:
        print(f"PyTorch version: {existing.get('TORCH_VERSION', 'unknown')}")
        print(f"PyTorch CUDA build: {existing.get('CUDA_BUILD', 'unknown')}")
        print(f"GPU visible to PyTorch: {existing.get('GPU_AVAILABLE', 'False')}")
    else:
        print("PyTorch is missing or could not be loaded: " + existing.get("error", "unknown"))

    # Auto is deliberately CPU-first for students. Only an explicit GPU
    # selection can trigger a CUDA installation or a GPU requirement.
    if preference in ("auto", "cpu") or existing.get("GPU_AVAILABLE") == "True":
        return

    # A CUDA build is already installed. Reinstalling cannot fix a missing
    # driver or an unsupported GPU, so do not repeat the download every run.
    if existing["ok"] and existing.get("CUDA_BUILD") not in (None, "None", ""):
        message = (
            "CUDA-enabled PyTorch is installed, but it cannot access a GPU. "
            "Check nvidia-smi, NVIDIA driver compatibility, and GPU support."
        )
        if preference == "gpu":
            raise RuntimeError(message)
        print("WARNING: " + message + " Using CPU.")
        return

    gpu = _nvidia_gpu_info()
    if gpu is None:
        if preference == "gpu":
            raise RuntimeError(
                "GPU 0 was requested, but PyTorch cannot access it and nvidia-smi "
                "cannot detect an NVIDIA GPU. Install/check your NVIDIA driver "
                "and run nvidia-smi from this terminal."
            )
        print("WARNING: NVIDIA GPU/driver not detected; using CPU. "
              "No CUDA packages were installed.")
        return

    name, maximum_cuda = gpu
    print(f"NVIDIA GPU detected: {name}")
    print("Driver maximum CUDA version: " + (
        f"{maximum_cuda[0]}.{maximum_cuda[1]}" if maximum_cuda else "unknown"
    ))
    if not auto_install_cuda:
        if preference == "gpu":
            raise RuntimeError(
                "GPU requested but PyTorch cannot access it. Enable 'Install "
                "CUDA-enabled PyTorch' in the setup window, or install a CUDA "
                "build manually using https://pytorch.org/get-started/locally/."
            )
        print("WARNING: GPU is detected but CUDA PyTorch installation was disabled; "
              "using CPU.")
        return

    cuda_variant = _cuda_index_for_driver(maximum_cuda)
    index_url = f"https://download.pytorch.org/whl/{cuda_variant}"
    # Explicit opt-in from the setup form; force replacement of +cpu wheels.
    command = [
        sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall",
        f"torch=={_TORCH_VERSION}", f"torchvision=={_TORCHVISION_VERSION}",
        "--index-url", index_url,
    ]
    print(f"Installing CUDA-enabled PyTorch ({cuda_variant}) in this Python environment...")
    print("This may download several GB and change packages in the active environment.")
    print("Command: " + subprocess.list2cmdline(command))
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"CUDA PyTorch installation failed (exit code {exc.returncode}). "
            f"Check the pip messages above and use {index_url} for troubleshooting."
        ) from exc

    updated = _probe_pytorch()
    if not updated["ok"] or updated.get("GPU_AVAILABLE") != "True":
        raise RuntimeError(
            "CUDA PyTorch was installed, but a fresh Python process still cannot "
            "access the GPU. Verify your driver/GPU compatibility and check "
            f"the installation output. Probe: {updated}"
        )
    print(f"CUDA PyTorch ready: {updated.get('TORCH_VERSION')} | "
          f"GPU: {updated.get('GPU_NAME')}")
    if "torch" in sys.modules:
        raise RuntimeError(
            "PyTorch was already imported in this process before the CUDA "
            "installation. Restart the Python process or Spyder kernel, then "
            "run DSL_train.py again."
        )


def _select_training_device(device, torch):
    """Use CPU by default. Require an explicit selection to train on GPU 0."""
    preference = _normalize_device_choice(device)
    available = bool(torch.cuda.is_available())
    print(f"Active PyTorch version: {torch.__version__}")
    print(f"Active CUDA build: {torch.version.cuda or 'not installed'}")
    print(f"GPU visible to PyTorch: {'Yes' if available else 'No'}")
    if preference == "auto":
        print("Selected training device: CPU (student-friendly automatic default).")
        return "cpu"
    if preference == "cpu":
        print("Selected training device: CPU (requested explicitly).")
        return "cpu"
    if not available:
        raise RuntimeError(
            "GPU 0 was selected but CUDA-enabled PyTorch is not available in this "
            "Python process. Check the NVIDIA driver and the PyTorch installation."
        )
    print(f"Selected training device: GPU 0 ({torch.cuda.get_device_name(0)}).")
    return 0


# --- SHARED GUI CONTROLS ---
def _select_multiple_directories(parent, initial_dir=None, title="Select dataset folders"):
    """Choose several folders in one dialog, with Ctrl/Shift multi-selection.

    Tkinter's native askdirectory only returns one path. This custom dialog
    lets users add many sibling folders at once, and accumulate folders from
    different locations before clicking Done. No extra package is required.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from pathlib import Path

    current = Path(initial_dir or Path.home()).expanduser().resolve()
    if not current.is_dir():
        current = Path.home()
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.geometry("820x650")
    dialog.minsize(660, 500)
    dialog.transient(parent)
    dialog.columnconfigure(0, weight=1)
    dialog.rowconfigure(2, weight=1)
    dialog.rowconfigure(5, weight=1)

    ttk.Label(dialog, text="Select multiple folders with Ctrl-click or Shift-click; then Add selected.",
              padding=(14, 12, 14, 4)).grid(row=0, column=0, sticky="w")
    navigation = ttk.Frame(dialog, padding=(14, 3, 14, 6))
    navigation.grid(row=1, column=0, sticky="ew")
    navigation.columnconfigure(0, weight=1)
    location_var = tk.StringVar(value=str(current))
    location_entry = ttk.Entry(navigation, textvariable=location_var)
    location_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))

    folder_frame = ttk.Frame(dialog, padding=(14, 0, 14, 0))
    folder_frame.grid(row=2, column=0, sticky="nsew")
    folder_frame.rowconfigure(0, weight=1)
    folder_frame.columnconfigure(0, weight=1)
    folder_list = tk.Listbox(folder_frame, selectmode="extended", exportselection=False,
                             activestyle="dotbox", height=10)
    folder_list.grid(row=0, column=0, sticky="nsew")
    folder_scroll = ttk.Scrollbar(folder_frame, orient="vertical", command=folder_list.yview)
    folder_scroll.grid(row=0, column=1, sticky="ns")
    folder_list.configure(yscrollcommand=folder_scroll.set)
    displayed = []
    basket = []

    selected_title = tk.StringVar(value="Selected folders (0)")
    ttk.Label(dialog, textvariable=selected_title, padding=(14, 10, 14, 3)).grid(
        row=4, column=0, sticky="w")
    basket_frame = ttk.Frame(dialog, padding=(14, 0, 14, 0))
    basket_frame.grid(row=5, column=0, sticky="nsew")
    basket_frame.columnconfigure(0, weight=1)
    basket_frame.rowconfigure(0, weight=1)
    basket_list = tk.Listbox(basket_frame, height=5, selectmode="extended", exportselection=False)
    basket_list.grid(row=0, column=0, sticky="nsew")
    basket_scroll = ttk.Scrollbar(basket_frame, command=basket_list.yview)
    basket_scroll.grid(row=0, column=1, sticky="ns")
    basket_list.configure(yscrollcommand=basket_scroll.set)

    def navigate(path):
        nonlocal current, displayed
        candidate = Path(str(path).strip().strip('"')).expanduser()
        try:
            candidate = candidate.resolve()
            if not candidate.is_dir():
                raise NotADirectoryError(candidate)
            children = sorted((p for p in candidate.iterdir() if p.is_dir()),
                              key=lambda p: p.name.casefold())
        except (OSError, ValueError) as exc:
            messagebox.showerror("Cannot open folder", str(exc), parent=dialog)
            return
        current = candidate
        location_var.set(str(current))
        displayed = children
        folder_list.delete(0, tk.END)
        for folder in displayed:
            folder_list.insert(tk.END, folder.name)

    def go_up():
        navigate(current.parent)

    def browse_location():
        path = filedialog.askdirectory(parent=dialog, initialdir=str(current),
                                       title="Navigate to a folder", mustexist=True)
        if path:
            navigate(path)

    ttk.Button(navigation, text="Go", command=lambda: navigate(location_var.get())).grid(
        row=0, column=1, padx=(0, 5))
    ttk.Button(navigation, text="Up", command=go_up).grid(row=0, column=2, padx=(0, 5))
    ttk.Button(navigation, text="Browse location...", command=browse_location).grid(row=0, column=3)
    location_entry.bind("<Return>", lambda _event: navigate(location_var.get()))

    def enter_folder(_event=None):
        indices = folder_list.curselection()
        if indices:
            navigate(displayed[indices[0]])

    folder_list.bind("<Double-Button-1>", enter_folder)

    def add_folders(paths):
        for folder in paths:
            folder = Path(folder).resolve()
            if folder not in basket:
                basket.append(folder)
                basket_list.insert(tk.END, str(folder))
        selected_title.set(f"Selected folders ({len(basket)})")

    def add_selected():
        indices = folder_list.curselection()
        if not indices:
            messagebox.showinfo("Select folders", "Select one or more folders from the list. "
                                "Use Ctrl-click or Shift-click to select multiple.", parent=dialog)
            return
        add_folders([displayed[i] for i in indices])

    def remove_selected():
        for i in reversed(basket_list.curselection()):
            del basket[i]
            basket_list.delete(i)
        selected_title.set(f"Selected folders ({len(basket)})")

    actions = ttk.Frame(dialog, padding=(14, 8, 14, 0))
    actions.grid(row=3, column=0, sticky="ew")
    ttk.Button(actions, text="Add selected folders", command=add_selected).pack(side="left", padx=(0, 8))
    ttk.Button(actions, text="Add current folder", command=lambda: add_folders([current])).pack(side="left")
    ttk.Button(actions, text="Remove from selection", command=remove_selected).pack(side="right")
    chosen = []

    def confirm():
        chosen.extend(str(folder) for folder in basket)
        dialog.destroy()

    footer = ttk.Frame(dialog, padding=14)
    footer.grid(row=6, column=0, sticky="e")
    ttk.Button(footer, text="Cancel", command=dialog.destroy).pack(side="right", padx=(8, 0))
    ttk.Button(footer, text="Add folders to dataset", command=confirm).pack(side="right")
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    navigate(current)
    dialog.grab_set()
    folder_list.focus_set()
    parent.wait_window(dialog)
    return chosen


def _make_split_sliders(parent, initial_weights, tk, ttk):
    """Return text variables for three linked sliders that always total 100%."""
    initial = [float(x) for x in initial_weights]
    total = sum(initial)
    if total <= 0:
        initial = [80.0, 10.0, 10.0]
        total = 100.0
    values = [round(100.0 * x / total, 1) for x in initial[:2]]
    values.append(round(100.0 - sum(values), 1))
    texts = [tk.StringVar(value=f"{v:g}") for v in values]
    sliders = []
    syncing = False

    parent.columnconfigure(1, weight=1)
    for i, name in enumerate(("Train", "Validation", "Test")):
        ttk.Label(parent, text=name).grid(row=i, column=0, sticky="w", padx=(0, 12), pady=7)
        scale = ttk.Scale(parent, from_=0, to=100, orient="horizontal")
        scale.grid(row=i, column=1, sticky="ew", padx=(0, 12), pady=7)
        sliders.append(scale)
        entry = ttk.Entry(parent, textvariable=texts[i], width=7, justify="right")
        entry.grid(row=i, column=2, sticky="e", pady=7)
        ttk.Label(parent, text="%").grid(row=i, column=3, sticky="w", padx=(4, 0))

        def submit_entry(_event=None, index=i):
            try:
                number = float(texts[index].get().replace(",", "."))
                if not 0 <= number <= 100 or not __import__("math").isfinite(number):
                    raise ValueError("out of range")
            except ValueError:
                texts[index].set(f"{values[index]:g}")
                return
            adjust(index, number)

        entry.bind("<Return>", submit_entry)
        entry.bind("<FocusOut>", submit_entry)

    def adjust(index, requested):
        nonlocal syncing
        if syncing:
            return
        number = round(max(0, min(100, float(requested))), 1)
        remaining = round(100 - number, 1)
        a, b = [i for i in range(3) if i != index]
        other_sum = values[a] + values[b]
        portion = values[a] / other_sum if other_sum > 0 else 0.5
        a_new = round(remaining * portion, 1)
        b_new = round(remaining - a_new, 1)
        values[index], values[a], values[b] = number, a_new, b_new
        syncing = True
        try:
            for i in range(3):
                texts[i].set(f"{values[i]:g}")
                sliders[i].set(values[i])
        finally:
            syncing = False

    for i, scale in enumerate(sliders):
        scale.configure(command=lambda val, index=i: adjust(index, round(float(val))))
    syncing = True
    for i, scale in enumerate(sliders):
        scale.set(values[i])
    syncing = False
    return texts


# --- NO-COPY ULTRALYTICS AUTOSPLIT TOOL ---
def _autosplit_source_paths(dataset_paths):
    """Normalize one directory or a sequence, retaining selection order."""
    from pathlib import Path

    if dataset_paths is None:
        return []
    if isinstance(dataset_paths, (str, os.PathLike)):
        dataset_paths = [dataset_paths]
    seen = set()
    result = []
    for item in dataset_paths:
        text = os.fspath(item).strip().strip('"')
        if not text:
            continue
        path = Path(os.path.expanduser(text)).resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Dataset folder not found: {path}")
        if path not in seen:
            result.append(path)
            seen.add(path)
    return result


def _autosplit_layout(selected):
    """Return (images directory, dataset root) for a source selection."""
    if selected.name.lower() == "images":
        return selected, selected.parent
    if (selected / "images").is_dir():
        return selected / "images", selected
    return selected, selected


def _show_autosplit_configuration(dataset_paths, weights, ignore_empty_images,
                                  output_dir):
    """GUI: add/remove source folders, choose output folder and split settings."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from pathlib import Path

    root = tk.Tk()
    root.title("Ultralytics AutoSplit - Multiple Datasets, No Copies")
    root.geometry("840x680")
    root.minsize(720, 570)
    try:
        ttk.Style(root).theme_use("clam")
    except tk.TclError:
        pass

    content = ttk.Frame(root, padding=18)
    content.pack(fill="both", expand=True)
    content.columnconfigure(1, weight=1)
    content.rowconfigure(1, weight=1)

    ttk.Label(content, text="Combine Dataset Folders into Split Lists",
              font=("Segoe UI", 15, "bold")).grid(
        row=0, column=0, columnspan=4, sticky="w", pady=(0, 12)
    )
    ttk.Label(content, text="Source folders").grid(
        row=1, column=0, sticky="nw", padx=(0, 12), pady=6
    )
    source_frame = ttk.Frame(content)
    source_frame.grid(row=1, column=1, columnspan=2, sticky="nsew", pady=6)
    source_frame.columnconfigure(0, weight=1)
    source_frame.rowconfigure(0, weight=1)
    source_list = tk.Listbox(source_frame, height=7, selectmode="extended",
                             exportselection=False)
    source_list.grid(row=0, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(source_frame, command=source_list.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    source_list.configure(yscrollcommand=scrollbar.set)

    sources = [str(p) for p in dataset_paths]
    for path in sources:
        source_list.insert(tk.END, path)

    output_var = tk.StringVar(value=str(output_dir) if output_dir else "")
    output_auto = not bool(output_dir)

    def update_default_output():
        if not output_auto or not sources:
            return
        first = Path(sources[0])
        images_dir, root_dir = _autosplit_layout(first)
        if len(sources) > 1:
            output_var.set(str(root_dir / "combined_split_lists"))
        elif images_dir == root_dir:
            output_var.set(str(root_dir / "split_lists"))
        else:
            output_var.set(str(root_dir))

    update_default_output()

    actions = ttk.Frame(content)
    actions.grid(row=1, column=3, sticky="n", padx=(12, 0), pady=6)

    def add_folder():
        initial = str(Path(sources[-1]).parent) if sources else os.path.expanduser("~")
        chosen = _select_multiple_directories(root, initial, "Add dataset folders")
        for selected in chosen:
            path = str(Path(selected).resolve())
            if path not in sources:
                sources.append(path)
                source_list.insert(tk.END, path)
        if chosen:
            update_default_output()

    def remove_selected():
        for idx in reversed(source_list.curselection()):
            del sources[idx]
            source_list.delete(idx)
        update_default_output()

    ttk.Button(actions, text="Add Folder(s)...", command=add_folder).pack(
        fill="x", pady=(0, 8))
    ttk.Button(actions, text="Remove Selected", command=remove_selected).pack(fill="x")

    ttk.Label(
        content,
        text="Each folder can contain images/ + labels/ or image files beside their .txt labels.",
        foreground="#555555",
    ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(2, 12))

    ttk.Label(content, text="Output folder").grid(
        row=3, column=0, sticky="w", padx=(0, 12), pady=6)
    ttk.Entry(content, textvariable=output_var).grid(
        row=3, column=1, columnspan=2, sticky="ew", pady=6)

    def browse_output():
        nonlocal output_auto
        initial = output_var.get().strip()
        if not os.path.isdir(initial):
            initial = str(Path(initial).parent) if initial else os.path.expanduser("~")
        if not os.path.isdir(initial):
            initial = os.path.expanduser("~")
        chosen = filedialog.askdirectory(parent=root, title="Choose Output Folder",
                                         initialdir=initial, mustexist=True)
        if chosen:
            output_auto = False
            output_var.set(chosen)

    ttk.Button(content, text="Browse...", command=browse_output).grid(
        row=3, column=3, padx=(12, 0), pady=6)

    ttk.Label(content, text="Split percentages").grid(
        row=4, column=0, sticky="nw", padx=(0, 12), pady=8)
    ratios = ttk.Frame(content)
    ratios.grid(row=4, column=1, columnspan=3, sticky="ew", pady=8)
    split_vars = _make_split_sliders(ratios, weights, tk, ttk)

    ignore_var = tk.BooleanVar(value=bool(ignore_empty_images))
    ttk.Checkbutton(
        content,
        text="IGNORE EMPTY IMAGES (missing, empty or whitespace-only .txt labels)",
        variable=ignore_var,
    ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(12, 4))
    ttk.Label(
        content,
        text="Combines all selected folders into train.txt / val.txt / test.txt + autosplit_data.yaml.",
        foreground="#555555",
    ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(4, 3))
    ttk.Label(
        content, text="Only file paths are written. Source images and annotations stay in place.",
        foreground="#555555",
    ).grid(row=7, column=0, columnspan=4, sticky="w", pady=(0, 8))

    result = {}

    def confirm():
        if not sources:
            messagebox.showerror("Missing Dataset", "Add at least one source folder.",
                                 parent=root)
            return
        try:
            paths = _autosplit_source_paths(sources)
            values = tuple(float(v.get().strip().replace(",", ".")) for v in split_vars)
            if (not all(0 <= v <= 100 for v in values)
                    or abs(sum(values) - 100) > 0.01):
                raise ValueError("Train, validation and test percentages must add up to 100.")
            if values[0] == 0:
                raise ValueError("Training percentage must be above zero.")
            dest = output_var.get().strip().strip('"')
            if not dest:
                raise ValueError("Select an output folder.")
            dest = Path(os.path.expanduser(dest)).resolve()
            if dest.exists() and not dest.is_dir():
                raise ValueError("Output path exists but is not a folder.")
        except (ValueError, NotADirectoryError) as exc:
            messagebox.showerror("Invalid Settings", str(exc), parent=root)
            return
        existing = [str(dest / filename) for filename in
                    ("train.txt", "val.txt", "test.txt", "autosplit_data.yaml")
                    if (dest / filename).exists()]
        if existing and not messagebox.askyesno(
            "Replace Existing Split Lists",
            "These output files already exist and will be replaced:\n\n"
            + "\n".join(existing) + "\n\nContinue?", parent=root,
        ):
            return
        result.update({
            "dataset_paths": paths,
            "output_dir": dest,
            "weights": tuple(v / 100 for v in values),
            "ignore_empty_images": ignore_var.get(),
        })
        root.destroy()

    controls = ttk.Frame(content)
    controls.grid(row=8, column=0, columnspan=4, sticky="e", pady=(6, 0))
    ttk.Button(controls, text="Cancel", command=root.destroy).pack(
        side="left", padx=(0, 10))
    ttk.Button(controls, text="Create Combined Split Lists", command=confirm).pack(side="left")
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return result or None


def _autosplit_read_names(dataset_root):
    """Get explicit class names from an original dataset YAML, if supplied."""
    import yaml

    for filename in ("data.yaml", "dataset.yaml"):
        source = dataset_root / filename
        if source.is_file():
            data = yaml.safe_load(source.read_text(encoding="utf-8-sig")) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Invalid class YAML: {source}")
            names = data.get("names")
            if isinstance(names, list):
                return {i: str(name) for i, name in enumerate(names)}
            if isinstance(names, dict) and names:
                try:
                    return {int(key): str(value) for key, value in names.items()}
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid class ID in {source}") from exc
    return None


def _autosplit_combined_names(sources, eligible_images, label_for):
    """Validate that source datasets use compatible class IDs and names."""
    names = None
    names_origin = None
    for _, root_dir in sources:
        source_names = _autosplit_read_names(root_dir)
        if source_names is None:
            continue
        if names is None:
            names, names_origin = source_names, root_dir
        elif source_names != names:
            raise ValueError(
                f"Class names do not match across dataset folders:\n"
                f"  {names_origin}: {names}\n  {root_dir}: {source_names}\n"
                "Class ID mappings must be identical before combining datasets."
            )

    max_class_id = -1
    for image in eligible_images:
        label = label_for(image)
        if not label.is_file():
            continue
        with label.open("r", encoding="utf-8-sig", errors="replace") as stream:
            for line in stream:
                columns = line.split()
                if not columns:
                    continue
                try:
                    class_id = int(columns[0])
                except ValueError:
                    continue
                if class_id >= 0:
                    max_class_id = max(max_class_id, class_id)

    if names is None:
        if max_class_id > 1000:
            raise ValueError("Unexpected class ID > 1000; check annotation files.")
        return {i: f"class_{i}" for i in range(max(1, max_class_id + 1))}, True

    if max_class_id >= 0 and max_class_id not in names:
        raise ValueError(
            f"Annotation uses class ID {max_class_id}, but the supplied data.yaml "
            "does not define that ID. Check class mappings before training."
        )
    return names, False


def autosplit_dataset(dataset_path=None, weights=(0.8, 0.2, 0.0),
                      ignore_empty_images=True, dataset_paths=None, output_dir=None):
    """Combine one or more datasets into YOLO split lists, without copying images.

    ``dataset_path`` accepts one folder or a list of folders for compatibility.
    ``dataset_paths`` is the clearer multi-source keyword argument.
    With no paths provided, opens a Tkinter window allowing the user to add
    multiple source folders, remove selections, choose an output directory,
    set split ratios and toggle IGNORE EMPTY IMAGES (on by default).

    Each source may contain ``images/`` and ``labels/`` subfolders, or mixed
    images and labels together. The function invokes Ultralytics autosplit
    individually on each images folder, then combines the split assignments
    into three absolute-path manifests with duplicate paths removed. Existing
    autosplit_*.txt files in source folders are restored, not overwritten.

    IMPORTANT: Class IDs must have the same meaning in every source dataset.
    When data.yaml class names exist, mismatches raise an error. If no class
    YAML exists, placeholder names are generated and need user verification.
    """
    import math
    from pathlib import Path

    weights = tuple(float(w) for w in weights)
    if (len(weights) != 3 or not all(math.isfinite(w) and w >= 0 for w in weights)
            or sum(weights) <= 0 or weights[0] <= 0):
        raise ValueError("weights must be (train, val, test), nonnegative, with train > 0.")
    weights = tuple(w / sum(weights) for w in weights)

    if dataset_path is not None and dataset_paths is not None:
        raise ValueError("Use dataset_path or dataset_paths, not both.")
    supplied = dataset_paths if dataset_paths is not None else dataset_path
    selected_paths = _autosplit_source_paths(supplied)

    if not selected_paths:
        options = _show_autosplit_configuration([], weights, ignore_empty_images, output_dir)
        if options is None:
            print("AutoSplit cancelled by the user.")
            return None
        selected_paths = options["dataset_paths"]
        weights = options["weights"]
        ignore_empty_images = options["ignore_empty_images"]
        output_dir = options["output_dir"]

    sources = [_autosplit_layout(p) for p in selected_paths]
    if output_dir is None:
        first_images, first_root = sources[0]
        if len(selected_paths) > 1:
            output_dir = first_root / "combined_split_lists"
        else:
            output_dir = first_root / "split_lists" if first_images == first_root else first_root
    output_dir = Path(os.path.expanduser(os.fspath(output_dir).strip().strip('"'))).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output folder is not a directory: {output_dir}")

    _load_dependency("ultralytics", "ultralytics")
    from ultralytics.data.split import autosplit
    from ultralytics.data.utils import IMG_FORMATS, img2label_paths
    import yaml

    def label_for(image):
        return Path(img2label_paths([str(image)])[0])

    def nonempty_label(label):
        if not label.is_file():
            return False
        with label.open("r", encoding="utf-8-sig", errors="replace") as stream:
            return any(line.strip() for line in stream)

    raw_names = ("train", "val", "test")
    combined = {split: [] for split in raw_names}
    seen_images = set()
    stats = {
        "excluded_empty_images": 0,
        "excluded_duplicates": 0,
        "excluded_old_split_images": 0,
    }
    total_seen = 0
    print(f"Combining {len(sources)} source dataset(s), no image/label copies.")
    print(f"Ignore empty images: {'YES' if ignore_empty_images else 'NO'}")
    print(f"Output folder: {output_dir}")
    for idx, (images_dir, root_dir) in enumerate(sources, start=1):
        print(f"[{idx}/{len(sources)}] Ultralytics autosplit: {images_dir}")
        if not any(p.is_file() and p.suffix[1:].lower() in IMG_FORMATS
                   for p in images_dir.rglob("*")):
            raise ValueError(f"No supported images found in {images_dir}")

        intermediates = {
            kind: images_dir.parent / f"autosplit_{kind}.txt"
            for kind in raw_names
        }
        # Ultralytics deletes/recreates intermediates in images_dir.parent.
        # Preserve any user files already present there, even after a failure.
        original = {path: path.read_bytes() if path.exists() else None
                    for path in intermediates.values()}
        try:
            autosplit(path=images_dir, weights=weights, annotated_only=False)
            for kind, raw in intermediates.items():
                if not raw.is_file():
                    continue
                for entry in raw.read_text(encoding="utf-8-sig").splitlines():
                    if not entry.strip():
                        continue
                    image = (raw.parent / entry.strip()).resolve()
                    if not image.is_file():
                        raise FileNotFoundError(f"Ultralytics AutoSplit returned missing image: {image}")
                    total_seen += 1
                    if image in seen_images:
                        stats["excluded_duplicates"] += 1
                        continue
                    seen_images.add(image)
                    # Exclude previously generated split directories from source scans.
                    if images_dir == root_dir:
                        parts = image.relative_to(images_dir).parts
                        if parts and (parts[0].lower() in (
                                "split_lists", "combined_split_lists", "splitted_dataset")
                                or parts[0].lower().startswith("splitted_dataset_")):
                            stats["excluded_old_split_images"] += 1
                            continue
                    if ignore_empty_images and not nonempty_label(label_for(image)):
                        stats["excluded_empty_images"] += 1
                        continue
                    combined[kind].append(image)
        finally:
            for path, previous_bytes in original.items():
                if previous_bytes is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(previous_bytes)

    total = sum(len(group) for group in combined.values())
    if not total:
        raise ValueError("No eligible images found. Check your labels and the ignore-empty setting.")
    if not combined["train"]:
        raise ValueError("Train split is empty. Increase train percentage or add more images.")

    images = [image for group in combined.values() for image in group]
    names, placeholders = _autosplit_combined_names(sources, images, label_for)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for kind, group in combined.items():
        target = output_dir / f"{kind}.txt"
        target.write_text("".join(f"{p.as_posix()}\n" for p in group), encoding="utf-8")
        results[kind] = str(target)
        print(f"{kind}: {len(group)} images -> {target}")
        if weights[raw_names.index(kind)] > 0 and not group:
            print(f"WARNING: {kind} received no images. This is a random split; try more images.")

    yaml_data = {
        "path": str(output_dir),
        "train": "train.txt",
        "val": "val.txt",
        "names": names,
    }
    if weights[2] > 0:
        yaml_data["test"] = "test.txt"
    yaml_path = output_dir / "autosplit_data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8")

    if placeholders:
        print("NOTE: No shared class names found in source data.yaml files. "
              "Verify/edit the class names in autosplit_data.yaml before training.")
    print(f"Total eligible image paths: {total}")
    print(f"Excluded missing/empty/whitespace labels: {stats['excluded_empty_images']}")
    print(f"Deduplicated overlapping source paths: {stats['excluded_duplicates']}")
    print(f"YOLO dataset configuration: {yaml_path}")
    print("Source images, labels and pre-existing AutoSplit files were left unchanged.")
    return {
        **results,
        "yaml": str(yaml_path),
        "counts": {kind: len(group) for kind, group in combined.items()},
        "source_folders": [str(p) for p in selected_paths],
        "output_dir": str(output_dir),
        **stats,
    }


# --- ALL-IN-ONE MULTI-DATASET SPLIT + TRAIN (NO IMAGE/LABEL COPYING) ---
def _all_in_one_gui():
    """One Tkinter window containing both splitting and training settings."""
    import math
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from pathlib import Path

    root = tk.Tk()
    root.title("YOLO Dataset Builder")
    root.geometry("880x735")
    root.minsize(760, 660)
    try:
        ttk.Style(root).theme_use("clam")
    except tk.TclError:
        pass
    main = ttk.Frame(root, padding=16)
    main.pack(fill="both", expand=True)
    ttk.Label(main, text="YOLO Dataset Builder",
              font=("Segoe UI", 17, "bold")).pack(anchor="w", pady=(0, 4))
    ttk.Label(main, text="Combine datasets using path lists. No images or labels are copied.",
              foreground="#555555").pack(anchor="w", pady=(0, 14))

    # Keep the split-only switch visible regardless of the active tab.
    do_train = tk.BooleanVar(value=True)
    mode_bar = ttk.Frame(main)
    mode_bar.pack(fill="x", pady=(0, 10))
    ttk.Label(mode_bar, text="After creating split lists:").pack(side="left", padx=(0, 12))
    ttk.Radiobutton(mode_bar, text="Train model", variable=do_train,
                    value=True).pack(side="left", padx=(0, 12))
    ttk.Radiobutton(mode_bar, text="Splits only", variable=do_train,
                    value=False).pack(side="left")
    # Reserve the bottom command row before packing the expandable notebook.
    buttons = ttk.Frame(main)
    buttons.pack(side="bottom", fill="x", pady=(12, 0))
    notebook = ttk.Notebook(main)
    notebook.pack(fill="both", expand=True)
    split_tab = ttk.Frame(notebook, padding=18)
    train_page = ttk.Frame(notebook)
    notebook.add(split_tab, text="1. Dataset & split")
    notebook.add(train_page, text="2. Model & training")
    split_tab.columnconfigure(1, weight=1)
    train_page.columnconfigure(0, weight=1)
    train_page.rowconfigure(0, weight=1)
    training_canvas = tk.Canvas(train_page, highlightthickness=0, bd=0)
    training_canvas.grid(row=0, column=0, sticky="nsew")
    training_scroll = ttk.Scrollbar(train_page, orient="vertical",
                                    command=training_canvas.yview)
    training_scroll.grid(row=0, column=1, sticky="ns")
    training_canvas.configure(yscrollcommand=training_scroll.set)
    train_tab = ttk.Frame(training_canvas, padding=18)
    train_tab.columnconfigure(1, weight=1)
    training_window = training_canvas.create_window((0, 0), window=train_tab, anchor="nw")
    training_canvas.bind("<Configure>", lambda event: training_canvas.itemconfigure(
        training_window, width=event.width))
    train_tab.bind("<Configure>", lambda _event: training_canvas.configure(
        scrollregion=training_canvas.bbox("all")))

    def training_mousewheel(event):
        if getattr(event, "num", 0) == 4:
            delta = -1
        elif getattr(event, "num", 0) == 5:
            delta = 1
        else:
            delta = -int(event.delta / 120) if event.delta else 0
        if delta:
            training_canvas.yview_scroll(delta, "units")

    def enable_training_scroll(_event):
        root.bind_all("<MouseWheel>", training_mousewheel)
        root.bind_all("<Button-4>", training_mousewheel)
        root.bind_all("<Button-5>", training_mousewheel)

    def disable_training_scroll(_event):
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            root.unbind_all(sequence)

    training_canvas.bind("<Enter>", enable_training_scroll)
    training_canvas.bind("<Leave>", disable_training_scroll)

    # Sources: a list of independently labeled mixed folders or YOLO roots.
    sources = []
    ttk.Label(split_tab, text="Source folders").grid(row=0, column=0, sticky="nw", pady=6)
    list_frame = ttk.Frame(split_tab)
    list_frame.grid(row=0, column=1, sticky="nsew", padx=12, pady=6)
    list_frame.columnconfigure(0, weight=1)
    list_frame.rowconfigure(0, weight=1)
    folders = tk.Listbox(list_frame, height=8, selectmode="extended", exportselection=False)
    folders.grid(row=0, column=0, sticky="nsew")
    scroll = ttk.Scrollbar(list_frame, command=folders.yview)
    scroll.grid(row=0, column=1, sticky="ns")
    folders.config(yscrollcommand=scroll.set)
    split_output = tk.StringVar()
    results_folder = tk.StringVar()
    output_custom = False
    results_custom = False

    def fill_defaults():
        if not sources:
            return
        root_dir = _autosplit_layout(Path(sources[0]))[1]
        default = root_dir / "combined_split_lists"
        if not output_custom:
            split_output.set(str(default))
        if not results_custom:
            results_folder.set(str(Path(split_output.get()) / "training_runs"))

    def add_source():
        initial = str(Path(sources[-1]).parent) if sources else str(Path.home())
        selected = _select_multiple_directories(root, initial, "Select dataset folders")
        for path in selected:
            path = str(Path(path).resolve())
            if path not in sources:
                sources.append(path)
                folders.insert(tk.END, path)
        if selected:
            fill_defaults()

    def remove_source():
        for index in reversed(folders.curselection()):
            folders.delete(index)
            sources.pop(index)
        fill_defaults()

    actions = ttk.Frame(split_tab)
    actions.grid(row=0, column=2, sticky="n", pady=6)
    ttk.Button(actions, text="Add folder(s)...", command=add_source).pack(fill="x", pady=(0, 8))
    ttk.Button(actions, text="Remove selected", command=remove_source).pack(fill="x")
    ttk.Label(split_tab, text="Mixed image.jpg/image.txt folders or images/ + labels/ folders",
              foreground="#555555").grid(row=1, column=0, columnspan=3,
                                           sticky="w", pady=(0, 14))

    ttk.Label(split_tab, text="Split-list output").grid(row=2, column=0, sticky="w", pady=7)
    ttk.Entry(split_tab, textvariable=split_output).grid(row=2, column=1, sticky="ew", padx=12)

    def browse_output():
        nonlocal output_custom
        initial = Path(split_output.get()).parent if split_output.get() else Path.home()
        if not initial.is_dir():
            initial = Path.home()
        path = filedialog.askdirectory(parent=root, title="Choose split output folder",
                                       initialdir=str(initial), mustexist=True)
        if path:
            output_custom = True
            split_output.set(path)
            fill_defaults()

    ttk.Button(split_tab, text="Browse...", command=browse_output).grid(row=2, column=2)
    ttk.Label(split_tab, text="Train / Val / Test %").grid(row=3, column=0, sticky="nw", pady=8)
    pframe = ttk.Frame(split_tab)
    pframe.grid(row=3, column=1, columnspan=2, sticky="ew", padx=12, pady=8)
    percentages = _make_split_sliders(pframe, (80, 10, 10), tk, ttk)
    ttk.Label(split_tab, text="Adjust any split; totals remain 100%.",
              foreground="#555555").grid(row=4, column=0, columnspan=3,
                                             sticky="w", pady=(0, 8))
    ignore_empty = tk.BooleanVar(value=True)
    ttk.Checkbutton(split_tab, variable=ignore_empty,
                    text="Exclude images without non-empty annotations").grid(
        row=5, column=0, columnspan=3, sticky="w", pady=(15, 10))
    ttk.Label(split_tab, text="Class names (optional)").grid(row=6, column=0, sticky="w", pady=8)
    class_names = tk.StringVar()
    ttk.Entry(split_tab, textvariable=class_names).grid(
        row=6, column=1, columnspan=2, sticky="ew", padx=12)
    ttk.Label(split_tab,
              text="Comma-separated, e.g. plastic, metal. Leave blank to use data.yaml "
                   "from the dataset folders (or generated class_0 names).",
              wraplength=740, foreground="#555555").grid(
        row=7, column=0, columnspan=3, sticky="w", pady=(4, 12))
    ttk.Label(split_tab, text="Output: train.txt, val.txt, test.txt and autosplit_data.yaml.",
              wraplength=740, foreground="#555555").grid(
        row=8, column=0, columnspan=3, sticky="w", pady=10)

    # Only show the training settings when training is selected.
    ttk.Label(train_tab, text="Training configuration",
              font=("Segoe UI", 12, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

    def field(label, row):
        ttk.Label(train_tab, text=label).grid(row=row, column=0,
                                              sticky="w", padx=(0, 15), pady=8)

    version = tk.StringVar(value="11")
    size = tk.StringVar(value="l")
    field("YOLO version", 1)
    ttk.Combobox(train_tab, textvariable=version, values=("8", "11", "26"),
                 width=16, state="readonly").grid(row=1, column=1, sticky="w", pady=8)
    field("Model size", 2)
    ttk.Combobox(train_tab, textvariable=size, values=("n", "s", "m", "l", "x"),
                 width=16, state="readonly").grid(row=2, column=1, sticky="w", pady=8)
    field("Pretrained weights (.pt)", 3)
    custom_weights = tk.StringVar()
    ttk.Entry(train_tab, textvariable=custom_weights).grid(row=3, column=1,
                                                            sticky="ew", pady=8)

    def choose_weights():
        path = filedialog.askopenfilename(parent=root, title="Select existing YOLO weights",
                                          filetypes=[("PyTorch YOLO weights", "*.pt")])
        if path:
            custom_weights.set(path)

    ttk.Button(train_tab, text="Browse...", command=choose_weights).grid(row=3, column=2,
                                                                           padx=(12, 0))
    ttk.Label(train_tab, text="Optional: choose an existing best.pt to fine-tune; "
              "otherwise use version + size.", foreground="#555555").grid(
        row=4, column=1, columnspan=2, sticky="w")
    field("Training duration", 5)
    duration_mode = tk.StringVar(value="Epochs")
    duration = tk.StringVar(value="100")
    durations = ttk.Frame(train_tab)
    durations.grid(row=5, column=1, sticky="w", pady=8)
    duration_picker = ttk.Combobox(durations, textvariable=duration_mode,
                                   values=("Epochs", "Hours"), state="readonly", width=12)
    duration_picker.pack(side="left", padx=(0, 8))
    ttk.Entry(durations, textvariable=duration, width=12).pack(side="left")
    duration_picker.bind("<<ComboboxSelected>>", lambda _event: duration.set(
        "100" if duration_mode.get() == "Epochs" else "1"))
    field("Batch size", 6)
    batch = tk.StringVar(value="4")
    ttk.Entry(train_tab, textvariable=batch, width=15).grid(row=6, column=1, sticky="w")
    field("Image size (imgsz)", 7)
    imgsz = tk.StringVar(value="640")
    ttk.Entry(train_tab, textvariable=imgsz, width=15).grid(row=7, column=1, sticky="w")
    field("CLS loss weight", 8)
    cls_weight = tk.StringVar(value="0.5")
    ttk.Entry(train_tab, textvariable=cls_weight, width=15).grid(row=8, column=1, sticky="w")
    ttk.Label(train_tab, text="Default 0.5; higher values prioritize classification loss.",
              foreground="#555555").grid(row=8, column=2, sticky="w", padx=(8, 0))
    field("Training device", 9)
    choices = {"auto": "Auto (CPU default)", "cpu": "CPU", "gpu": "GPU 0 (CUDA)"}
    device = tk.StringVar(value=choices["auto"])
    ttk.Combobox(train_tab, textvariable=device, values=tuple(choices.values()),
                 width=25, state="readonly").grid(row=9, column=1, sticky="w", pady=8)
    cuda_install = tk.BooleanVar(value=False)
    cuda_checkbox = ttk.Checkbutton(train_tab, variable=cuda_install,
                                    text="Install CUDA PyTorch if needed (NVIDIA, opt-in)")
    cuda_checkbox.grid(row=10, column=0, columnspan=3, sticky="w", pady=(0, 10))

    def toggle_cuda(*_args):
        if device.get() == choices["gpu"]:
            cuda_checkbox.state(["!disabled"])
        else:
            cuda_install.set(False)
            cuda_checkbox.state(["disabled"])

    device.trace_add("write", toggle_cuda)
    toggle_cuda()
    field("Training results folder", 11)
    ttk.Entry(train_tab, textvariable=results_folder).grid(row=11, column=1,
                                                            sticky="ew", pady=8)

    def browse_results():
        nonlocal results_custom
        path = filedialog.askdirectory(parent=root, title="Choose training results folder",
                                       initialdir=str(Path.home()), mustexist=True)
        if path:
            results_custom = True
            results_folder.set(path)

    ttk.Button(train_tab, text="Browse...", command=browse_results).grid(row=11, column=2,
                                                                           padx=(12, 0))
    field("Run name", 12)
    run_name = tk.StringVar(value="train")
    ttk.Entry(train_tab, textvariable=run_name).grid(row=12, column=1, sticky="ew")
    compact_progress = tk.BooleanVar(value=True)
    ttk.Checkbutton(train_tab, variable=compact_progress,
                    text="Compact training progress (one update per epoch)").grid(
        row=13, column=0, columnspan=3, sticky="w", pady=(12, 6))
    ttk.Label(train_tab, text="Training creates model checkpoints and metrics in the results folder.",
              foreground="#555555").grid(row=14, column=0, columnspan=3,
                                           sticky="w", pady=(6, 0))

    chosen_settings = {}

    def submit():
        try:
            if not sources:
                raise ValueError("Add at least one source dataset folder.")
            weights = [float(x.get().strip().replace(",", ".")) for x in percentages]
            if (not all(math.isfinite(w) and 0 <= w <= 100 for w in weights)
                    or abs(sum(weights) - 100) > 0.01 or weights[0] <= 0):
                raise ValueError("Split percentages must add up to 100 (train > 0%).")
            if do_train.get() and weights[1] <= 0:
                raise ValueError("Validation percentage must be above zero for training.")
            out = split_output.get().strip().strip('"')
            if not out:
                raise ValueError("Choose a split output folder.")
            out = Path(out).expanduser().resolve()
            if out.exists() and not out.is_dir():
                raise ValueError("Split output destination is not a folder.")
            names = [n.strip() for n in class_names.get().split(",")] if class_names.get().strip() else None
            if names is not None and (not all(names) or len(names) != len(set(names))):
                raise ValueError("Class names must be nonempty and unique.")
            values = {
                "dataset_paths": sources[:], "output_dir": out,
                "weights": tuple(w / 100 for w in weights),
                "ignore_empty_images": ignore_empty.get(),
                "class_names": names, "train_after_split": do_train.get(),
            }
            if do_train.get():
                model_weights = custom_weights.get().strip().strip('"')
                if model_weights and not Path(model_weights).expanduser().is_file():
                    raise ValueError(f"Weights file not found: {model_weights}")
                if not model_weights:
                    _build_model_filename(version.get(), size.get())
                n_batch = int(batch.get())
                n_imgsz = int(imgsz.get())
                selected_cls_weight = _validate_cls_weight(cls_weight.get())
                if n_batch <= 0 or n_imgsz <= 0:
                    raise ValueError("Batch and image size must be positive integers.")
                duration_text = duration.get().strip().replace(",", ".")
                hours = None
                epochs = None
                if duration_mode.get() == "Hours":
                    hours = float(duration_text)
                    if not math.isfinite(hours) or hours <= 0:
                        raise ValueError("Training hours must be positive.")
                else:
                    epochs = int(duration_text)
                    if epochs <= 0:
                        raise ValueError("Training epochs must be positive.")
                project_path = results_folder.get().strip().strip('"') or str(out / "training_runs")
                values.update({
                    "version": version.get(), "size": size.get(),
                    "model_path": model_weights or None, "epochs": epochs,
                    "time": hours, "batch": n_batch, "imgsz": n_imgsz,
                    "cls_weight": selected_cls_weight,
                    "device": next(k for k, v in choices.items() if v == device.get()),
                    "auto_install_cuda": cuda_install.get(),
                    "project": project_path, "run_name": run_name.get().strip() or "train",
                    "compact_progress": compact_progress.get(),
                })
            existing = [str(out / p) for p in ("train.txt", "val.txt", "test.txt", "autosplit_data.yaml")
                        if (out / p).exists()]
            if existing and not messagebox.askyesno(
                "Overwrite split files", "Replace these files?\n\n" + "\n".join(existing),
                parent=root,
            ):
                return
        except (ValueError, TypeError, OSError) as exc:
            messagebox.showerror("Invalid settings", str(exc), parent=root)
            return
        chosen_settings.update(values)
        root.destroy()

    ttk.Button(buttons, text="Cancel", command=root.destroy).pack(side="right")
    launch_button = ttk.Button(buttons, command=submit)
    launch_button.pack(side="right", padx=(0, 12))

    def update_launch(*_args):
        enabled = do_train.get()
        notebook.tab(train_page, state="normal" if enabled else "disabled")
        if not enabled and notebook.select() == str(train_page):
            notebook.select(split_tab)
        launch_button.configure(text="Create splits + train" if enabled
                                else "Create split lists")

    do_train.trace_add("write", update_launch)
    update_launch()
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return chosen_settings or None


def _train_with_compact_progress(model, train_args, compact_progress=True):
    """Suppress Ultralytics per-batch tqdm redraws, reporting once per epoch.

    Patches only the trainer/validator progress bar constructors during this
    single synchronous model.train() call. Warnings, exceptions and other
    important logging stay visible, and the originals are restored afterward.
    """
    if not compact_progress:
        return model.train(**train_args)

    from contextlib import ExitStack
    from unittest.mock import patch
    from functools import partial
    import ultralytics.engine.trainer as trainer_module
    import ultralytics.engine.validator as validator_module

    def epoch_summary(trainer):
        epoch = getattr(trainer, "epoch", -1) + 1
        total = getattr(trainer, "epochs", "?")
        parts = [f"Epoch {epoch}/{total} complete"]
        losses = getattr(trainer, "tloss", None)
        if isinstance(losses, dict) and losses:
            formatted = []
            for name, loss in losses.items():
                try:
                    formatted.append(f"{name}={float(loss):.4f}")
                except (TypeError, ValueError):
                    continue
            if formatted:
                parts.append(" ".join(formatted))
        metrics = getattr(trainer, "metrics", None)
        if isinstance(metrics, dict):
            for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
                if key in metrics:
                    try:
                        parts.append(f"{key.split('/')[-1]}={float(metrics[key]):.4f}")
                    except (TypeError, ValueError):
                        pass
        print(" | ".join(parts), flush=True)

    if hasattr(model, "add_callback"):
        model.add_callback("on_fit_epoch_end", epoch_summary)

    with ExitStack() as stack:
        for module in (trainer_module, validator_module):
            original = getattr(module, "TQDM", None)
            if original is not None:
                # Passing disable=True prevents progress updates from printing
                # but preserves tqdm's iteration and set_description interface.
                stack.enter_context(patch.object(module, "TQDM", partial(original, disable=True)))
        return model.train(**train_args)


def autosplit_and_train_model(
    dataset_paths=None, output_dir=None, weights=(0.8, 0.1, 0.1),
    ignore_empty_images=True, train_after_split=True, class_names=None,
    version="11", size="l", model_path=None, epochs=100, time=None,
    batch=4, imgsz=640, device="auto", auto_install_cuda=False,
    project=None, run_name="train", compact_progress=True, cls_weight=0.5,
):
    """Combine original image paths from several datasets into split lists and train.

    With no dataset_paths: display the integrated GUI. When dataset_paths is
    passed, run without prompting. Source files are never copied or moved.
    """
    import math
    from pathlib import Path

    if dataset_paths is None:
        settings = _all_in_one_gui()
        if settings is None:
            print("Cancelled: no split or training started.")
            return None
        return autosplit_and_train_model(**settings)
    weights = tuple(float(w) for w in weights)
    if train_after_split and len(weights) == 3 and weights[1] <= 0:
        raise ValueError("Training requires a validation split above zero.")
    if train_after_split:
        batch = int(batch)
        imgsz = int(imgsz)
        if batch <= 0 or imgsz <= 0:
            raise ValueError("Batch and image size must be positive.")
        cls_weight = _validate_cls_weight(cls_weight)
        if time is not None:
            time = float(time)
            if not math.isfinite(time) or time <= 0:
                raise ValueError("Training time must be finite and positive.")
        else:
            epochs = int(epochs)
            if epochs <= 0:
                raise ValueError("Epochs must be positive.")
        model_file = (str(Path(model_path).expanduser().resolve()) if model_path else
                      _build_model_filename(version, size))
        if model_path and not Path(model_file).is_file():
            raise FileNotFoundError(f"YOLO weights not found: {model_file}")

    split_result = autosplit_dataset(
        dataset_paths=dataset_paths, output_dir=output_dir,
        weights=weights, ignore_empty_images=ignore_empty_images,
    )
    if split_result is None:
        return None
    import yaml
    yaml_file = Path(split_result["yaml"])
    config = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    if isinstance(class_names, str):
        class_names = [n.strip() for n in class_names.split(",")]
    if class_names is not None:
        class_names = [str(n).strip() for n in class_names]
        if not class_names or any(not n for n in class_names) or len(set(class_names)) != len(class_names):
            raise ValueError("Class names must be nonempty and unique.")
        if len(class_names) < len(config["names"]):
            raise ValueError(f"Detected {len(config['names'])} class IDs, but received "
                             f"only {len(class_names)} names. Update autosplit_data.yaml.")
        config["names"] = {i: n for i, n in enumerate(class_names)}
        yaml_file.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                             encoding="utf-8")
        print(f"Using {len(class_names)} user-specified class names.")

    if not train_after_split:
        print("Split lists created; training was disabled.")
        return {"split": split_result, "training": None}
    if split_result["counts"]["val"] <= 0:
        raise ValueError("No validation images after filtering. Training did not start. "
                         "Increase validation % or add more labeled images. "
                         f"The lists remain at {split_result['output_dir']}")
    if all(str(n).startswith("class_") for n in config["names"].values()):
        print("WARNING: Class names are placeholders. Check autosplit_data.yaml "
              "for the correct names/IDs.")

    # Reuse the existing CUDA/CPU handling and optional dependency installer.
    device = _normalize_device_choice(device)
    _ensure_gpu_pytorch(device, auto_install_cuda)
    _load_dependency("ultralytics", "ultralytics")
    from ultralytics import YOLO
    import torch
    selected_device = _select_training_device(device, torch)

    project_dir = (Path(project).expanduser().resolve() if project else
                   Path(split_result["output_dir"]) / "training_runs")
    project_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "data": str(yaml_file), "imgsz": imgsz, "batch": batch,
        "device": selected_device, "project": str(project_dir),
        "name": str(run_name).strip() or "train", "cls": cls_weight,
    }
    if time is not None:
        kwargs["time"] = time
    else:
        kwargs["epochs"] = epochs
    print(f"Training {model_file} on {sum(split_result['counts'].values())} images")
    print(f"Data: {yaml_file} | device: {selected_device} | image size: {imgsz} | cls: {cls_weight:g}")
    trained = _train_with_compact_progress(YOLO(model_file), kwargs, compact_progress)
    print(f"Training finished. Results saved in: {project_dir}")
    return {"split": split_result, "training": trained}


if __name__ == "__main__":
    # Default: combined no-copy split + training GUI.
    # --autosplit: split-only GUI (same no-copy backend).
    args = sys.argv[1:]
    if not args:
        autosplit_and_train_model()
    elif args == ["--autosplit"]:
        autosplit_dataset()
    else:
        raise SystemExit("Usage: python DSL_TOOLS_with_CLS_weight_clean.py [--autosplit]")
