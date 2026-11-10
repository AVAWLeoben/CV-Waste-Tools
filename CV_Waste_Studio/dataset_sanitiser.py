"""Safely preview and sanitise YOLO datasets without modifying training code.

Scan creates a JSON plan and never edits the dataset. Applying a plan checks
source-file fingerprints before copying a filtered dataset or deleting pairs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid

from preannotate_dataset import IMAGE_SUFFIXES

PLAN_FORMAT = "cv-waste-dataset-sanitise-v1"
SPLIT_LISTS = frozenset({"train.txt", "val.txt", "test.txt", "autosplit_train.txt",
                         "autosplit_val.txt", "autosplit_test.txt"})


def _root(path):
    if not str(path or "").strip():
        raise ValueError("Select a dataset folder before scanning.")
    source = Path(str(path)).expanduser().resolve()
    if source == Path(source.anchor):
        raise ValueError("Select a dataset folder, not the filesystem root.")
    if source.is_file() and source.suffix.lower() in {".yaml", ".yml"}:
        source = source.parent
    if not source.is_dir():
        raise ValueError(f"Dataset folder does not exist: {source}")
    if source.name.lower() == "images" and (source.parent / "labels").is_dir():
        source = source.parent
    return source


def _within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_output(source, output):
    if not str(output or "").strip():
        raise ValueError("Choose an output dataset folder for 'Create cleaned copy'.")
    destination = Path(output).expanduser().resolve()
    if destination == source or _within(destination, source) or _within(source, destination):
        raise ValueError("The cleaned dataset must be outside the source dataset and must not contain it.")
    if destination.exists():
        raise FileExistsError(f"Output already exists; select a new folder: {destination}")
    if not destination.parent.is_dir():
        raise ValueError(f"Output parent folder does not exist: {destination.parent}")
    return destination


def validate_settings(settings):
    source = _root(settings.get("source", ""))
    method = settings.get("method", "labels")
    if method not in ("labels", "model"):
        raise ValueError("Choose either existing label files or YOLO model inference.")
    action = settings.get("action", "copy")
    if action not in ("copy", "inplace"):
        raise ValueError("Choose a cleaned copy or in-place removal.")
    output = str(_validate_output(source, settings.get("output", ""))) if action == "copy" else ""
    values = {"source": str(source), "method": method, "action": action, "output": output,
              "include_missing": bool(settings.get("include_missing", False))}
    if method == "model":
        model = Path(str(settings.get("model", ""))).expanduser().resolve()
        if not model.is_file() or model.suffix.lower() not in {".pt", ".engine", ".onnx"}:
            raise ValueError("Choose an existing YOLO .pt, .engine or .onnx model.")
        try:
            conf = float(settings.get("conf", 0.25))
            imgsz = int(settings.get("imgsz", 640))
            batch = int(settings.get("batch", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("Confidence, image size and batch must be numeric.") from exc
        device = str(settings.get("device", "auto")).strip().lower()
        if not (math.isfinite(conf) and 0 < conf <= 1):
            raise ValueError("Confidence must be greater than 0 and at most 1.")
        if imgsz < 32 or batch < 1 or device not in {"auto", "cpu", "0"}:
            raise ValueError("Image size >=32, batch >=1, and device auto/CPU/GPU 0 are required.")
        values.update(model=str(model), conf=conf, imgsz=imgsz, batch=batch, device=device)
    return values


def _fingerprint(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Unsafe, missing or non-regular dataset file: {path}")
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def discover_pairs(source):
    """Support images/train + labels/train, train/images + train/labels, and mixed folders."""
    source = _root(source)
    image_dirs = sorted((d for d in source.rglob("images") if d.is_dir() and
                         not d.is_symlink() and not any(part.startswith(".") for part in d.relative_to(source).parts)),
                        key=lambda p: str(p).casefold())
    # Avoid descending another 'images' directory under an already selected images root.
    roots = [d for d in image_dirs if not any(_within(d, earlier) for earlier in image_dirs if earlier != d)]
    if roots:
        images = sorted({p for folder in roots for p in folder.rglob("*")
                         if p.is_file() and not p.is_symlink() and p.suffix.lower() in IMAGE_SUFFIXES},
                        key=lambda p: str(p).casefold())
    else:
        images = sorted((p for p in source.rglob("*") if p.is_file() and
                         not p.is_symlink() and p.suffix.lower() in IMAGE_SUFFIXES and
                         not any(part.startswith(".") for part in p.relative_to(source).parts)),
                        key=lambda p: str(p).casefold())
    if not images:
        raise ValueError(f"No supported images were found below {source}")
    seen_labels = set()
    pairs = []
    for image in images:
        relative = image.relative_to(source)
        parts = list(relative.parts)
        if "images" in [p.lower() for p in parts[:-1]]:
            idx = next(i for i, part in enumerate(parts[:-1]) if part.lower() == "images")
            parts[idx] = "labels"
            label_rel = Path(*parts).with_suffix(".txt")
        else:
            label_rel = relative.with_suffix(".txt")
        key = str(label_rel).casefold()
        if key in seen_labels:
            raise ValueError(f"Multiple images would share the label file {label_rel}. Rename duplicate stems first.")
        seen_labels.add(key)
        label = source / label_rel
        if label.is_symlink():
            raise ValueError(f"Refusing to handle a symbolic link as a label: {label}")
        pairs.append((image, label if label.is_file() else None))
    return pairs


def _detect_empty(detector, images, values):
    args = {"conf": values["conf"], "imgsz": values["imgsz"], "batch": values["batch"],
            "verbose": False, "save": False}
    if values["device"] != "auto":
        args["device"] = values["device"]
    task = getattr(detector, "task", None)
    if task not in (None, "detect", "segment", "pose", "obb"):
        raise ValueError(f"This sanitizer requires a detection/segmentation/pose/OBB model, got {task!r}.")
    for offset in range(0, len(images), values["batch"]):
        group = images[offset:offset + values["batch"]]
        result_set = list(detector.predict(source=[str(p) for p in group], **args))
        if len(result_set) != len(group):
            raise RuntimeError(f"YOLO returned {len(result_set)} results for {len(group)} images.")
        for image, result in zip(group, result_set):
            boxes = getattr(result, "obb", None)
            if boxes is None:
                boxes = getattr(result, "boxes", None)
            if boxes is None:
                raise RuntimeError(f"YOLO did not return detection boxes for {image}.")
            yield len(boxes) == 0
        if offset == 0 or (offset + len(group)) % 50 == 0 or offset + len(group) == len(images):
            print(f"SCAN_PROGRESS {min(offset + len(group), len(images))}/{len(images)}", flush=True)


def scan_dataset(settings, plan_path):
    values = validate_settings(settings)
    source = Path(values["source"])
    pairs = discover_pairs(source)
    print(f"Scanning {len(pairs)} images in {source}", flush=True)
    if values["method"] == "model":
        os.environ["YOLO_AUTOINSTALL"] = "False"
        from ultralytics import YOLO
        print(f"Using YOLO model: {values['model']} | conf={values['conf']:g} | imgsz={values['imgsz']}", flush=True)
        empties = _detect_empty(YOLO(values["model"]), [image for image, _ in pairs], values)
    else:
        empties = (bool(label and not label.read_text(encoding="utf-8", errors="replace").strip()) or
                   bool(label is None and values["include_missing"]) for _, label in pairs)
    removed, all_images = [], []
    missing = 0
    for i, ((image, label), empty) in enumerate(zip(pairs, empties), 1):
        if label is None:
            missing += 1
        rel_image = str(image.relative_to(source))
        all_images.append(rel_image)
        if empty:
            removed.append({"image": rel_image, "label": str(label.relative_to(source)) if label else None,
                            "image_stat": _fingerprint(image),
                            "label_stat": _fingerprint(label) if label else None})
        if values["method"] == "labels" and (i % 1000 == 0 or i == len(pairs)):
            print(f"SCAN_PROGRESS {i}/{len(pairs)}", flush=True)
    manifest = {"format": PLAN_FORMAT, "settings": values, "total": len(pairs),
                "remove": removed, "all_images": all_images, "missing_labels": missing,
                "scanned_at": datetime.now(timezone.utc).isoformat()}
    plan_path = Path(plan_path).expanduser().resolve()
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    temp = plan_path.with_name(plan_path.name + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temp.replace(plan_path)
    print(f"SCAN_COMPLETE images={len(pairs)} remove={len(removed)} keep={len(pairs)-len(removed)} "
          f"missing_labels={missing} plan={plan_path}", flush=True)
    return manifest


def load_plan(plan_path):
    manifest = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if manifest.get("format") != PLAN_FORMAT:
        raise ValueError("Invalid or unsupported sanitisation preview plan.")
    return manifest


def _validated_plan(manifest):
    settings = validate_settings(manifest["settings"])
    source = Path(settings["source"])
    if settings != manifest["settings"]:
        raise ValueError("Sanitisation settings changed since preview; scan again.")
    pairs = discover_pairs(source)
    current_images = [str(p.relative_to(source)) for p, _ in pairs]
    if current_images != manifest["all_images"]:
        raise ValueError("Dataset images changed since preview; scan again.")
    label_locations = {str(image.relative_to(source)): label for image, label in pairs}
    candidates = []
    for record in manifest["remove"]:
        image = source / record["image"]
        label = source / record["label"] if record["label"] else None
        if not _within(image.resolve(), source) or (label and not _within(label.resolve(), source)):
            raise ValueError("Sanitisation plan contains a path outside the dataset.")
        if _fingerprint(image) != record["image_stat"]:
            raise ValueError(f"Image changed since preview; scan again: {image}")
        if label is not None and _fingerprint(label) != record["label_stat"]:
            raise ValueError(f"Label changed since preview; scan again: {label}")
        if label is None and label_locations.get(record["image"]) is not None:
            raise ValueError("A label appeared since preview; scan again.")
        candidates.append((image, label))
    return settings, candidates


def _path_line_target(line, manifest_folder, source):
    stripped = line.strip().strip('"\'')
    if not stripped or stripped.startswith("#"):
        return None
    path = Path(stripped).expanduser()
    if path.is_absolute():
        return path.resolve()
    # Ultralytics split lists may be based on the text-file location or dataset root.
    from_list = (manifest_folder / path).resolve()
    if from_list.is_file() or _within(from_list, source):
        return from_list
    return (source / path).resolve()


def _rewrite_manifests(dest, source, remove_set, *, inplace=False):
    """Keep common Ultralytics text-based split manifests consistent."""
    rewritten, dropped = 0, 0
    for manifest in dest.rglob("*.txt"):
        if manifest.name.lower() not in SPLIT_LISTS or manifest.is_symlink():
            continue
        original = manifest.read_text(encoding="utf-8")
        lines = []
        for line in original.splitlines(keepends=True):
            path = _path_line_target(line, manifest.parent, dest if inplace else source)
            if path is None:
                lines.append(line)
                continue
            # For copies, resolve paths relative to their original directory.
            if not inplace and not Path(line.strip()).is_absolute():
                path = _path_line_target(line, source / manifest.parent.relative_to(dest), source)
            if path is not None and path in remove_set:
                dropped += 1
                continue
            if not inplace and path is not None and _within(path, source):
                if Path(line.strip()).is_absolute():
                    new_path = dest / path.relative_to(source)
                    line = line.replace(line.strip(), str(new_path))
            lines.append(line)
        content = "".join(lines)
        if content != original:
            temp = manifest.with_name(manifest.name + ".sanitise-tmp")
            temp.write_text(content, encoding="utf-8")
            temp.replace(manifest)
            rewritten += 1
    if not inplace:
        for yaml_file in list(dest.rglob("*.yaml")) + list(dest.rglob("*.yml")):
            if yaml_file.is_symlink():
                continue
            content = yaml_file.read_text(encoding="utf-8")
            # Rewrite absolute paths in common Ultralytics YAML keys, leaving other content alone.
            def replace_path(match):
                value = match.group(2).strip()
                quote = value[:1] if value[:1] in ("'", '"') else ""
                bare = value.strip("'\"")
                p = Path(bare)
                if not p.is_absolute() or not _within(p, source):
                    return match.group(0)
                rewritten_path = str(dest / p.relative_to(source))
                return match.group(1) + (quote + rewritten_path + quote if quote else rewritten_path)
            revised = re.sub(r"(?m)^(\s*(?:path|train|val|test):\s*)([^\n#]+)", replace_path, content)
            if revised != content:
                yaml_file.write_text(revised, encoding="utf-8")
    return rewritten, dropped


def apply_plan(plan_path):
    manifest = load_plan(plan_path)
    values, candidates = _validated_plan(manifest)
    source = Path(values["source"])
    remove_set = {p for image, label in candidates for p in (image, label) if p is not None}
    print(f"Applying {len(candidates)} removals: {values['action']}", flush=True)
    if values["action"] == "copy":
        output = _validate_output(source, values["output"])
        staging = output.with_name(output.name + ".sanitise-incomplete-" + uuid.uuid4().hex[:8])
        def ignore(current, names):
            folder = Path(current)
            return [name for name in names if (folder / name).is_symlink() or (folder / name) in remove_set]
        try:
            shutil.copytree(source, staging, ignore=ignore)
            # Rewrite output references to the staging folder, then rename via a second pass.
            _rewrite_manifests(staging, source, remove_set)
            staging.rename(output)
            # Rewrite any absolute path references that were written against staging.
            for filename in list(output.rglob("*.yaml")) + list(output.rglob("*.yml")) + [
                    f for f in output.rglob("*.txt") if f.name.lower() in SPLIT_LISTS]:
                if filename.is_symlink():
                    continue
                text = filename.read_text(encoding="utf-8")
                if str(staging) in text:
                    filename.write_text(text.replace(str(staging), str(output)), encoding="utf-8")
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        location = output
    else:
        # Record planned deletions before removing anything; this is not an undo backup.
        audit_path = Path(plan_path).with_name(Path(plan_path).stem + "_delete_audit.json")
        audit_path.write_text(json.dumps({"source": str(source), "removals": manifest["remove"],
                                          "created": datetime.now(timezone.utc).isoformat()}, indent=2),
                              encoding="utf-8")
        _rewrite_manifests(source, source, remove_set, inplace=True)
        for index, (image, label) in enumerate(candidates, 1):
            image.unlink()
            if label is not None:
                label.unlink()
            if index % 100 == 0 or index == len(candidates):
                print(f"DELETE_PROGRESS {index}/{len(candidates)}", flush=True)
        location = source
    print(f"APPLY_COMPLETE removed={len(candidates)} retained={manifest['total']-len(candidates)} "
          f"destination={location}", flush=True)
    return {"removed": len(candidates), "retained": manifest["total"] - len(candidates),
            "destination": str(location), "action": values["action"]}
