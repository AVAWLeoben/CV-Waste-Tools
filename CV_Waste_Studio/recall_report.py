"""Standalone Recall@Precision result cards and saved-evaluation discovery.

Reads existing Ultralytics evaluation summaries and confidence curves; does not
rerun predictions or change the metric calculation.
"""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path

REPORT_NAME = "evaluation_summary.png"
SUMMARY_NAME = "operating_point_summary.json"


def saved_evaluations(folder: str | Path) -> list[Path]:
    """Newest completed evaluation folders, excluding the prediction cache."""
    base = Path(folder).expanduser()
    if not base.is_dir():
        return []
    return sorted(
        (p for p in base.glob("precision_recall_*")
         if p.is_dir() and (p / SUMMARY_NAME).is_file()),
        key=lambda p: p.name, reverse=True)


def load_evaluation(folder: str | Path) -> dict:
    """Reopen a saved evaluation without model loading or inference."""
    selected = Path(folder).expanduser().resolve()
    if selected.is_file() and selected.name == SUMMARY_NAME:
        selected = selected.parent
    path = selected / SUMMARY_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No completed evaluation found at: {path}")
    summary = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(summary, dict) or "target_reached" not in summary or "target_precision" not in summary:
        raise ValueError(f"Not a valid Recall@Precision result: {path}")
    summary["output_directory"] = str(selected)
    return summary


def _font(size: int, *, bold=False):
    from PIL import ImageFont
    candidates = (["C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
                  if bold else
                  ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _curve_points(folder: Path, summary: dict):
    name = Path(str(summary.get("curve_csv", ""))).name
    csv_path = folder / (name if name.endswith(".csv") else "food_precision_recall_by_confidence.csv")
    if not csv_path.is_file():
        csv_path = folder / "food_precision_recall_by_confidence.csv"
    if not csv_path.is_file():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        result = []
        for row in rows:
            try:
                result.append((float(row["confidence"]), float(row["precision"]), float(row["recall"])))
            except (ValueError, KeyError, TypeError):
                continue
    return sorted(result)


def save_evaluation_report(summary: dict, output: str | Path | None = None) -> Path:
    """Save a single readable, shareable PNG for a completed evaluation.

    The output is based on the saved metrics/CSV, so historical runs can be
    rendered again without running the model.
    """
    from PIL import Image, ImageDraw

    folder = Path(str(summary["output_directory"])).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {folder}")
    destination = Path(output) if output else folder / REPORT_NAME
    image = Image.new("RGB", (1440, 900), "#f3f6fb")
    draw = ImageDraw.Draw(image)
    navy, muted, blue, teal, green, orange = "#14243c", "#62758c", "#3168df", "#058f89", "#12845b", "#d97706"
    card = "#ffffff"
    draw.rounded_rectangle((28, 26, 1412, 874), radius=24, fill=card)
    draw.rounded_rectangle((28, 26, 1412, 160), radius=24, fill=navy)
    draw.rectangle((28, 130, 1412, 160), fill=navy)
    title = _font(39, bold=True)
    subtitle = _font(22)
    label = _font(21)
    value = _font(39, bold=True)
    fine = _font(18)
    draw.text((70, 57), "RECALL @ TARGET PRECISION", font=title, fill="#ffffff")
    run_id = folder.name.removeprefix("precision_recall_")
    draw.text((73, 119), f"CV Waste Studio  |  Evaluation {run_id}", font=subtitle, fill="#c8d8f0")
    reached = bool(summary.get("target_reached"))
    fill_status = "#e4f5ea" if reached else "#fff0da"
    color_status = green if reached else orange
    draw.rounded_rectangle((1130, 77, 1370, 122), radius=18, fill=fill_status)
    draw.text((1151, 87), "TARGET MET" if reached else "NOT REACHED", font=_font(20, bold=True), fill=color_status)

    class_name = str(summary.get("class_name", f"class_{summary.get('class_id', '?')}"))
    data_label = f"Target class: {class_name} (ID {summary.get('class_id', '?')})  |  Split: {summary.get('split', 'unknown')}"
    draw.text((70, 184), data_label[:106], font=_font(25, bold=True), fill=navy)
    model_name = Path(str(summary.get("model", "unknown"))).name
    draw.text((72, 224), f"Model: {model_name}   |   Images: {summary.get('image_count', '—')}", font=label, fill=muted)

    metrics = [
        ("TARGET PRECISION", summary.get("target_precision"), "percent"),
        ("ACHIEVED PRECISION", summary.get("achieved_precision"), "percent"),
        ("RECALL", summary.get("recall_at_precision"), "percent"),
        ("CONFIDENCE THRESHOLD", summary.get("confidence_threshold"), "conf"),
    ]
    for i, (heading, amount, fmt) in enumerate(metrics):
        left = 70 + i * 334
        draw.rounded_rectangle((left, 277, left+316, 395), radius=14, fill="#edf3fd")
        draw.text((left+17, 293), heading, font=_font(17, bold=True), fill=muted)
        if amount is None:
            shown = "N/A"
        elif fmt == "percent":
            shown = f"{float(amount):.2%}"
        else:
            shown = f"{float(amount):.5f}"
        draw.text((left+17, 322), shown, font=value, fill=navy if i != 2 else teal)

    count_fields = [("TRUE POSITIVES", "true_positives", green),
                    ("FALSE POSITIVES", "false_positives", orange),
                    ("FALSE NEGATIVES", "false_negatives", "#c14657")]
    for i, (title_text, key, color) in enumerate(count_fields):
        left = 70 + i * 445
        draw.rounded_rectangle((left, 417, left + 427, 501), radius=12, fill="#f6f8fc")
        draw.text((left+19, 430), title_text, font=_font(18, bold=True), fill=muted)
        draw.text((left+19, 453), str(summary.get(key, "—")), font=_font(30, bold=True), fill=color)

    draw.text((70, 534), "Confidence sweep", font=_font(25, bold=True), fill=navy)
    x0, y0, x1, y1 = 115, 585, 1192, 764
    for tick in (0, .25, .5, .75, 1):
        y = y1 - tick * (y1-y0)
        draw.line((x0, int(y), x1, int(y)), fill="#dbe4ef", width=2)
        draw.text((62, int(y)-12), f"{tick:.0%}", font=_font(15), fill=muted)
    for tick in (0, .25, .5, .75, 1):
        x = x0 + tick * (x1-x0)
        draw.text((int(x)-12, y1+9), f"{tick:.2f}", font=_font(15), fill=muted)
    draw.text((545, 796), "Confidence threshold", font=fine, fill=muted)
    draw.text((1230, 585), "Precision", font=_font(18, bold=True), fill=blue)
    draw.text((1230, 614), "Recall", font=_font(18, bold=True), fill=teal)
    draw.line((1205, 596, 1222, 596), fill=blue, width=5)
    draw.line((1205, 625, 1222, 625), fill=teal, width=5)
    points = _curve_points(folder, summary)
    # Downsample long sweeps only for drawing performance. All actual points
    # remain preserved in the original CSV.
    if len(points) > 4500:
        step = (len(points) - 1) / 4499
        points = [points[round(i*step)] for i in range(4500)]
    if points:
        px = lambda conf: x0 + max(0, min(1, conf))*(x1-x0)
        py = lambda rate: y1 - max(0, min(1, rate))*(y1-y0)
        for idx, color in ((1, blue), (2, teal)):
            line = [(round(px(row[0])), round(py(row[idx]))) for row in points]
            if len(line) >= 2:
                draw.line(line, fill=color, width=4, joint="curve")
            elif len(line) == 1:
                x, y = line[0]
                draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
        threshold = summary.get("confidence_threshold")
        if threshold is not None:
            x = px(float(threshold))
            draw.line((x, y0, x, y1), fill=orange, width=2)
            draw.text((min(x+8, x1-130), y0-27), "Selected cutoff", font=fine, fill=orange)
    else:
        draw.text((365, 665), "Confidence curve not available for this saved run", font=label, fill=muted)
    draw.line((70, 832, 1370, 832), fill="#dbe4ef", width=2)
    foot = (f"Matching IoU: {summary.get('iou_match_threshold', '—')}  |  "
            f"NMS IoU: {summary.get('nms_iou_threshold', '—')}  |  "
            "Instance-level, one-to-one same-class matching; not mass-weighted purity")
    draw.text((71, 841), foot[:143], font=_font(16), fill=muted)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)
    return destination
