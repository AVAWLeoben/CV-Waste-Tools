"""Readable epoch logging without depending on PyTorch or Tk."""
import logging
import json
import math
import re

ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')

def readable_line(message):
    text = ANSI.sub('', str(message))
    if re.search(r'\bEpoch\s+GPU_mem\s+box_loss\s+cls_loss', text):
        return 'Training epoch started — losses will be reported at epoch end.'
    parts = text.split()
    if len(parts) == 7 and parts[0] == 'all':
        try:
            nums = [float(x) for x in parts[1:]]
            if not all(math.isfinite(x) for x in nums): return text
            if not nums[0].is_integer() or not nums[1].is_integer(): return text
        except ValueError:
            return text
        names = ['images','objects','precision','recall','mAP50','mAP50–95']
        return 'Validation | ' + ' | '.join(f'{k}={v}' for k,v in zip(names, parts[1:]))
    return text

# A machine-readable line emitted by Ultralytics training callbacks.  The GUI
# consumes it to update the in-epoch progress bar without appending every tick
# to the visible activity log.
BATCH_PROGRESS_PREFIX = "STUDIO_TRAIN_PROGRESS"
BATCH_PROGRESS_PATTERN = re.compile(
    r"^STUDIO_TRAIN_PROGRESS epoch=(\d+) epochs=(\d+) iteration=(\d+) iterations=(\d+)$"
)


def format_iteration_progress(epoch, epochs, iteration, iterations):
    return (f"{BATCH_PROGRESS_PREFIX} epoch={epoch} epochs={epochs} "
            f"iteration={iteration} iterations={iterations}")


def parse_iteration_progress(line):
    match = BATCH_PROGRESS_PATTERN.fullmatch(ANSI.sub("", str(line)).strip())
    if match is None:
        return None
    epoch, epochs, iteration, iterations = map(int, match.groups())
    if epochs < 1 or epoch < 1 or epoch > epochs or iterations < 1 or not 0 <= iteration <= iterations:
        return None
    return epoch, epochs, iteration, iterations


# Emit one structured event per completed epoch. Kept in the ordinary job log
# for troubleshooting, but consumed silently by the application's live graph.
EPOCH_METRICS_PREFIX = "STUDIO_TRAIN_METRICS "
EPOCH_METRIC_KEYS = frozenset((
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
))


def epoch_metrics_from_trainer(trainer):
    """Snapshot native Ultralytics losses and validation metrics, without tensors."""
    epoch = int(trainer.epoch) + 1
    total = int(trainer.epochs)
    result = {}
    losses = getattr(trainer, "tloss", None)
    if losses is not None:
        if isinstance(losses, dict):
            loss_items = losses.items()
        else:
            if hasattr(losses, "detach"):
                losses = losses.detach().cpu()
            if hasattr(losses, "tolist"):
                losses = losses.tolist()
            if not isinstance(losses, (tuple, list)):
                losses = [losses]
            names = getattr(trainer, "loss_names", ())
            if isinstance(names, str):
                names = [names]
            loss_items = zip(names, losses)
        for name, value in loss_items:
            key = "train/" + str(name).removeprefix("train/")
            if key in EPOCH_METRIC_KEYS:
                try:
                    number = float(value)
                    if math.isfinite(number):
                        result[key] = number
                except (ValueError, TypeError, OverflowError):
                    continue
    metrics = getattr(trainer, "metrics", None)
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if key in EPOCH_METRIC_KEYS:
                try:
                    number = float(value)
                    if math.isfinite(number):
                        result[key] = number
                except (ValueError, TypeError, OverflowError):
                    continue
    return {"epoch": epoch, "epochs": total, "metrics": result}


def format_epoch_metrics(event):
    return EPOCH_METRICS_PREFIX + json.dumps(event, separators=(",", ":"), allow_nan=False)


def parse_epoch_metrics(line):
    clean = ANSI.sub("", str(line)).strip()
    if not clean.startswith(EPOCH_METRICS_PREFIX):
        return None
    try:
        event = json.loads(clean[len(EPOCH_METRICS_PREFIX):])
        epoch, epochs = event["epoch"], event["epochs"]
        metrics = event["metrics"]
        if (type(epoch) is not int or type(epochs) is not int or not 1 <= epoch <= epochs
                or not isinstance(metrics, dict)):
            return None
        checked = {}
        for key, value in metrics.items():
            if key not in EPOCH_METRIC_KEYS or type(value) not in (int, float):
                return None
            number = float(value)
            if not math.isfinite(number):
                return None
            checked[key] = number
        return {"epoch": epoch, "epochs": epochs, "metrics": checked}
    except (ValueError, TypeError, KeyError, OverflowError):
        return None


class ReadableTrainingFilter(logging.Filter):
    def filter(self, record):
        # Preserve exceptions, warnings, and all messages not matching known tables.
        if record.levelno < logging.WARNING:
            record.msg = readable_line(record.getMessage())
            record.args = ()
        return True

def loss_summary(trainer):
    losses = getattr(trainer, 'tloss', None)
    if losses is None: return 'losses unavailable'
    if isinstance(losses, dict):
        pairs = losses.items()
    else:
        if hasattr(losses, 'detach'): losses = losses.detach().cpu()
        if hasattr(losses, 'tolist'): losses = losses.tolist()
        if not isinstance(losses, (list,tuple)): losses = [losses]
        names = getattr(trainer, 'loss_names', ())
        if isinstance(names,str): names = [names]
        pairs = [(names[i] if i<len(names) else f'loss_{i+1}',v) for i,v in enumerate(losses)]
    output=[]
    for name,value in pairs:
        try: output.append(f'{name}={float(value):.4f}')
        except (ValueError,TypeError): continue
    return ' | '.join(output) or 'losses unavailable'

# Exact native Ultralytics run directory, including name auto-increment suffixes.
# Printed by the worker after model.train() returns, then consumed by Studio.
TRAIN_RUN_DIR_PREFIX = "STUDIO_TRAIN_RUN_DIR "

def format_training_run_dir(folder):
    return TRAIN_RUN_DIR_PREFIX + json.dumps(str(folder), ensure_ascii=False)

def parse_training_run_dir(line):
    clean = ANSI.sub("", str(line)).strip()
    if not clean.startswith(TRAIN_RUN_DIR_PREFIX):
        return None
    try:
        path = json.loads(clean[len(TRAIN_RUN_DIR_PREFIX):])
        return path if isinstance(path, str) and path.strip() else None
    except (ValueError, TypeError):
        return None
