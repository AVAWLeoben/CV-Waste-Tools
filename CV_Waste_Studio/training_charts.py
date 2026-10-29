"""Small Tkinter charts for live Ultralytics epoch metrics (no plotting dependency)."""
from __future__ import annotations

import math
import time
import tkinter as tk

PLOTS = (
    ("Training losses", (("train/box_loss", "box", "#3C70D7"),
                         ("train/cls_loss", "cls", "#E68C3A"),
                         ("train/dfl_loss", "dfl", "#27A28C")), False),
    ("Validation losses", (("val/box_loss", "box", "#3C70D7"),
                           ("val/cls_loss", "cls", "#E68C3A"),
                           ("val/dfl_loss", "dfl", "#27A28C")), False),
    ("Precision / recall", (("metrics/precision(B)", "precision", "#3C70D7"),
                            ("metrics/recall(B)", "recall", "#E68C3A")), True),
    ("Detection mAP", (("metrics/mAP50(B)", "mAP50", "#3C70D7"),
                       ("metrics/mAP50-95(B)", "mAP50–95", "#E68C3A")), True),
)


def epoch_ticks(first, last, max_ticks=7):
    """Readable integer epoch ticks, including the first and latest epoch."""
    if first >= last:
        return [first]
    span = last - first
    target_step = span / max(1, max_ticks - 1)
    power = 10 ** math.floor(math.log10(max(1, target_step)))
    candidates = sorted({max(1, int(multiple * power * factor))
                         for factor in (0.1, 1, 10)
                         for multiple in (1, 2, 5, 10)})
    step = min(candidates, key=lambda candidate: abs(math.log(candidate / target_step)))
    ticks = [first]
    ticks.extend(epoch for epoch in range(((first // step) + 1) * step, last, step)
                 if epoch > first)
    ticks.append(last)
    return ticks


class EpochHistory:
    """Store epoch-indexed measurements; an updated final epoch replaces its old value."""

    def __init__(self):
        self.epochs = {}
        self.total = 0

    def reset(self):
        self.epochs.clear()
        self.total = 0

    def add(self, event):
        epoch = event["epoch"]
        self.total = max(self.total, event["epochs"])
        old = self.epochs.get(epoch, {})
        merged = dict(old)
        merged.update(event["metrics"])
        if old == merged:
            return False
        self.epochs[epoch] = merged
        return True

    def series(self, key):
        return [(epoch, values[key]) for epoch, values in sorted(self.epochs.items())
                if key in values and math.isfinite(values[key])]


class LiveMetricsGraph(tk.Canvas):
    """Clickable four-chart activity view, with an animated single-chart zoom."""

    ANIMATION_SECONDS = 0.22

    def __init__(self, parent, **kwargs):
        super().__init__(parent, height=240, background="#F8FAFD", bd=0,
                         highlightthickness=1, highlightbackground="#E1E7F1", **kwargs)
        self.history = EpochHistory()
        self.expanded = None  # Plot index; None means 2 × 2 overview.
        self.hovered = None
        self.close_hovered = False
        self.inspected_epoch = None
        self._transition = None
        self._animation_job = None
        self.bind("<Configure>", lambda _event: self.redraw())
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Escape>", lambda _event: self.collapse())

    def reset(self):
        self._stop_animation()
        self.expanded = self.hovered = None
        self.close_hovered = False
        self.inspected_epoch = None
        self.configure(cursor="")
        self.history.reset()
        self.redraw()

    def record(self, event):
        if self.history.add(event):
            self.redraw()

    def _tile(self, index, width, height):
        return ((index % 2) * width / 2, (index // 2) * height / 2,
                width / 2, height / 2)

    @staticmethod
    def _close_bounds(width):
        return width - 38, 10, width - 11, 35

    def _hit_tile(self, x, y):
        width, height = self.winfo_width(), self.winfo_height()
        if x < 0 or y < 0 or x >= width or y >= height:
            return None
        return int(x >= width / 2) + 2 * int(y >= height / 2)

    def _on_motion(self, event):
        if self._transition is not None:
            return
        if self.expanded is None:
            hover = self._hit_tile(event.x, event.y)
            if hover != self.hovered:
                self.hovered = hover
                self.configure(cursor="hand2" if hover is not None else "")
                self.redraw()
        else:
            x0, y0, x1, y1 = self._close_bounds(self.winfo_width())
            hovering = x0 <= event.x <= x1 and y0 <= event.y <= y1
            selected = None
            if not hovering and self.history.epochs:
                left, right = 49, max(50, self.winfo_width() - 18)
                if left <= event.x <= right:
                    lo, hi = min(self.history.epochs), max(self.history.epochs)
                    target = lo if lo == hi else lo + (hi - lo) * (event.x - left) / (right - left)
                    selected = min(self.history.epochs, key=lambda epoch: abs(epoch - target))
            if hovering != self.close_hovered or selected != self.inspected_epoch:
                self.close_hovered = hovering
                self.inspected_epoch = selected
                self.configure(cursor="hand2" if hovering else "")
                self.redraw()

    def _on_leave(self, _event):
        if self.hovered is not None or self.close_hovered or self.inspected_epoch is not None:
            self.hovered = None
            self.close_hovered = False
            self.inspected_epoch = None
            if self._transition is None:
                self.redraw()
        self.configure(cursor="")

    def _on_click(self, event):
        if self._transition is not None:
            return
        self.focus_set()
        if self.expanded is None:
            index = self._hit_tile(event.x, event.y)
            if index is not None:
                self._animate(index, opening=True)
        else:
            x0, y0, x1, y1 = self._close_bounds(self.winfo_width())
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                self.collapse()

    def collapse(self):
        if self.expanded is not None and self._transition is None:
            self._animate(self.expanded, opening=False)

    def _stop_animation(self):
        if self._animation_job is not None:
            try:
                self.after_cancel(self._animation_job)
            except tk.TclError:
                pass
        self._animation_job = None
        self._transition = None

    def _animate(self, index, opening):
        self._stop_animation()
        self.hovered = None
        self.close_hovered = False
        self.inspected_epoch = None
        self.configure(cursor="")
        self._transition = {"index": index, "opening": opening,
                            "start": time.monotonic(), "progress": 0.0}
        self._animation_frame()

    def _animation_frame(self):
        transition = self._transition
        if transition is None:
            return
        t = min(1.0, (time.monotonic() - transition["start"]) / self.ANIMATION_SECONDS)
        transition["progress"] = t * t * (3 - 2 * t)  # ease-in-out
        if t >= 1.0:
            self.expanded = transition["index"] if transition["opening"] else None
            self._transition = None
            self._animation_job = None
            self.redraw()
        else:
            self.redraw()
            self._animation_job = self.after(16, self._animation_frame)

    def redraw(self):
        self.delete("all")
        width, height = self.winfo_width(), self.winfo_height()
        if width < 100 or height < 120:
            return
        if self._transition is not None:
            transition = self._transition
            index = transition["index"]
            for i in range(len(PLOTS)):
                if i != index:
                    self._draw_plot(*self._tile(i, width, height), i, animated=True)
            start = self._tile(index, width, height)
            full = (0, 0, width, height)
            progress = transition["progress"]
            if not transition["opening"]:
                progress = 1 - progress
            position = tuple(a + (b - a) * progress for a, b in zip(start, full))
            self._draw_plot(*position, index, animated=True)
        elif self.expanded is not None:
            self._draw_plot(0, 0, width, height, self.expanded, enlarged=True)
            if self.inspected_epoch is not None:
                self._draw_epoch_inspector(width, height, self.expanded, self.inspected_epoch)
            self._draw_close(width)
        else:
            for i in range(len(PLOTS)):
                self._draw_plot(*self._tile(i, width, height), i, hovered=i == self.hovered)

    def _draw_epoch_inspector(self, width, height, plot_index, epoch):
        """Show recorded (not interpolated) values under the cursor."""
        data = self.history.epochs.get(epoch)
        if not data:
            return
        _, traces, _ = PLOTS[plot_index]
        rows = [(label, data[key], color) for key, label, color in traces if key in data]
        if not rows:
            return
        lines = len(rows)
        left, top = max(8, width - 213), max(75, height - 45 - lines * 20)
        self.create_rectangle(left, top, width - 13, top + 31 + lines * 20,
                              fill="#F9FBFF", outline="#89B6F3", width=2)
        self.create_text(left + 12, top + 15, text=f"Epoch {epoch}", anchor="w",
                         fill="#20334D", font=("Segoe UI", 10, "bold"))
        for i, (label, value, color) in enumerate(rows):
            y = top + 37 + i * 20
            self.create_oval(left + 12, y - 3, left + 19, y + 4,
                             fill=color, outline=color)
            self.create_text(left + 28, y, text=f"{label}: {value:.4f}", anchor="w",
                             fill="#34455E", font=("Segoe UI", 9))

    def _draw_close(self, width):
        x0, y0, x1, y1 = self._close_bounds(width)
        self.create_rectangle(x0, y0, x1, y1, outline="#7FA8E7" if self.close_hovered
                              else "#CAD7E8", fill="#DFEBFF" if self.close_hovered
                              else "#F4F7FC", width=1)
        self.create_text((x0 + x1) / 2, (y0 + y1) / 2, text="×",
                         fill="#1F4A8E", font=("Segoe UI", 13, "bold"))

    def _draw_plot(self, x0, y0, width, height, index, hovered=False, enlarged=False,
                   animated=False):
        title, traces, normalized = PLOTS[index]
        if hovered:
            # Soft blue focus halo: visual indication that the tile is clickable.
            for inset, shade, thickness in ((1, "#CFE3FF", 3), (3, "#91BEFF", 2),
                                            (5, "#438CF5", 1)):
                self.create_rectangle(x0 + inset, y0 + inset,
                                      x0 + width - inset, y0 + height - inset,
                                      outline=shade, width=thickness)
        self.create_rectangle(x0 + 7, y0 + 7, x0 + width - 7, y0 + height - 7,
                              outline="#89B6F3" if hovered else "#E5EBF5", fill="#FFFFFF",
                              width=1)
        self.create_text(x0 + 14, y0 + 18 if enlarged else y0 + 14,
                         text=title, anchor="w", fill="#20334D",
                         font=("Segoe UI", 11 if enlarged else 9, "bold"))
        if hovered and width > 210:
            self.create_text(x0 + width - 16, y0 + 15, text="Click to expand",
                             anchor="e", fill="#3676D7", font=("Segoe UI", 8))
        xlegend = x0 + 14
        legend_y = y0 + (43 if enlarged else 34)
        for key, label, color in traces:
            self.create_line(xlegend, legend_y, xlegend + 11, legend_y,
                             fill=color, width=2)
            self.create_oval(xlegend + 4, legend_y - 3, xlegend + 10, legend_y + 3,
                             fill=color, outline="white")
            self.create_text(xlegend + 15, legend_y, text=label,
                             anchor="w", fill="#53657A", font=("Segoe UI", 8))
            xlegend += 27 + 6 * len(label)
        px0, px1 = x0 + 49, x0 + width - 18
        py0, py1 = y0 + (59 if enlarged else 49), y0 + height - 27
        if py1 <= py0 or px1 <= px0:
            return
        points = {key: self.history.series(key) for key, _, _ in traces}
        values = [v for line in points.values() for _, v in line]
        if not values:
            self.create_text((px0 + px1) / 2, (py0 + py1) / 2,
                             text="Waiting for epoch metrics", fill="#8392A7",
                             font=("Segoe UI", 9))
            return
        if normalized:
            low, high = 0.0, 1.0
        else:
            low, high = min(values), max(values)
            spread = high - low
            pad = max(spread * 0.12, abs(high) * 0.04, 0.01)
            low, high = max(0.0, low - pad), high + pad
            if high <= low:
                high = low + 1
        for fraction in (0.0, 0.5, 1.0):
            y = py1 - (py1 - py0) * fraction
            self.create_line(px0, y, px1, y, fill="#E8EDF4", dash=(2, 3))
            self.create_text(px0 - 6, y,
                             text=f"{low + (high - low) * fraction:.2f}",
                             anchor="e", fill="#71849C", font=("Segoe UI", 8 if enlarged else 7))
        x_min, x_max = min(self.history.epochs), max(self.history.epochs)

        def xy(epoch, value):
            xx = (px0 + px1) / 2 if x_min == x_max else (
                px0 + (px1 - px0) * (epoch - x_min) / (x_max - x_min))
            yy = py1 - (py1 - py0) * (value - low) / (high - low)
            return xx, max(py0, min(py1, yy))

        ticks = epoch_ticks(x_min, x_max, 11 if enlarged else 6)
        previous_x = -100.0
        for epoch in ticks:
            x = xy(epoch, low)[0]
            # Always show the first and latest epoch; avoid cramped middle ticks.
            if epoch not in (x_min, x_max) and (
                    x - previous_x < 25 or px1 - x < 24):
                continue
            self.create_line(x, py0, x, py1, fill="#F1F4F9", dash=(2, 4))
            self.create_line(x, py1, x, py1 + 4, fill="#8C9DB3")
            self.create_text(x, py1 + 14, text=str(epoch), anchor="center",
                             fill="#60748E", font=("Segoe UI", 8 if enlarged else 7))
            previous_x = x

        # Mark the actual epoch measurements; straight lines only interpolate
        # between those recorded epochs, never inventing data points.
        for key, _, color in traces:
            line = points[key]
            coords = [position for epoch, value in line for position in xy(epoch, value)]
            if len(coords) >= 4:
                self.create_line(*coords, fill=color, width=2 if enlarged else 1.8,
                                 smooth=False)
            stride = max(1, math.ceil(len(line) / 120)) if animated else 1
            radius = 2.7 if enlarged else (2.2 if len(line) < 100 else 1.4)
            for point_index in range(0, len(coords), 2 * stride):
                xx, yy = coords[point_index:point_index + 2]
                self.create_oval(xx - radius, yy - radius, xx + radius, yy + radius,
                                 fill=color, outline="white" if radius > 2 else color,
                                 width=0.8)
