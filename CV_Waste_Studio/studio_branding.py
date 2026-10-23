"""Apply the user's original CV Waste Studio branding to the unified GUI.

The assets directory is resolved relative to this module, so VBS, BAT, shortcut,
and direct-Python startup all find exactly the same files. No Pillow dependency
is required at application runtime: smaller GUI icons are prepared in assets.
"""
from __future__ import annotations

import sys
from pathlib import Path
import tkinter as tk

ASSET_DIR = Path(__file__).resolve().parent / "assets"
ICON_ICO = ASSET_DIR / "cv_waste_studio_windows_v5.ico"
ICON_PNG = ASSET_DIR / "cv_waste_studio.png"
SIDEBAR_ICON = ASSET_DIR / "cv_waste_studio_sidebar_48.png"
WINDOWS_APP_ID = "CVWasteStudio.UnifiedApplication"


def set_windows_app_id() -> None:
    """Register identity before the Tk root is created, for taskbar grouping."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except (AttributeError, OSError):
        pass


def apply_window_icon(root: tk.Tk) -> None:
    """Use user's artwork for titlebar, taskbar, and child windows/dialogs."""
    # Explicit native sizes improve Windows DPI scaling. The versioned ICO
    # path also bypasses stale Explorer shortcut-icon caches.
    icons = []
    for n in (16, 24, 32, 48, 64, 128, 256):
        icon = ASSET_DIR / f"cv_waste_studio_{n}.png"
        if icon.is_file():
            try:
                icons.append(tk.PhotoImage(master=root, file=str(icon)))
            except (tk.TclError, OSError):
                pass
    if not icons and ICON_PNG.is_file():
        try:
            icons.append(tk.PhotoImage(master=root, file=str(ICON_PNG)))
        except (tk.TclError, OSError):
            pass
    if icons:
        try:
            root.iconphoto(True, *icons)
            root._cv_waste_icon = icons[0]  # compatibility with GUI smoke check
            root._cv_waste_icons = icons      # prevent Tk image garbage collection
        except (tk.TclError, OSError):
            pass
    if sys.platform == "win32" and ICON_ICO.is_file():
        try:
            # Windows handles native .ico multi-size resources in window chrome.
            root.iconbitmap(str(ICON_ICO))
        except (tk.TclError, OSError):
            pass


def sidebar_icon(root: tk.Tk):
    """Return a 48x48 version of the original sidebar image."""
    if not SIDEBAR_ICON.is_file():
        return None
    try:
        return tk.PhotoImage(master=root, file=str(SIDEBAR_ICON))
    except (tk.TclError, OSError):
        return None
