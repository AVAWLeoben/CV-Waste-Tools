# -*- coding: utf-8 -*-
"""Windows-only single-instance guard for the CV Waste Studio GUI.

Only the managed Python GUI owns the mutex. The base Conda launcher checks
for it but never takes ownership, so the existing base -> env handoff works.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from ctypes import wintypes
from pathlib import Path

_INSTANCE_MUTEX = None
_ERROR_ALREADY_EXISTS = 183
_SYNCHRONIZE = 0x00100000


def _is_windows():
    return os.name == "nt"


def _mutex_name():
    root = str(Path(__file__).resolve().parent).casefold()
    suffix = hashlib.sha256(root.encode("utf-8")).hexdigest()[:24]
    return "Local\\CVWasteStudio_" + suffix


def _kernel32():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenMutexW.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.OpenMutexW.restype = wintypes.HANDLE
    kernel.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


def another_instance_running():
    """Check without acquiring a mutex (used by the base Conda launcher)."""
    if not _is_windows():
        return False
    kernel = _kernel32()
    handle = kernel.OpenMutexW(_SYNCHRONIZE, False, _mutex_name())
    if not handle:
        return False
    kernel.CloseHandle(handle)
    return True


def claim_gui_instance():
    """Return False when another managed GUI already owns the named mutex."""
    global _INSTANCE_MUTEX
    if not _is_windows():
        return True
    if _INSTANCE_MUTEX is not None:
        return True
    kernel = _kernel32()
    handle = kernel.CreateMutexW(None, False, _mutex_name())
    if not handle:
        raise OSError(ctypes.get_last_error(), "Could not create CV Waste Studio instance mutex")
    if ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
        kernel.CloseHandle(handle)
        return False
    _INSTANCE_MUTEX = handle
    return True


def release_gui_instance():
    global _INSTANCE_MUTEX
    if _INSTANCE_MUTEX is not None:
        _kernel32().CloseHandle(_INSTANCE_MUTEX)
        _INSTANCE_MUTEX = None


def activate_existing_window():
    """Best-effort restore/focus; Windows can refuse foreground activation."""
    if not _is_windows():
        return
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
    user32.FindWindowW.restype = wintypes.HWND
    user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
    user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
    user32.FlashWindow.argtypes = (wintypes.HWND, wintypes.BOOL)
    hwnd = user32.FindWindowW(None, "CV Waste Studio")
    if hwnd:
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        if not user32.SetForegroundWindow(hwnd):
            user32.FlashWindow(hwnd, True)
