"""Regression checks for duplicate Windows launches (no real Win32 required)."""
from unittest.mock import patch

import studio_single_instance as guard


class FakeKernel:
    def __init__(self, opened=0, created=42):
        self.opened = opened
        self.created = created
        self.closed = []

    def OpenMutexW(self, access, inherit, name):
        assert name.startswith("Local\\CVWasteStudio_")
        return self.opened

    def CreateMutexW(self, security, initial, name):
        assert name.startswith("Local\\CVWasteStudio_")
        return self.created

    def CloseHandle(self, handle):
        self.closed.append(handle)


def test_second_launch_finds_running_mutex_without_owning_it():
    api = FakeKernel(opened=13)
    with patch.object(guard, "_is_windows", return_value=True), patch.object(guard, "_kernel32", return_value=api):
        assert guard.another_instance_running() is True
        assert api.closed == [13]


def test_first_launch_does_not_confuse_missing_mutex_with_active_one():
    with patch.object(guard, "_is_windows", return_value=True), patch.object(guard, "_kernel32", return_value=FakeKernel()):
        assert guard.another_instance_running() is False


def test_duplicate_claim_closes_its_handle():
    api = FakeKernel(created=17)
    guard._INSTANCE_MUTEX = None
    with patch.object(guard, "_is_windows", return_value=True), patch.object(guard, "_kernel32", return_value=api), patch.object(guard.ctypes, "get_last_error", return_value=183, create=True):
        assert guard.claim_gui_instance() is False
        assert api.closed == [17]
        assert guard._INSTANCE_MUTEX is None


def test_owner_keeps_mutex_until_closure():
    api = FakeKernel(created=42)
    guard._INSTANCE_MUTEX = None
    with patch.object(guard, "_is_windows", return_value=True), patch.object(guard, "_kernel32", return_value=api), patch.object(guard.ctypes, "get_last_error", return_value=0, create=True):
        assert guard.claim_gui_instance() is True
        assert guard.claim_gui_instance() is True
        assert api.closed == []
        guard.release_gui_instance()
        assert api.closed == [42]
        assert guard._INSTANCE_MUTEX is None


def test_relaunch_exits_before_environment_or_gui_setup():
    import greenplast_launcher as launcher

    with patch.object(launcher, "another_instance_running", return_value=True), patch.object(launcher, "activate_existing_window") as focus, patch.object(launcher, "Studio") as studio:
        launcher.main([])
        focus.assert_called_once()
        studio.assert_not_called()


def test_explicit_new_instance_bypasses_mutex_and_opens_studio():
    import greenplast_launcher as launcher

    with patch.object(launcher, "another_instance_running") as check, \
         patch.object(launcher, "claim_gui_instance") as claim, \
         patch.object(launcher, "activate_existing_window") as focus, \
         patch.object(launcher, "Studio") as studio:
        launcher.main(["--new-instance"])
        check.assert_not_called()
        claim.assert_not_called()
        focus.assert_not_called()
        assert studio.call_args.kwargs["allow_multiple"] is True
        studio.return_value.run.assert_called_once_with()


def test_new_window_environment_flag_survives_batch_startup():
    import greenplast_launcher as launcher

    with patch.dict(launcher.os.environ, {"CV_WASTE_STUDIO_NEW_INSTANCE": "1"}), \
         patch.object(launcher, "another_instance_running") as check, \
         patch.object(launcher, "Studio") as studio:
        launcher.main([])
        check.assert_not_called()
        assert studio.call_args.kwargs["allow_multiple"] is True


def test_shortcut_installer_exposes_both_modes():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    shortcuts = (root / "CREATE_CV_WASTE_STUDIO_SHORTCUTS.vbs").read_text(encoding="utf-8")
    launcher = (root / "START_CV_WASTE_TOOLS.vbs").read_text(encoding="utf-8")
    assert 'CV Waste Studio - New Window.lnk' in shortcuts
    assert 'CV Waste Studio.lnk' in shortcuts
    assert 'link.Arguments = link.Arguments & " --new-instance"' in shortcuts
    assert 'shell.Environment("PROCESS")("CV_WASTE_STUDIO_NEW_INSTANCE") = "1"' in launcher
