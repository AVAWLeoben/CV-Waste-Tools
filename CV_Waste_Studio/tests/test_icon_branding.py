from pathlib import Path
import unittest
import sys
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from studio_branding import ICON_ICO, ICON_PNG

class IconAssets(unittest.TestCase):
    def test_versioned_icon_present(self):
        self.assertTrue(ICON_ICO.is_file())
        self.assertEqual(ICON_ICO.name, "cv_waste_studio_windows_v5.ico")
        self.assertGreater(ICON_ICO.stat().st_size, 10000)
    def test_original_brand_png_preserved(self):
        self.assertTrue(ICON_PNG.is_file())
    def test_shortcut_references_same_icon(self):
        s = (root / "CREATE_CV_WASTE_STUDIO_SHORTCUTS.vbs").read_text()
        self.assertIn(ICON_ICO.name, s)

if __name__ == "__main__": unittest.main()
