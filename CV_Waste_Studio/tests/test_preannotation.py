"""Pre-annotation regression tests (no GPU/Ultralytics install required)."""
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preannotate_dataset as annotate


class FakeNumber:
    def __init__(self, value):
        self.value = value
    def item(self):
        return self.value


class FakeCoords:
    def __init__(self, coords):
        self.coords = coords
    def tolist(self):
        return self.coords


class FakeBox:
    def __init__(self, klass, coords):
        self.cls = FakeNumber(klass)
        self.xywhn = [FakeCoords(coords)]


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'images').mkdir()
        (self.root / 'images' / 'nested').mkdir()
        (self.root / 'images' / 'a.jpg').write_bytes(b'dummy')
        (self.root / 'images' / 'nested' / 'b.png').write_bytes(b'dummy')
        (self.root / 'labels').mkdir()
        (self.root / 'labels' / 'a.txt').write_text('7 0.5 0.5 0.1 0.1\n')
        self.calls = []

        calls = self.calls
        class FakeYOLO:
            task = 'detect'
            names = {0:'object', 7:'other'}
            def __init__(self, path):
                calls.append(('model', path))
            def predict(self, source, **kwargs):
                calls.append(('predict', str(source), kwargs))
                return iter([types.SimpleNamespace(boxes=[FakeBox(0, [0.5, 0.5, 0.3, 0.4])])])
        self.fake_ultra = types.ModuleType('ultralytics')
        self.fake_ultra.YOLO = FakeYOLO

    def run_annotation(self, **kwargs):
        with patch.dict(sys.modules, {'ultralytics': self.fake_ultra}):
            return annotate.preannotate_dataset(self.root, 'yolo11n.pt', **kwargs)

    def test_default_skip_existing_and_sibling_layout(self):
        out = self.run_annotation(conf=0.42, iou=0.6, max_det=77, imgsz=960, device='cpu')
        self.assertEqual(out['created'], 1)
        self.assertEqual(out['skipped'], 1)
        self.assertEqual((self.root/'labels'/'a.txt').read_text(), '7 0.5 0.5 0.1 0.1\n')
        self.assertEqual((self.root/'labels'/'nested'/'b.txt').read_text(),
                         '0 0.500000 0.500000 0.300000 0.400000\n')
        self.assertEqual(self.calls[1][2]['conf'], 0.42)
        self.assertEqual(self.calls[1][2]['iou'], 0.6)
        self.assertEqual(self.calls[1][2]['max_det'], 77)
        self.assertEqual(self.calls[1][2]['device'], 'cpu')
        self.assertEqual(self.calls[1][2]['imgsz'], 960)

    def test_overwrite_is_explicit(self):
        out = self.run_annotation(overwrite=True)
        self.assertEqual(out['created'], 2)
        self.assertEqual(out['skipped'], 0)
        self.assertEqual((self.root/'labels'/'a.txt').read_text().split()[0], '0')

    def test_validation_rejects_invalid_inference_values(self):
        for bad in ({'conf': 2}, {'iou': -1}, {'max_det':0}, {'imgsz':8},
                    {'device':'bogus'}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                annotate._settings(**({**dict(dataset_folder=self.root, model='yolo11n.pt',
                                          conf=0.25, iou=0.7, max_det=300, imgsz=1280,
                                          device='auto', labels_folder=None, overwrite=False), **bad}))

    def test_no_images_and_duplicate_stems(self):
        (self.root/'images'/'a.png').write_bytes(b'dummy')
        with self.assertRaisesRegex(ValueError, 'share the same label'):
            self.run_annotation()

    def test_flat_image_folder_labels_alongside_images(self):
        flat = self.root/'flat'
        flat.mkdir()
        (flat/'image.jpg').write_bytes(b'dummy')
        with patch.dict(sys.modules, {'ultralytics':self.fake_ultra}):
            annotate.preannotate_dataset(flat, 'yolo11n.pt')
        self.assertTrue((flat/'image.txt').exists())


if __name__ == '__main__':
    unittest.main()
