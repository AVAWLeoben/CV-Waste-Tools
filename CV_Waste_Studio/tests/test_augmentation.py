import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from augmentation_options import validate_augmentation, GENTLE

class AugmentationTests(unittest.TestCase):
    def test_defaults_omitted(self):
        self.assertEqual(validate_augmentation(None),{})
        self.assertEqual(validate_augmentation({'mosaic':'','degrees':' '}),{})
    def test_preset_json_roundtrip(self):
        self.assertEqual(validate_augmentation(json.loads(json.dumps(GENTLE))),GENTLE)
    def test_bad_values_rejected(self):
        for value in [{'mosaic':1.1},{'degrees':-1},{'hsv_s':'nan'},{'scale':'inf'},
                      {'mixup':True},{'close_mosaic':1.5},{'batch':8},{'hsv_h':'oops'}]:
            with self.subTest(value=value),self.assertRaises(ValueError):validate_augmentation(value)
    def test_integer_epochs(self):
        self.assertIsInstance(validate_augmentation({'close_mosaic':'10'})['close_mosaic'],int)
