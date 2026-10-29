import logging
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training_log import readable_line, loss_summary, ReadableTrainingFilter
import train_and_split_tool_GUI as backend

class TrainingLogTests(unittest.TestCase):
    def test_validation_labels(self):
        line=readable_line(' all 46 48 0.656 0.816 0.752 0.572')
        self.assertIn('images=46 | objects=48',line)
        self.assertIn('precision=0.656',line)
        self.assertNotIn('box_loss',line)
    def test_loss_formats(self):
        class Tensor:
            def detach(self):return self
            def cpu(self):return self
            def tolist(self):return [.8,2.,1.1]
        for values in [Tensor(),[.8,2.,1.1],{'box_loss':.8,'cls_loss':2.,'dfl_loss':1.1}]:
            line=loss_summary(types.SimpleNamespace(tloss=values,loss_names=['box_loss','cls_loss','dfl_loss']))
            self.assertIn('cls_loss=2.0000',line)
    def test_errors_and_unknown_text_preserved(self):
        for line in ['Traceback: failure','WARNING: CUDA unavailable','something new','all incomplete']:
            self.assertEqual(readable_line(line),line)
        record=logging.LogRecord('x',logging.ERROR,'x',1,'all 46 48 .1 .2 .3 .4',(),None)
        ReadableTrainingFilter().filter(record)
        self.assertEqual(record.msg,'all 46 48 .1 .2 .3 .4')
    def test_cleanup_after_training_failure(self):
        logger=logging.getLogger('test-training-cleanup')
        original=lambda *a,**kw: None
        trainer=types.ModuleType('ultralytics.engine.trainer');trainer.TQDM=original
        validator=types.ModuleType('ultralytics.engine.validator');validator.TQDM=original
        engine=types.ModuleType('ultralytics.engine');engine.trainer=trainer;engine.validator=validator
        ultra=types.ModuleType('ultralytics');ultra.engine=engine
        utils=types.ModuleType('ultralytics.utils');utils.LOGGER=logger
        class Model:
            def __init__(self):self.callbacks={'on_fit_epoch_end':[], 'on_train_epoch_start':[], 'on_train_batch_end':[]}
            def add_callback(self,k,v):self.callbacks[k].append(v)
            def train(self,**kw):raise RuntimeError('test failure')
        model=Model()
        modules={'ultralytics':ultra,'ultralytics.engine':engine,'ultralytics.engine.trainer':trainer,'ultralytics.engine.validator':validator,'ultralytics.utils':utils}
        with patch.dict(sys.modules,modules):
            with self.assertRaisesRegex(RuntimeError,'test failure'):backend._train_with_compact_progress(model,{})
        self.assertEqual(logger.filters,[])
        self.assertEqual(model.callbacks['on_fit_epoch_end'],[])
        self.assertIs(trainer.TQDM,original)


class IterationProgressTests(unittest.TestCase):
    def test_protocol_validation_and_ansi(self):
        from training_log import format_iteration_progress, parse_iteration_progress
        for args in [(1, 2, 0, 5), (1, 2, 3, 5), (2, 2, 5, 5)]:
            self.assertEqual(parse_iteration_progress(format_iteration_progress(*args)), args)
            self.assertEqual(parse_iteration_progress("\x1b[32m" + format_iteration_progress(*args)), args)
        for text in ["Training | epoch=1/3", "STUDIO_TRAIN_PROGRESS epoch=0 epochs=5 iteration=0 iterations=12",
                     "STUDIO_TRAIN_PROGRESS epoch=1 epochs=5 iteration=7 iterations=6",
                     "STUDIO_TRAIN_PROGRESS epoch=1 epochs=5 iteration=1 iterations=0"]:
            self.assertIsNone(parse_iteration_progress(text))

    def test_ultralytics_callbacks_emit_first_last_and_reset_each_epoch(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from training_log import parse_iteration_progress
        import time

        logger = logging.getLogger('test-training-callback-progress')
        trainer_module = types.ModuleType('ultralytics.engine.trainer')
        trainer_module.TQDM = lambda *args, **kwargs: None
        validator_module = types.ModuleType('ultralytics.engine.validator')
        validator_module.TQDM = lambda *args, **kwargs: None
        engine = types.ModuleType('ultralytics.engine')
        engine.trainer = trainer_module
        engine.validator = validator_module
        ultra = types.ModuleType('ultralytics')
        ultra.engine = engine
        utils = types.ModuleType('ultralytics.utils')
        utils.LOGGER = logger

        class Model:
            def __init__(self):
                self.callbacks = {key: [] for key in (
                    'on_train_epoch_start', 'on_train_batch_end', 'on_fit_epoch_end')}
                self.args = None
            def add_callback(self, key, callback):
                self.callbacks[key].append(callback)
            def train(self, **kwargs):
                self.args = kwargs
                for epoch in range(2):
                    trainer = types.SimpleNamespace(
                        epoch=epoch, epochs=2, train_loader=list(range(5)),
                        tloss=[0.5, 0.8, 1.2], loss_names=['box_loss', 'cls_loss', 'dfl_loss'],
                        metrics={'metrics/recall(B)': 0.6})
                    for callback in self.callbacks['on_train_epoch_start']:
                        callback(trainer)
                    for _ in trainer.train_loader:
                        for callback in self.callbacks['on_train_batch_end']:
                            callback(trainer)
                    for callback in self.callbacks['on_fit_epoch_end']:
                        callback(trainer)
                return 'native-train-return'

        modules = {'ultralytics': ultra, 'ultralytics.engine': engine,
                   'ultralytics.engine.trainer': trainer_module,
                   'ultralytics.engine.validator': validator_module,
                   'ultralytics.utils': utils}
        for compact in (True, False):
            model = Model()
            output = StringIO()
            with patch.dict(sys.modules, modules), redirect_stdout(output):
                self.assertEqual(backend._train_with_compact_progress(
                    model, {'workers': 8, 'cache': 'disk'}, compact), 'native-train-return')
            self.assertEqual(model.args, {'workers': 8, 'cache': 'disk'})
            events = [parsed for line in output.getvalue().splitlines()
                      if (parsed := parse_iteration_progress(line))]
            self.assertEqual(events, [(1, 2, 0, 5), (1, 2, 5, 5),
                                      (2, 2, 0, 5), (2, 2, 5, 5)])
            self.assertEqual(len(output.getvalue().split('Training | epoch=')) - 1,
                             2 if compact else 0)
            self.assertTrue(all(not callbacks for callbacks in model.callbacks.values()))
            self.assertEqual(logger.filters, [])

    def test_gui_progress_updates_without_spamming_activity_log(self):
        from greenplast_launcher import Studio
        class DummyBar:
            def configure(self, **kwargs):
                self.value = kwargs['value']
        class DummyVar:
            def set(self, value):
                self.value = value
        class DummyStudio:
            iteration_bar = DummyBar()
            iteration_label = DummyVar()
            def __init__(self):
                self.lines = []
            def log_line(self, key, line):
                self.lines.append((key, line))
        studio = DummyStudio()
        Studio._training_output_line(studio, 'STUDIO_TRAIN_PROGRESS epoch=2 epochs=20 iteration=11 iterations=44')
        self.assertEqual(studio.iteration_bar.value, 25)
        self.assertIn('11/44 batches', studio.iteration_label.value)
        self.assertEqual(studio.lines, [])
        Studio._training_output_line(studio, 'STUDIO_TRAIN_PROGRESS epoch=2 epochs=20 iteration=44 iterations=44')
        self.assertIn('validating / saving', studio.iteration_label.value)
        Studio._training_output_line(studio, 'Training | epoch=2/20')
        self.assertEqual(studio.lines, [('train', 'Training | epoch=2/20')])
