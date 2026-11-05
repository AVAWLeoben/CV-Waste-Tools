"""True subprocess/dunder-main smoke test using a local fake Ultralytics wheel."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest


class ExportWorkerSubprocessTests(unittest.TestCase):
    def test_export_worker_process_returns_output_event(self):
        with tempfile.TemporaryDirectory() as rootstr:
            root=Path(rootstr)
            pkg=root/'ultralytics';pkg.mkdir()
            (pkg/'__init__.py').write_text('''from pathlib import Path
class YOLO:
    def __init__(self, path): self.path = Path(path)
    def export(self, **kw):
        assert kw["format"] == "onnx"
        assert kw["imgsz"] == (640, 1088)
        out = self.path.with_suffix(".onnx")
        out.write_bytes(b"fake onnx")
        return str(out)
''')
            (pkg/'cfg.py').write_text('DEFAULT_CFG_DICT={"quantize": None}\n')
            # Native worker probes its runtime, so supply the actual supported
            # format list and lightweight fake PyTorch CUDA information.
            engine=pkg/'engine';engine.mkdir()
            (engine/'__init__.py').write_text('')
            (engine/'exporter.py').write_text('def export_formats(): return {"Argument":["onnx","engine","litert"]}\n')
            (pkg/'__init__.py').write_text((pkg/'__init__.py').read_text() + '\n__version__="8.4.test"\n')
            weights=root/'model.pt';weights.write_bytes(b'weights')
            cfg=root/'request.json'
            cfg.write_text(json.dumps({'model':str(weights),'output':str(root/'out'),'format':'onnx',
                                       'imgsz':'640x1088','batch':1,'precision':'fp32'}),encoding='utf-8')
            worker=Path(__file__).resolve().parents[1]/'run_model_export.py'
            env=os.environ.copy();env['PYTHONPATH']=str(root)
            run=subprocess.run([sys.executable,str(worker),str(cfg)],env=env,capture_output=True,
                               text=True,check=False,timeout=30)
            self.assertEqual(run.returncode,0,(run.stdout,run.stderr))
            line=next(x for x in run.stdout.splitlines() if x.startswith('STUDIO_EXPORT_RESULT='))
            result=json.loads(line.partition('=')[2])
            self.assertTrue(Path(result['exported_model']).is_file())
            self.assertEqual(weights.read_bytes(),b'weights')
            self.assertTrue((Path(result['output_directory'])/'export_summary.json').exists())

if __name__=='__main__': unittest.main()
