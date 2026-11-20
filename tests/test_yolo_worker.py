"""Worker failure paths and parameter forwarding without a GPU or downloaded weights."""

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from utils_training import train_yolo_worker as worker

ROOT = Path(__file__).resolve().parents[1]


class YoloWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = self.root / "данные" / "data.yaml"
        self.data.parent.mkdir()
        self.data.write_text("names: [object]\n", encoding="utf-8")
        self.weights = self.root / "initial.pt"
        self.weights.write_bytes(b"test")
        self.params = {
            "model": str(self.weights),
            "data": str(self.data),
            "batch": -1,
            "imgsz": 1280,
            "epochs": 4,
            "device": "auto",
            "export_onnx": False,
            "plots": False,
        }
        self.models = []
        self.error_stage = None
        self.failure = RuntimeError("CUDA out of memory: original failure")
        self.stop = False
        self.task = "detect"
        self.module = ModuleType("ultralytics")
        self.module.__version__ = "worker-test"
        self.module.YOLO = self.factory
        self.onnx = ModuleType("onnx")
        self.onnx.load = lambda path: SimpleNamespace(
            opset_import=[SimpleNamespace(version=18)]
        )
        self.onnx.checker = SimpleNamespace(check_model=lambda graph: None)

    def factory(self, model_path):
        owner = self

        class Model:
            task = owner.task

            def __init__(self):
                self.callbacks = {}
                self.calls = {}
                self.run_dir = owner.root / f"фактический_запуск_{len(owner.models)}"
                self.path = model_path

            def add_callback(self, event, callback):
                self.callbacks[event] = callback

            def train(self, **kwargs):
                self.calls["train"] = kwargs
                if owner.error_stage == "train":
                    raise owner.failure
                self.trainer = SimpleNamespace(
                    args=SimpleNamespace(**kwargs),
                    epochs=kwargs["epochs"],
                    epoch=0,
                    batch_size=2 if kwargs["batch"] < 1 else kwargs["batch"],
                    amp=False,
                    device="cpu",
                    stop=False,
                    save_dir=self.run_dir,
                )
                self.callbacks["on_pretrain_routine_start"](self.trainer)
                self.callbacks["on_train_start"](self.trainer)
                if owner.stop:
                    worker.signal_handler(signal.SIGTERM, None)
                    self.callbacks["on_train_batch_end"](self.trainer)
                self.callbacks["on_train_epoch_end"](self.trainer)
                (self.run_dir / "weights").mkdir(parents=True)
                for name in ("best.pt", "last.pt"):
                    (self.run_dir / "weights" / name).write_bytes(b"checkpoint")
                # Real train() can return metrics without a useful save_dir.
                return SimpleNamespace()

            def val(self, **kwargs):
                self.calls["val"] = kwargs
                if owner.error_stage == "val":
                    raise owner.failure
                metric = SimpleNamespace(mp=0.8, mr=0.7, map50=0.9, map=0.6)
                return SimpleNamespace(box=metric, results_dict={})

            def export(self, **kwargs):
                self.calls["export"] = kwargs
                if owner.error_stage == "export":
                    raise ValueError("export test failure")
                path = self.run_dir / "weights" / "best.onnx"
                path.write_bytes(b"graph")
                return str(path)

        model = Model()
        self.models.append(model)
        return model

    def run_worker(self, params=None, params_file=False):
        params = self.params if params is None else params
        if params_file:
            path = self.root / "параметры.json"
            path.write_text(
                json.dumps(params, ensure_ascii=False), encoding="utf-8-sig"
            )
            arguments = ["--params_file", str(path)]
        else:
            arguments = ["--params", json.dumps(params, ensure_ascii=False)]
        output = io.StringIO()
        with (
            patch.dict(sys.modules, {"ultralytics": self.module, "onnx": self.onnx}),
            contextlib.redirect_stdout(output),
        ):
            status = worker.main(arguments)
        return status, output.getvalue()

    def read_meta(self, model, name="best"):
        return json.loads(
            (model.run_dir / "weights" / f"{name}_meta.json").read_text(
                encoding="utf-8"
            )
        )

    def test_autobatch_and_memory_fraction_are_not_replaced(self):
        for batch in (-1, 0.3, 3):
            with self.subTest(batch=batch):
                status, output = self.run_worker(dict(self.params, batch=batch))
                self.assertEqual(status, 0, output)
                model = self.models[-1]
                self.assertEqual(model.calls["train"]["batch"], batch)
                self.assertNotIn("device", model.calls["train"])
                metadata = self.read_meta(model)
                self.assertEqual(metadata["batch_requested"], batch)
                self.assertEqual(metadata["batch"], 2 if batch < 1 else batch)

    def test_validation_uses_selected_batch_and_exact_run_directory(self):
        status, output = self.run_worker()
        self.assertEqual(status, 0, output)
        model = self.models[-1]
        self.assertEqual(model.calls["val"]["batch"], 2)
        self.assertEqual(model.calls["val"]["imgsz"], 1280)
        self.assertFalse(model.calls["val"]["half"])
        self.assertEqual(model.calls["val"]["project"], str(model.run_dir))
        self.assertIn(f"RESULT_DIR:{model.run_dir}", output)
        best, last = self.read_meta(model), self.read_meta(model, "last")
        self.assertEqual(best["optimizer"], "auto")
        self.assertEqual(best["lr0"], 0.01)
        self.assertEqual(best["task"], "detect")
        self.assertEqual(best["model_type"], "YOLO-detect")
        self.assertIn("validation_metrics", best)
        self.assertNotIn("validation_metrics", last)
        self.assertLessEqual(best["date_start"], best["date_end"])

    def test_training_oom_keeps_original_traceback_without_cleanup_retry(self):
        self.error_stage = "train"
        status, output = self.run_worker()
        self.assertEqual(status, 1)
        self.assertIn("original failure", output)
        self.assertIn("Traceback", output)
        self.assertIn("[CUDA_OOM]", output)
        self.assertIn("AutoBatch", output)
        self.assertNotIn("UnboundLocalError", output)
        self.assertNotIn("SUCCESS", output)
        self.assertEqual(set(self.models[-1].calls), {"train"})

    def test_validation_error_retains_checkpoints_and_does_not_export(self):
        self.error_stage = "val"
        status, output = self.run_worker(dict(self.params, export_onnx=True))
        self.assertEqual(status, 1)
        self.assertIn("валидация", output)
        self.assertIn("original failure", output)
        model = self.models[-1]
        self.assertTrue((model.run_dir / "weights" / "best.pt").is_file())
        self.assertNotIn("export", model.calls)

    def test_non_cuda_error_is_not_reported_as_oom(self):
        self.error_stage = "train"
        self.failure = ValueError("invalid dataset test")
        status, output = self.run_worker()
        self.assertEqual(status, 1)
        self.assertIn("invalid dataset test", output)
        self.assertNotIn("[CUDA_OOM]", output)

    def test_cancellation_sets_real_stop_flag_and_skips_postprocessing(self):
        self.stop = True
        status, output = self.run_worker(dict(self.params, export_onnx=True))
        self.assertEqual(status, worker.CANCELLED, output)
        model = self.models[-1]
        self.assertTrue(model.trainer.stop)
        self.assertFalse(hasattr(model.trainer, "stop_training"))
        self.assertNotIn("val", model.calls)
        self.assertNotIn("export", model.calls)
        self.assertNotIn("SUCCESS", output)
        self.assertEqual(self.read_meta(model)["status"], "stopped")
        self.assertTrue((model.run_dir / "weights" / "best_interrupted.pt").is_file())
        self.assertIsNone(worker.trainer_ref)

    def test_keyboard_interrupt_is_cancellation(self):
        self.error_stage = "train"
        self.failure = KeyboardInterrupt()
        status, output = self.run_worker()
        self.assertEqual(status, worker.CANCELLED)
        self.assertIn("CANCELLED", output)
        self.assertNotIn("SUCCESS", output)

    def test_bom_file_unicode_paths_and_explicit_amp_workers(self):
        status, output = self.run_worker(
            dict(self.params, data=str(self.data.parent), amp=False, workers=0),
            params_file=True,
        )
        self.assertEqual(status, 0, output)
        self.assertFalse(self.models[-1].calls["train"]["amp"])
        self.assertEqual(self.models[-1].calls["train"]["workers"], 0)
        self.assertEqual(self.models[-1].calls["train"]["data"], str(self.data))

    def test_export_failure_is_optional_and_original_traceback_is_available(self):
        self.error_stage = "export"
        status, output = self.run_worker(dict(self.params, export_onnx=True))
        self.assertEqual(status, 0, output)
        self.assertIn("[EXPORT][WARNING]", output)
        self.assertIn("export test failure", output)
        self.assertNotIn("UnboundLocalError", output)
        self.assertIn("SUCCESS", output)

    def test_export_uses_cpu_and_real_task(self):
        self.task = "obb"
        status, output = self.run_worker(dict(self.params, export_onnx=True))
        self.assertEqual(status, 0, output)
        model = self.models[-1]
        self.assertEqual(model.calls["export"]["device"], "cpu")
        self.assertEqual(model.calls["export"]["batch"], 1)
        path = model.run_dir / "weights" / "best_onnx_meta.json"
        self.assertEqual(json.loads(path.read_text())["task"], "obb")

    def test_invalid_parameters_fail_before_loading_model(self):
        for batch in (0, -2, True, 1.5, float("nan"), "auto"):
            with self.subTest(batch=batch):
                status, output = self.run_worker(dict(self.params, batch=batch))
                self.assertEqual(status, 1, output)
        status, output = self.run_worker([])
        self.assertEqual(status, 1, output)
        self.assertEqual(self.models, [])
        for model in ("yolo11m.pt", "yolov8n-seg.pt", "yolo26s-obb.pt"):
            self.assertTrue(worker.is_standard_yolo_model(model))
        for model in (
            "yolo11missing.pt",
            "C:\\models\\yolo11n.pt",
            "models/yolo11n.pt",
        ):
            self.assertFalse(worker.is_standard_yolo_model(model))

    def test_disable_augmentation_includes_mosaic_and_cutmix(self):
        args = worker.build_train_args(
            dict(self.params, disable_aug=True, mosaic=1, close_mosaic=10)
        )
        for name in ("mosaic", "cutmix", "mixup", "close_mosaic", "fliplr", "scale"):
            self.assertEqual(args[name], 0)
        self.assertIsNone(args["auto_augment"])

    def test_cli_exit_code_and_utf8_error_from_another_working_directory(self):
        stub = self.root / "ultralytics.py"
        stub.write_text(
            """__version__ = "cli-test"
class YOLO:
    task = "detect"
    def __init__(self, path): pass
    def add_callback(self, event, callback): pass
    def train(self, **kwargs):
        import sys
        assert "torch" not in sys.modules
        assert not any(name.startswith("PyQt5") for name in sys.modules)
        raise RuntimeError("CUDA out of memory: исходная ошибка")
""",
            encoding="utf-8",
        )
        path = self.root / "params.json"
        path.write_text(json.dumps(self.params), encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "utils_training/train_yolo_worker.py"),
                "--params_file",
                str(path),
            ],
            cwd=self.root,
            env=dict(os.environ, PYTHONPATH=str(self.root)),
            capture_output=True,
            timeout=15,
            check=False,
        )
        output = result.stdout.decode("utf-8")
        self.assertEqual(result.returncode, 1, output)
        self.assertIn("исходная ошибка", output)
        self.assertNotIn("UnboundLocalError", output)


@unittest.skipUnless(
    os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "opt-in GUI/runtime suite"
)
class YoloWorkerGuiTests(unittest.TestCase):
    def test_fragmented_utf8_protocol_finish_and_cancellation(self):
        code = r"""
import os,sys,tempfile
from pathlib import Path
from unittest.mock import patch
from PyQt5.QtCore import QProcess
from PyQt5.QtTest import QSignalSpy
from PyQt5.QtWidgets import QApplication,QMessageBox
from tabs.deep_learning_training import YoloTrainWindow
app=QApplication([])
window=YoloTrainWindow();window.show()
with tempfile.TemporaryDirectory() as directory:
    run=Path(directory)/'точный_запуск';(run/'weights').mkdir(parents=True)
    best=run/'weights'/'best.pt';best.write_bytes(b'test')
    for status in (0,130):
        window._yolo_output_buffer=''
        import codecs
        window._yolo_decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')
        window._yolo_stop_requested=False;window._yolo_run_dir=None
        window.log_text.clear()
        progress=[];window.yolo_train_widget.progress_bar.valueChanged.connect(progress.append)
        process=QProcess();window.train_process=process
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(window._on_yolo_output)
        process.finished.connect(window._on_yolo_finished)
        payload=('PROGRESS:50:1:2\nRESULT_DIR:'+str(run)+'\nготово').replace('\\n','\n').encode('utf-8')
        child='import os,time;data='+repr(payload)+'\nfor byte in data:\n os.write(1,bytes([byte]));time.sleep(.002)\nraise SystemExit('+str(status)+')'
        with patch.object(QMessageBox,'information') as info,patch.object(QMessageBox,'critical') as error,patch.object(window,'_detect_model_info') as inspect:
            finished=QSignalSpy(process.finished)
            process.start(sys.executable,['-c',child]);assert process.waitForStarted(3000)
            assert finished.wait(10000)
            assert 50 in progress,progress
            assert 'готово' in window.log_text.toPlainText(),window.log_text.toPlainText()
            assert window._yolo_run_dir==str(run)
            error.assert_not_called()
            if status==0:
                info.assert_called_once();inspect.assert_called_once_with(str(best))
            else:
                info.assert_not_called();inspect.assert_not_called()
                assert 'остановлено пользователем' in window.log_text.toPlainText()
        window.yolo_train_widget.progress_bar.valueChanged.disconnect(progress.append)
    # Windows terminate may kill the process without executing Python handlers.
    process=QProcess();window.train_process=process
    process.finished.connect(window._on_yolo_finished)
    window._yolo_decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')
    window._yolo_output_buffer=''
    with patch.object(QMessageBox,'critical') as error,patch.object(QMessageBox,'information') as info:
        process.start(sys.executable,['-c','import time;time.sleep(2)']);assert process.waitForStarted(3000)
        window.on_stop_training()
        error.assert_not_called();info.assert_not_called()
    invalid_data=Path(directory)/'invalid.yaml';invalid_data.write_text('names: [object]')
    with patch.object(QMessageBox,'critical') as error:
        window._run_yolo_training_from_widget(dict(data=str(invalid_data),model=str(Path(directory)/'missing-custom.pt')))
        process=window.train_process
        if process is not None:
            finished=QSignalSpy(process.finished);assert finished.wait(10000)
        assert window.train_process is None
        assert window.yolo_train_widget.run_btn.isEnabled()
        assert not window.yolo_train_widget.stop_btn.isEnabled()
        error.assert_called_once()
window.close();app.processEvents()
"""
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(
                os.environ,
                QT_QPA_PLATFORM="offscreen",
                XDG_CONFIG_HOME=directory,
                XDG_CACHE_HOME=directory,
                MPLCONFIGDIR=directory,
                YOLO_CONFIG_DIR=directory,
                NO_ALBUMENTATIONS_UPDATE="1",
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
