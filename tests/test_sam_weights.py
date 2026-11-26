"""Offline regression checks for checkpoint integrity, retry and authentication."""

from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import CancelledError
from http.client import IncompleteRead
import io
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
import unittest
import zipfile
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

from utils_new.import_export import sam_weights as downloader


_checkpoint = io.BytesIO()
with zipfile.ZipFile(_checkpoint, "w") as _archive:
    _archive.writestr("archive/data.pkl", b"pickle not executed by downloader")
    _archive.writestr("archive/data/0", b"tensor storage")
WEIGHTS = _checkpoint.getvalue()
TOKENIZER = gzip.compress(b"test vocabulary")
ROOT = Path(__file__).resolve().parents[1]


class Response(io.BytesIO):
    def __init__(self, payload, *, length=None):
        super().__init__(payload)
        self.status = 200
        self.headers = {"Content-Length": str(len(payload) if length is None else length)}


class Opener:
    def __init__(self, *results):
        self.results = iter(results)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result


class SAMDownloadTests(unittest.TestCase):
    def setUp(self):
        home = TemporaryDirectory()
        self.addCleanup(home.cleanup)
        credentials = patch.dict(os.environ, dict(HF_TOKEN="", HUGGING_FACE_HUB_TOKEN="",
            HF_HOME=home.name, HF_TOKEN_PATH=str(Path(home.name) / "token"),
            HF_HUB_DISABLE_IMPLICIT_TOKEN="0"))
        credentials.start()
        self.addCleanup(credentials.stop)

    def test_sam31_has_separate_revision_paths_and_is_not_an_image_architecture(self):
        self.assertNotIn("sam3.1", downloader.MODELS)
        model = downloader.DOWNLOAD_MODELS["sam3.1"]
        self.assertTrue(model.assets[0].path.endswith("sam3.1_multiplex.pt"))
        self.assertIn(downloader.SAM31_WEIGHTS_REVISION, model.assets[0].url)
        self.assertNotEqual(model.assets[0].path, downloader.MODELS["sam3"].assets[0].path)
        self.assertIn("sam3.1", downloader.select_models(["all"]))
        self.assertNotIn("grounding-dino-tiny", downloader.select_models(["all"]))

    def test_environment_and_active_hub_login_token_are_resolved_without_writing(self):
        path = Path(os.environ["HF_TOKEN_PATH"])
        path.write_text("hf_ACTIVE_LOGIN\n")
        self.assertEqual(downloader.resolve_hf_token(), "hf_ACTIVE_LOGIN")
        with patch.dict(os.environ, dict(HF_TOKEN="hf_ENV")):
            self.assertEqual(downloader.resolve_hf_token(), "hf_ENV")
        with patch.dict(os.environ, dict(HF_HUB_DISABLE_IMPLICIT_TOKEN="1")):
            self.assertIsNone(downloader.resolve_hf_token())
        self.assertEqual(path.read_text(), "hf_ACTIVE_LOGIN\n")

    def test_sam31_authenticates_weights_and_config_but_not_public_tokenizer(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as captured:
            opener = Opener(Response(WEIGHTS), Response(b'{"model_type":"sam3"}'), Response(TOKENIZER))
            plan = downloader.plan_model_download("sam3.1", Path(directory))
            paths = downloader.download_model(plan, opener=opener, token="hf_TEST_SECRET")
            for request in opener.requests:
                auth = request.get_header("Authorization")
                self.assertEqual(auth, "Bearer hf_TEST_SECRET" if "huggingface.co" in request.full_url else None)
            manifest = json.loads((Path(directory) / downloader.MANIFEST_NAME).read_text())
            self.assertNotIn("hf_TEST_SECRET", captured.getvalue() + json.dumps(manifest))
            self.assertEqual(len(paths), 3)

    def test_both_gated_models_prompt_once_and_keep_distinct_files(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            opener = Opener(Response(WEIGHTS), Response(TOKENIZER), Response(WEIGHTS),
                            Response(b'{"model_type":"sam3"}'), Response(TOKENIZER))
            with patch.object(downloader, "build_opener", return_value=opener), \
                    patch.object(downloader.getpass, "getpass", return_value="hf_INPUT") as ask:
                code = downloader.main(["--models", "sam3", "sam3.1", "--ask-token", "--output", directory])
            self.assertEqual(code, 0)
            ask.assert_called_once()
            self.assertTrue((Path(directory) / "sam3/sam3.pt").is_file())
            self.assertTrue((Path(directory) / "sam3.1/sam3.1_multiplex.pt").is_file())

    def test_sam31_existing_bundle_needs_no_token_or_network(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)
            plan = downloader.plan_model_download("sam3.1", output)
            for asset, path in plan.files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(WEIGHTS if asset.is_weights else TOKENIZER if asset.kind == "tokenizer" else b'{}')
            with patch.object(downloader, "resolve_hf_token") as token, \
                    patch.object(downloader.getpass, "getpass") as ask, \
                    patch.object(downloader, "build_opener") as opener:
                self.assertEqual(downloader.main(["--models", "sam3.1", "--ask-token", "--output", directory]), 0)
            token.assert_not_called()
            ask.assert_not_called()
            opener.return_value.open.assert_not_called()

    def test_sam31_gated_error_identifies_its_own_repo_without_secrets(self):
        error = HTTPError("https://cdn.example/SIGNED_SECRET", 403, "hf_PRIVATE", {}, None)
        message = downloader._http_error_message(error, downloader.DOWNLOAD_MODELS["sam3.1"].assets[1])
        self.assertIn("https://huggingface.co/facebook/sam3.1", message)
        self.assertNotIn("SIGNED_SECRET", message)
        self.assertNotIn("hf_PRIVATE", message)
        for url in ("http://huggingface.co/facebook/sam3.1/file", "https://huggingface.co.evil/facebook/sam3/file",
                    "https://huggingface.co/facebook/sam3.1-other/file"):
            self.assertIsNone(downloader.gated_repository(downloader.Asset("weights.pt", url)))

    def test_plan_is_local_and_reuses_uppercase_weights_without_copying(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            weights = root / "SAM3.pt"
            weights.write_bytes(WEIGHTS)
            output = root / "segment_anything"
            plan = downloader.plan_model_download("sam3", output)
            self.assertEqual(plan.files[0][1], weights)
            self.assertEqual([asset.kind for asset, _ in plan.missing], ["tokenizer"])
            self.assertFalse(output.exists())
            opener = Opener(Response(TOKENIZER))
            progress = []
            result = downloader.download_model(plan, opener=opener,
                                               token=lambda: self.fail("Local weights must not ask for a token"),
                                               progress=lambda *args: progress.append(args))
            self.assertEqual(len(opener.requests), 1)
            self.assertEqual(result["sam3/sam3.pt"], str(weights))
            self.assertEqual((root / "bpe_simple_vocab_16e6.txt.gz").read_bytes(), TOKENIZER)
            self.assertTrue(any(args[1] == "Скачивание" for args in progress))

    def test_download_plan_reuses_custom_weights_and_downloads_only_missing_config(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            weights = root / "external" / "custom.pt"
            weights.parent.mkdir()
            weights.write_bytes(WEIGHTS)
            plan = downloader.plan_model_download("sam2.1-small", root / "models", weights_file=weights)
            config = weights.parent / "configs/sam2.1/sam2.1_hiera_s.yaml"
            self.assertEqual(plan.files[1][1], config)
            opener = Opener(Response(b"model:\n  _target_: sam2.modeling.sam2_base.SAM2Base\n"))
            downloader.download_model(plan, opener=opener)
            self.assertTrue(config.is_file())
            self.assertEqual(weights.read_bytes(), WEIGHTS)
            self.assertEqual(len(opener.requests), 1)
            downloader.download_model(plan, opener=Opener())
            self.assertFalse(plan.missing)

    def test_plan_finds_existing_nested_tokenizer_next_to_separate_weights(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "external" / "SAM3.pt"
            weights.parent.mkdir()
            weights.write_bytes(WEIGHTS)
            bpe = weights.parent / "sam3" / "bpe_simple_vocab_16e6.txt.gz"
            bpe.parent.mkdir()
            bpe.write_bytes(TOKENIZER)
            plan = downloader.plan_model_download("sam3", root / "models" / "segment_anything",
                                                 weights_file=weights)
            self.assertEqual(plan.files[1][1], bpe)
            self.assertFalse(plan.missing)

    def test_cancel_during_read_does_not_publish_partial_file(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            cancel = Event()

            class CancellingResponse(Response):
                def read(self, size):
                    chunk = super().read(size)
                    cancel.set()
                    return chunk

            output = Path(directory)
            asset = downloader.MODELS["sam-vit-b"].assets[0]
            with self.assertRaises(CancelledError):
                downloader.download_asset(asset, output, {}, opener=Opener(CancellingResponse(WEIGHTS)),
                                          cancel=cancel)
            self.assertFalse((output / asset.path).exists())
            self.assertFalse(list(output.rglob("*.part")))
            with self.assertRaises(CancelledError):
                downloader.sha256_file(output / "nonexistent", cancel)

    def test_gui_bundle_sanitizes_auth_errors_and_retries_temporary_errors(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            plan = downloader.plan_model_download("sam3", Path(directory))
            failure = HTTPError("https://cdn.example/SIGNED_SECRET", 403, "hf_SECRET", {}, io.BytesIO(b"private"))
            with self.assertRaises(ValueError) as caught:
                downloader.download_model(plan, opener=Opener(failure), token="hf_SECRET")
            self.assertIn("https://huggingface.co/facebook/sam3", str(caught.exception))
            self.assertNotIn("SECRET", str(caught.exception))
            self.assertFalse(list(Path(directory).rglob("*.part")))
            plan = downloader.plan_model_download("sam-vit-b", Path(directory))
            failure = HTTPError("https://example.org/secret", 503, "Unavailable", {}, io.BytesIO())
            opener = Opener(failure, Response(WEIGHTS))
            with patch.object(downloader.time, "sleep"):
                downloader.download_model(plan, opener=opener)
            self.assertEqual(len(opener.requests), 2)

    def test_aliases_and_matching_sam2_assets(self):
        self.assertEqual(downloader.select_models(["sam", "sam-vit-b", "sam2", "sam3"]),
                         ["sam-vit-b", "sam2.1-tiny", "sam3"])
        self.assertEqual(len(downloader.select_models(["all"])), 13)
        for version in ("sam2", "sam2.1"):
            weights, config = downloader.MODELS[f"{version}-base-plus"].assets
            self.assertTrue(weights.path.endswith(f"{version}_hiera_base_plus.pt"))
            self.assertTrue(config.path.endswith(f"configs/{version}/{version}_hiera_b+.yaml"))
            self.assertIn(downloader.SAM2_REVISION, config.url)
        with self.assertRaises(ValueError):
            downloader.select_models(["unknown"])

    def test_complete_download_verified_skip_and_corruption(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)
            asset = downloader.MODELS["sam-vit-b"].assets[0]
            opener = Opener(Response(WEIGHTS))
            receipt = downloader.download_asset(asset, output, {}, opener=opener)
            self.assertEqual((output / asset.path).read_bytes(), WEIGHTS)
            manifest = {asset.path: receipt}
            self.assertEqual(downloader.download_asset(asset, output, manifest, opener=opener), receipt)
            self.assertEqual(len(opener.requests), 1)
            (output / asset.path).write_bytes(WEIGHTS[:-1] + b"!")
            with self.assertRaisesRegex(ValueError, "--force"):
                downloader.download_asset(asset, output, manifest, opener=opener)
            self.assertEqual(len(opener.requests), 1)

    def test_failed_force_preserves_existing_checkpoint_and_cleans_temporary(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)
            asset = downloader.MODELS["sam-vit-b"].assets[0]
            target = output / asset.path
            target.parent.mkdir(parents=True)
            target.write_bytes(WEIGHTS)
            for response in (Response(WEIGHTS, length=len(WEIGHTS) + 100),
                             Response(b"<!DOCTYPE html>access denied"), Response(b"")):
                with self.assertRaises(ValueError):
                    downloader.download_asset(asset, output, {}, opener=Opener(response), force=True)
                self.assertEqual(target.read_bytes(), WEIGHTS)
                self.assertFalse(list(output.rglob("*.part")))
            receipt = downloader.download_asset(asset, output, {}, opener=Opener())
            self.assertEqual(receipt["origin"], "existing-local")

    def test_wrong_auxiliary_files_are_not_published(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)
            for kind in ("config", "tokenizer"):
                asset = downloader.Asset(f"test/{kind}", "https://example.org/file", kind)
                with self.assertRaises(ValueError):
                    downloader.download_asset(asset, output, {}, opener=Opener(Response(b"<html>oops")))
                self.assertFalse((output / asset.path).exists())
                self.assertFalse(list(output.rglob("*.part")))

    def test_token_is_sent_only_to_huggingface_and_not_recorded(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)
            for model_key in ("sam-vit-b", "sam3"):
                asset = downloader.MODELS[model_key].assets[0]
                opener = Opener(Response(WEIGHTS))
                receipt = downloader.download_asset(asset, output, {}, opener=opener, token="hf_TEST_SECRET")
                auth = opener.requests[0].get_header("Authorization")
                self.assertEqual(auth, "Bearer hf_TEST_SECRET" if model_key == "sam3" else None)
                self.assertNotIn("hf_TEST_SECRET", json.dumps(receipt))

    def test_redirect_strips_authorization_and_blocks_http(self):
        handler = downloader.SafeRedirectHandler()
        original = Request("https://huggingface.co/file", headers={"Authorization": "Bearer secret"})
        redirected = handler.redirect_request(original, None, 302, "Found", {}, "https://cdn.example/file")
        self.assertIsNone(redirected.get_header("Authorization"))
        same_host = handler.redirect_request(original, None, 302, "Found", {}, "https://huggingface.co/other")
        self.assertEqual(same_host.get_header("Authorization"), "Bearer secret")
        with self.assertRaises(ValueError):
            handler.redirect_request(original, None, 302, "Found", {}, "http://cdn.example/file")

    def test_main_retries_temporary_http_failure_and_writes_manifest(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            failure = HTTPError("https://cdn.example/secret-url", 503, "Unavailable", {}, io.BytesIO())
            opener = Opener(failure, Response(WEIGHTS))
            with patch.object(downloader, "build_opener", return_value=opener), \
                    patch.object(downloader.time, "sleep"), patch.dict(os.environ, {"HF_TOKEN": ""}):
                result = downloader.main(["--models", "sam", "--output", directory])
            self.assertEqual(result, 0)
            self.assertEqual(len(opener.requests), 2)
            manifest = json.loads((Path(directory) / downloader.MANIFEST_NAME).read_text())
            self.assertIn("sam/sam_vit_b_01ec64.pth", manifest)

    def test_gated_failure_continues_other_models_without_leaking_response(self):
        with TemporaryDirectory() as directory:
            captured = io.StringIO()
            failure = HTTPError("https://cdn.example/SIGNED_SECRET", 403, "hf_SECRET", {},
                                io.BytesIO(b"private response hf_SECRET"))
            opener = Opener(failure, Response(TOKENIZER), Response(WEIGHTS))
            with redirect_stdout(captured), redirect_stderr(captured), \
                    patch.object(downloader, "build_opener", return_value=opener), \
                    patch.dict(os.environ, {"HF_TOKEN": "hf_SECRET"}):
                result = downloader.main(["--models", "sam3", "sam", "--output", directory])
            self.assertEqual(result, 1)
            self.assertEqual(len(opener.requests), 3)
            self.assertIn("--ask-token", captured.getvalue())
            self.assertNotIn("hf_SECRET", captured.getvalue())
            self.assertNotIn("SIGNED_SECRET", captured.getvalue())
            self.assertTrue((Path(directory) / "sam/sam_vit_b_01ec64.pth").is_file())
            self.assertTrue((Path(directory) / "sam3/bpe_simple_vocab_16e6.txt.gz").is_file())

    def test_interrupted_body_retries_and_keyboard_interrupt_cleans_up(self):
        class BrokenResponse(Response):
            def __init__(self, error):
                super().__init__(WEIGHTS)
                self.error = error

            def read(self, size):
                raise self.error

        for error, expected in ((IncompleteRead(b"PK", 100), 0), (KeyboardInterrupt(), 130)):
            with self.subTest(error=type(error).__name__), TemporaryDirectory() as directory, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                opener = Opener(BrokenResponse(error), Response(WEIGHTS))
                with patch.object(downloader, "build_opener", return_value=opener), \
                        patch.object(downloader.time, "sleep"), patch.dict(os.environ, {"HF_TOKEN": ""}):
                    result = downloader.main(["--models", "sam", "--output", directory])
                self.assertEqual(result, expected)
                self.assertFalse(list(Path(directory).rglob("*.part")))

    def test_dry_run_does_not_create_files_prompt_or_connect(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(downloader, "build_opener") as opener, \
                patch.object(downloader.getpass, "getpass") as getpass:
            output = Path(directory) / "not-created"
            result = downloader.main(["--dry-run", "--ask-token", "--output", str(output)])
            self.assertEqual(result, 0)
            self.assertFalse(output.exists())
            opener.assert_not_called()
            getpass.assert_not_called()

    def test_invalid_token_is_rejected_without_logging_it(self):
        with TemporaryDirectory() as directory:
            captured = io.StringIO()
            opener = Opener(Response(TOKENIZER))
            with patch.dict(os.environ, {"HF_TOKEN": "hf_SECRET\nBAD"}), \
                    redirect_stdout(captured), redirect_stderr(captured), \
                    patch.object(downloader, "build_opener", return_value=opener):
                result = downloader.main(["--models", "sam3", "--output", directory])
            self.assertEqual(result, 1)
            self.assertNotIn("hf_SECRET", captured.getvalue())
            self.assertEqual(len(opener.requests), 1)
            self.assertNotIn("huggingface.co", opener.requests[0].full_url)

    def test_cli_works_without_site_packages(self):
        result = subprocess.run([sys.executable, "-S", "-m", "utils_new.import_export.sam_weights", "--list"],
                                cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sam2.1-tiny", result.stdout)
        self.assertIn("sam3", result.stdout)

    def test_existing_uppercase_weights_without_manifest_need_only_tokenizer(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)
            weights = output / "sam3" / "SAM3.pt"
            weights.parent.mkdir()
            weights.write_bytes(WEIGHTS)
            original_mtime = weights.stat().st_mtime_ns
            opener = Opener(Response(TOKENIZER))
            with patch.object(downloader, "build_opener", return_value=opener), \
                    patch.object(downloader.getpass, "getpass") as getpass:
                result = downloader.main(["--models", "sam3", "--output", directory, "--ask-token"])
            self.assertEqual(result, 0)
            self.assertEqual(weights.stat().st_mtime_ns, original_mtime)
            self.assertEqual(weights.read_bytes(), WEIGHTS)
            self.assertEqual(len(opener.requests), 1)
            self.assertNotIn("huggingface.co", opener.requests[0].full_url)
            getpass.assert_not_called()
            manifest = json.loads((output / downloader.MANIFEST_NAME).read_text())
            self.assertEqual(manifest["sam3/sam3.pt"]["origin"], "existing-local")
            self.assertEqual(Path(manifest["sam3/sam3.pt"]["local_path"]), weights)

    def test_explicit_weights_default_to_auxiliary_files_beside_original(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            weights = Path(directory) / "my SAM3.pt"
            weights.write_bytes(WEIGHTS)
            opener = Opener(Response(TOKENIZER))
            args = ["--models", "sam3", "--weights-file", str(weights), "--ask-token"]
            with patch.object(downloader, "build_opener", return_value=opener), \
                    patch.object(downloader.getpass, "getpass") as getpass:
                self.assertEqual(downloader.main(args), 0)
                self.assertEqual(downloader.main(args), 0)  # Both assets are now verified locally.
            self.assertEqual(len(opener.requests), 1)
            self.assertEqual((weights.parent / "bpe_simple_vocab_16e6.txt.gz").read_bytes(), TOKENIZER)
            self.assertFalse((weights.parent / "sam3" / "sam3.pt").exists())
            getpass.assert_not_called()

    def test_extras_only_does_not_read_or_download_weights(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            # An invalid existing weights file must be irrelevant to this mode.
            weights = Path(directory) / "SAM3.pt"
            weights.write_text("not checked")
            opener = Opener(Response(TOKENIZER))
            with patch.object(downloader, "build_opener", return_value=opener), \
                    patch.object(downloader.getpass, "getpass") as getpass:
                result = downloader.main(["--models", "sam3", "--extras-only", "--output", directory, "--ask-token"])
            self.assertEqual(result, 0)
            self.assertEqual(weights.read_text(), "not checked")
            self.assertEqual(len(opener.requests), 1)
            getpass.assert_not_called()

    def test_invalid_existing_weights_are_preserved_but_auxiliary_download_continues(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            weights = Path(directory) / "SAM3.pt"
            weights.write_bytes(b"PK\x03\x04incomplete archive")
            opener = Opener(Response(TOKENIZER))
            with patch.object(downloader, "build_opener", return_value=opener):
                result = downloader.main(["--models", "sam3", "--weights-file", str(weights)])
            self.assertEqual(result, 1)
            self.assertEqual(weights.read_bytes(), b"PK\x03\x04incomplete archive")
            self.assertTrue((weights.parent / "bpe_simple_vocab_16e6.txt.gz").is_file())
            self.assertEqual(len(opener.requests), 1)

    def test_explicit_weights_can_use_separate_output_without_copying_checkpoint(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            weights = Path(directory) / "SAM3.pt"
            weights.write_bytes(WEIGHTS)
            output = Path(directory) / "resources"
            opener = Opener(Response(TOKENIZER))
            with patch.object(downloader, "build_opener", return_value=opener):
                result = downloader.main(["--models", "sam3", "--weights-file", str(weights), "--output", str(output)])
            self.assertEqual(result, 0)
            self.assertFalse(list(output.rglob("*.pt")))
            self.assertTrue((output / "bpe_simple_vocab_16e6.txt.gz").is_file())

    def test_truncated_gzip_is_not_adopted_or_published(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            asset = downloader.MODELS["sam3"].assets[1]
            output = Path(directory)
            with self.assertRaises((ValueError, OSError, EOFError)):
                downloader.download_asset(asset, output, {}, opener=Opener(Response(TOKENIZER[:-5])))
            self.assertFalse((output / asset.path).exists())
            self.assertFalse(list(output.rglob("*.part")))

    def test_grounding_model_bundle_is_public_pinned_and_separate_from_sam_ui_catalog(self):
        self.assertNotIn("grounding-dino-tiny", downloader.MODELS)
        self.assertEqual(downloader.select_models(["grounding-dino-tiny"]), ["grounding-dino-tiny"])
        assets = downloader.DOWNLOAD_MODELS["grounding-dino-tiny"].assets
        self.assertTrue(assets[0].is_weights)
        self.assertEqual(assets[0].kind, "safetensors")
        self.assertTrue(any(asset.path.endswith("tokenizer.json") for asset in assets))
        self.assertTrue(all("/resolve/a2bb814dd30d776dcf7e30523b00659f4f141c71/" in asset.url for asset in assets))

    def test_grounding_extras_only_does_not_download_weights_or_request_hf_token(self):
        assets = downloader.GROUNDING_MODELS["grounding-dino-tiny"].assets
        responses = [Response(b"{}" if asset.kind == "json" else b"vocabulary")
                     for asset in assets if not asset.is_weights]
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            opener = Opener(*responses)
            with patch.object(downloader, "build_opener", return_value=opener), \
                    patch.object(downloader.getpass, "getpass") as getpass:
                result = downloader.main(["--models", "grounding-dino-tiny", "--extras-only", "--ask-token", "--output", directory])
            self.assertEqual(result, 0)
            getpass.assert_not_called()
            self.assertTrue(all(not request.has_header("Authorization") for request in opener.requests))
            self.assertFalse(list(Path(directory).rglob("*.safetensors")))
            self.assertTrue((Path(directory) / "grounding_dino/tiny/config.json").is_file())

    def test_safetensors_container_rejects_truncation_and_accepts_existing_weight(self):
        header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        payload = len(header).to_bytes(8, "little") + header + b"\x00" * 4
        asset = downloader.GROUNDING_MODELS["grounding-dino-tiny"].assets[0]
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)
            receipt = downloader.download_asset(asset, output, {}, opener=Opener(Response(payload)), token=lambda: self.fail("Public model requested a token"))
            self.assertEqual(receipt["size"], len(payload))
            (output / asset.path).write_bytes(payload[:-1])
            with self.assertRaises(ValueError):
                downloader.download_asset(asset, output, {}, opener=Opener())


if __name__ == "__main__":
    unittest.main()
