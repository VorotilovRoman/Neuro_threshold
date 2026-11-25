"""Offline regression checks for checkpoint integrity, retry and authentication."""

from contextlib import redirect_stdout, redirect_stderr
from http.client import IncompleteRead
import io
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
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
    def test_aliases_and_matching_sam2_assets(self):
        self.assertEqual(downloader.select_models(["sam", "sam-vit-b", "sam2", "sam3"]),
                         ["sam-vit-b", "sam2.1-tiny", "sam3"])
        self.assertEqual(len(downloader.select_models(["all"])), 12)
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


if __name__ == "__main__":
    unittest.main()
