"""Download official SAM checkpoints without importing any ML libraries.

Run ``python -m utils_new.import_export.sam_weights --help`` from the project root.
"""

import argparse
from concurrent.futures import CancelledError
from dataclasses import dataclass, replace
import getpass
import gzip
import hashlib
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import sys
import tempfile
from threading import Lock
import time
import warnings
import zipfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


SAM2_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM3_REVISION = "31cb7e624e4fe95301b1be1d4e809111593d2e45"
SAM3_WEIGHTS_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
SAM31_WEIGHTS_REVISION = "daa63191845a41281374e725f4c9e51c7a824460"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "models" / "segment_anything"
MANIFEST_NAME = "download_manifest.json"
CHUNK_SIZE = 1024 * 1024
DEFAULT_MODELS = ("sam", "sam2", "sam3", "grounding-dino-tiny")


class SkipGatedFiles(Exception):
    """No credential was supplied; continue public assets and the remaining models."""


@dataclass(frozen=True)
class Asset:
    path: str
    url: str
    kind: str = "weights"

    @property
    def is_weights(self):
        return self.kind in ("weights", "safetensors")


@dataclass(frozen=True)
class Model:
    name: str
    assets: tuple[Asset, ...]


def _catalog():
    models = {}
    for variant, suffix in (("b", "01ec64"), ("l", "0b3195"), ("h", "4b8939")):
        filename = f"sam_vit_{variant}_{suffix}.pth"
        models[f"sam-vit-{variant}"] = Model(f"SAM ViT-{variant.upper()}", (
            Asset(f"sam/{filename}", f"https://dl.fbaipublicfiles.com/segment_anything/{filename}"),
        ))
    for version, release in (("sam2", "072824"), ("sam2.1", "092824")):
        for variant, config_suffix in (("tiny", "t"), ("small", "s"),
                                       ("base-plus", "b+"), ("large", "l")):
            filename = f"{version}_hiera_{variant.replace('-', '_')}.pt"
            config = f"configs/{version}/{version}_hiera_{config_suffix}.yaml"
            models[f"{version}-{variant}"] = Model(f"{version.upper()} Hiera {variant}", (
                Asset(f"sam2/{filename}",
                      f"https://dl.fbaipublicfiles.com/segment_anything_2/{release}/{filename}"),
                Asset(f"sam2/{config}",
                      f"https://raw.githubusercontent.com/facebookresearch/sam2/{SAM2_REVISION}/sam2/{config}",
                      "config"),
            ))
    models["sam3"] = Model("SAM3", (
        Asset("sam3/sam3.pt",
              f"https://huggingface.co/facebook/sam3/resolve/{SAM3_WEIGHTS_REVISION}/sam3.pt"),
        Asset("sam3/bpe_simple_vocab_16e6.txt.gz",
              f"https://raw.githubusercontent.com/facebookresearch/sam3/{SAM3_REVISION}"
              "/sam3/assets/bpe_simple_vocab_16e6.txt.gz", "tokenizer"),
    ))
    return models


MODELS = _catalog()

# Multiplex needs the official video backend, not the image backend used by our tabs.
# Keep downloadable checkpoints separate from architectures supported for inference.
DOWNLOAD_ONLY_MODELS = {
    "sam3.1": Model("SAM3.1 Object Multiplex (видеотрекинг; только скачивание)", (
        Asset("sam3.1/sam3.1_multiplex.pt",
              f"https://huggingface.co/facebook/sam3.1/resolve/{SAM31_WEIGHTS_REVISION}/sam3.1_multiplex.pt"),
        Asset("sam3.1/config.json",
              f"https://huggingface.co/facebook/sam3.1/resolve/{SAM31_WEIGHTS_REVISION}/config.json", "json"),
        Asset("sam3.1/bpe_simple_vocab_16e6.txt.gz",
              f"https://raw.githubusercontent.com/facebookresearch/sam3/{SAM3_REVISION}"
              "/sam3/assets/bpe_simple_vocab_16e6.txt.gz", "tokenizer"),
    )),
}
SAM_DOWNLOAD_MODELS = {**MODELS, **DOWNLOAD_ONLY_MODELS}


def gated_repository(asset):
    """Only the two official gated repos may receive the user's bearer token."""
    parsed = urlsplit(asset.url)
    parts = parsed.path.strip("/").split("/")
    if (parsed.scheme == "https" and parsed.netloc == "huggingface.co" and len(parts) > 2
            and parts[:2] in (["facebook", "sam3"], ["facebook", "sam3.1"])):
        return "https://huggingface.co/" + "/".join(parts[:2])
    return None


def resolve_hf_token():
    """Read the active HF credential without importing hub/model libraries or logging it."""
    token = os.environ.get("HF_TOKEN", "").strip() or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    if token:
        return token
    if os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN", "").upper() in ("1", "TRUE", "YES", "ON"):
        return None
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    home = Path(os.environ.get("HF_HOME") or cache / "huggingface").expanduser()
    path = Path(os.environ.get("HF_TOKEN_PATH") or home / "token").expanduser()
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except (OSError, UnicodeError):
        return None


def _grounding_catalog():
    models = {}
    for variant, revision in (("tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71"),
                              ("base", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb")):
        repo = f"IDEA-Research/grounding-dino-{variant}"
        files = [("model.safetensors", "safetensors"), ("config.json", "json"),
                 ("preprocessor_config.json", "json"), ("tokenizer_config.json", "json"),
                 ("tokenizer.json", "json"), ("special_tokens_map.json", "json"),
                 ("vocab.txt", "text"), ("README.md", "text")]
        if variant == "tiny":
            files.append(("added_tokens.json", "json"))
        models[f"grounding-dino-{variant}"] = Model(f"Grounding DINO {variant} (Transformers)", tuple(
            Asset(f"grounding_dino/{variant}/{name}", f"https://huggingface.co/{repo}/resolve/{revision}/{name}", kind)
            for name, kind in files))
    return models


GROUNDING_MODELS = _grounding_catalog()
DOWNLOAD_MODELS = {**SAM_DOWNLOAD_MODELS, **GROUNDING_MODELS}
ALIASES = {"sam": "sam-vit-b", "sam2": "sam2.1-tiny"}


class SafeRedirectHandler(HTTPRedirectHandler):
    """Do not forward a Hugging Face bearer token to a download CDN."""

    def redirect_request(self, request, fp, code, message, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise ValueError("Отклонено перенаправление на незащищённый адрес")
        redirected = super().redirect_request(request, fp, code, message, headers, newurl)
        if redirected is not None and urlsplit(request.full_url).netloc != urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def select_models(names):
    selected = []
    for name in names:
        keys = SAM_DOWNLOAD_MODELS if name == "all" else (ALIASES.get(name, name),)
        for key in keys:
            if key not in DOWNLOAD_MODELS:
                raise ValueError(f"Неизвестная модель: {key}. Список: --list")
            if key not in selected:
                selected.append(key)
    return selected


def check_cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise CancelledError()


def sha256_file(path, cancel=None):
    check_cancelled(cancel)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            check_cancelled(cancel)
            digest.update(chunk)
    check_cancelled(cancel)
    return digest.hexdigest()


def write_manifest(path, manifest):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".sam-manifest-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_asset_file(path, asset, cancel=None):
    """Check containers without loading Torch or executing checkpoint pickle."""
    check_cancelled(cancel)
    with path.open("rb") as stream:
        prefix = stream.read(1024)
    if asset.kind == "safetensors":
        header_size = int.from_bytes(prefix[:8], "little")
        size = path.stat().st_size
        if not 2 <= header_size <= min(100 * 1024 * 1024, size - 8):
            raise ValueError("Некорректный заголовок safetensors")
        with path.open("rb") as stream:
            stream.read(8)
            header = json.loads(stream.read(header_size))
        if not isinstance(header, dict):
            raise ValueError("Некорректный заголовок safetensors")
        tensors = [value for key, value in header.items() if key != "__metadata__"]
        ends = []
        for tensor in tensors:
            offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or not all(type(value) is int for value in offsets)
                    or not 0 <= offsets[0] <= offsets[1] <= size - 8 - header_size):
                raise ValueError("Некорректный или неполный safetensors checkpoint")
            ends.append(offsets[1])
        if not ends or max(ends) != size - 8 - header_size:
            raise ValueError("Некорректный размер safetensors checkpoint")
    elif asset.kind == "json":
        with path.open(encoding="utf-8") as stream:
            if not isinstance(json.load(stream), dict):
                raise ValueError("Ожидается JSON-объект конфигурации модели")
    elif asset.kind == "text":
        if not path.read_text(encoding="utf-8").strip():
            raise ValueError("Получен пустой текстовый ресурс модели")
    elif asset.kind == "weights":
        if prefix.startswith(b"PK\x03\x04"):
            try:
                with zipfile.ZipFile(path) as archive:
                    names = archive.namelist()
                    if (not any(name == "data.pkl" or name.endswith("/data.pkl") for name in names)
                            or not any(("/data/" in item.filename or item.filename.startswith("data/"))
                                       and item.file_size > 0 for item in archive.infolist())):
                        raise ValueError("ZIP не содержит структуру checkpoint PyTorch")
            except zipfile.BadZipFile as error:
                raise ValueError("Неполный или повреждённый ZIP весов PyTorch") from error
        elif not (prefix.startswith(b"\x80") and path.stat().st_size >= 1024):
            raise ValueError("Вместо весов получен неизвестный формат (возможно, HTML или Git LFS pointer)")
    elif asset.kind == "tokenizer":
        if not prefix.startswith(b"\x1f\x8b"):
            raise ValueError("Файл токенизатора не является GZIP")
        with gzip.open(path, "rb") as stream:
            while stream.read(CHUNK_SIZE):
                check_cancelled(cancel)  # Verify the GZIP footer/CRC too.
    elif asset.kind == "config" and not (b"_target_:" in prefix and b"sam2." in prefix):
        raise ValueError("Получен неизвестный формат конфигурации SAM2")


def existing_asset_path(asset, output, receipt, explicit=None):
    if explicit is not None:
        if not explicit.is_file():
            raise ValueError(f"Не найден локальный файл весов: {explicit}")
        return explicit
    canonical = output / asset.path
    if canonical.exists():
        return canonical
    if isinstance(receipt, dict) and receipt.get("url") == asset.url and receipt.get("local_path"):
        recorded = Path(receipt["local_path"])
        if recorded.is_file():
            return recorded
    # Check only the expected directory and its supplied root, never scan a dataset/venv.
    directories = (canonical.parent, output) if asset.kind in ("weights", "tokenizer") else (canonical.parent,)
    for directory in dict.fromkeys(directories):
        if directory.is_dir():
            matches = [path for path in directory.iterdir()
                       if path.is_file() and path.name.casefold() == canonical.name.casefold()]
            if len(matches) > 1:
                raise ValueError(f"Несколько файлов с именем {canonical.name} в {directory}; укажите --weights-file")
            if matches:
                return matches[0]
    return canonical


def download_asset(asset, output, manifest, *, opener, token=None, force=False, timeout=30,
                   existing_path=None, destination=None, cancel=None, progress=None):
    """Publish only a complete download; return its size and SHA-256 receipt."""
    check_cancelled(cancel)
    receipt = manifest.get(asset.path)
    target = Path(destination) if destination is not None else (
        output / asset.path if force else existing_asset_path(asset, output, receipt, existing_path))
    if target.exists() and not force:
        if progress:
            progress(asset.path, "Проверка локального файла", 0, None)
        print(f"Проверка: {target}", flush=True)
        if not target.is_file():
            raise ValueError(f"Ожидается файл: {target}")
        if isinstance(receipt, dict) and receipt.get("url") == asset.url:
            if (target.stat().st_size != receipt.get("size") or sha256_file(target, cancel) != receipt.get("sha256")):
                raise ValueError(f"{target}: файл повреждён или изменён. Для скачивания укажите --force.")
            print("  Уже скачан, SHA-256 совпадает.", flush=True)
            return dict(receipt, local_path=str(target.resolve()))
        validate_asset_file(target, asset, cancel)
        print("  Используется готовый локальный файл; повторное скачивание не требуется.", flush=True)
        return {"url": asset.url, "size": target.stat().st_size, "sha256": sha256_file(target, cancel),
                "local_path": str(target.resolve()), "origin": "existing-local"}

    target.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Neuro-threshold-SAM-downloader/1", "Accept-Encoding": "identity"}
    if gated_repository(asset):
        credential = token() if callable(token) else token
        if credential and any(character.isspace() or not character.isascii() for character in credential):
            raise ValueError("Некорректный HF_TOKEN: ожидается токен без пробелов и переносов строк")
        if credential:
            headers["Authorization"] = f"Bearer {credential}"
    temporary = None
    print(f"Скачивание: {asset.path}", flush=True)
    try:
        check_cancelled(cancel)
        with opener.open(Request(asset.url, headers=headers), timeout=timeout) as response:
            if response.status != 200:
                raise ValueError(f"Ожидался полный файл (HTTP 200), получен HTTP {response.status}")
            length = response.headers.get("Content-Length")
            total = int(length) if length is not None else None
            digest = hashlib.sha256()
            size = 0
            last_update = time.monotonic()
            if progress:
                progress(asset.path, "Скачивание", 0, total)
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}-",
                                             suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                while True:
                    check_cancelled(cancel)
                    chunk = response.read(CHUNK_SIZE)
                    check_cancelled(cancel)
                    if not chunk:
                        break
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    now = time.monotonic()
                    if now - last_update >= 1:
                        if progress:
                            progress(asset.path, "Скачивание", size, total)
                        percent = f" / {total / CHUNK_SIZE:.1f} MiB ({100 * size / total:.1f}%)" if total else ""
                        print(f"  {size / CHUNK_SIZE:.1f} MiB{percent}", flush=True)
                        last_update = now
            if size == 0 or (total is not None and size != total):
                raise ValueError(f"Неполный файл: получено {size} байт, ожидалось {total}")
            if progress:
                progress(asset.path, "Проверка скачанного файла", size, total)
            validate_asset_file(temporary, asset, cancel)
            check_cancelled(cancel)
            os.replace(temporary, target)
        print(f"  Сохранено: {target} ({size / CHUNK_SIZE:.1f} MiB)", flush=True)
        return {"url": asset.url, "size": size, "sha256": digest.hexdigest(),
                "local_path": str(target.resolve()), "origin": "download"}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _http_error_message(error, asset):
    # Never print a signed CDN URL, headers, bearer token or response body.
    repository = gated_repository(asset)
    if repository and error.code in (401, 403):
        return (f"HTTP {error.code}: требуется доступ к {repository}. "
                "Примите условия, дождитесь доступа и задайте HF_TOKEN с правом чтения "
                "этого репозитория либо используйте --ask-token.")
    return f"HTTP {error.code}; источник: {urlsplit(asset.url).hostname}"


@dataclass(frozen=True)
class ModelDownloadPlan:
    """Resolved destinations; creating a plan performs no network or weight reads."""

    model_key: str
    output: Path
    files: tuple

    @property
    def missing(self):
        return tuple((asset, path) for asset, path in self.files if not path.is_file())


def read_manifest(output):
    path = output / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(manifest, dict):
        raise ValueError(f"Некорректный манифест: {path}")
    return manifest


def plan_model_download(model_key, output=DEFAULT_OUTPUT, *, weights_file=None, tokenizer_file=None):
    """Reuse known local files, including separately downloaded/capitalized weights.

    Auxiliary files follow the selected weights; no recursive directory scans or
    hashing take place here, so the GUI can display this plan immediately.
    """
    output = Path(output).expanduser().resolve()
    model = DOWNLOAD_MODELS[model_key]
    manifest = read_manifest(output)
    weights_asset = model.assets[0]
    weights = existing_asset_path(weights_asset, output, manifest.get(weights_asset.path),
                                  Path(weights_file).expanduser().resolve() if weights_file else None)
    if not weights.is_file() and weights_file is None:
        candidate = existing_asset_path(weights_asset, output.parent, {})
        if candidate.is_file():
            weights = candidate
    files = [(weights_asset, weights)]
    for asset in model.assets[1:]:
        relative = Path(asset.path).relative_to(Path(weights_asset.path).parent).as_posix()
        adjacent = existing_asset_path(replace(asset, path=relative), weights.parent, {})
        known = existing_asset_path(asset, output, manifest.get(asset.path))
        if asset.kind == "tokenizer" and tokenizer_file:
            path = Path(tokenizer_file).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"Не найден токенизатор: {path}")
        else:
            candidates = (adjacent, existing_asset_path(asset, weights.parent, {}), known,
                          existing_asset_path(asset, output.parent, {}))
            path = next((candidate for candidate in candidates if candidate.is_file()), adjacent)
        files.append((asset, path))
    return ModelDownloadPlan(model_key, output, tuple(files))


# The annotation and deep-learning tabs may request the same model concurrently.
# Serialize bundles in this process so the second request reuses the first one's
# files and manifest instead of starting another multi-gigabyte transfer.
_model_download_lock = Lock()


def download_model(plan, *, cancel=None, progress=None, token=None, opener=None, retries=2):
    """Download/verify a GUI plan using the same atomic downloader as the CLI."""
    while not _model_download_lock.acquire(timeout=0.1):
        check_cancelled(cancel)
    try:
        check_cancelled(cancel)
        plan.output.mkdir(parents=True, exist_ok=True)
        manifest = read_manifest(plan.output)
        opener = opener or build_opener(SafeRedirectHandler())
        result = {}
        for asset, destination in plan.files:
            for attempt in range(retries + 1):
                try:
                    receipt = download_asset(asset, plan.output, manifest, opener=opener,
                                             destination=destination, token=token,
                                             cancel=cancel, progress=progress)
                except HTTPError as error:
                    message = _http_error_message(error, asset).replace(
                        "либо используйте --ask-token", "либо повторите скачивание и введите токен в окне")
                    retry = error.code in (408, 429, 500, 502, 503, 504)
                    error.close()
                except (URLError, TimeoutError, ConnectionError, IncompleteRead):
                    message = "Сетевой сбой или тайм-аут; проверьте подключение и повторите скачивание."
                    retry = True
                else:
                    manifest[asset.path] = receipt
                    write_manifest(plan.output / MANIFEST_NAME, manifest)
                    result[asset.path] = receipt["local_path"]
                    break
                check_cancelled(cancel)
                if not retry or attempt == retries:
                    raise ValueError(message) from None
                if progress:
                    progress(asset.path, f"Повтор подключения {attempt + 1}/{retries}", 0, None)
                if cancel is not None:
                    cancel.wait(min(attempt + 1, 5))
                else:
                    time.sleep(min(attempt + 1, 5))
                check_cancelled(cancel)
        return result
    finally:
        _model_download_lock.release()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Скачать веса SAM/SAM2/SAM3/SAM3.1, Grounding DINO и файлы для локальной загрузки. "
                    "Без установки PyTorch и моделей. По умолчанию: SAM ViT-B, SAM2.1 Tiny, SAM3 и Grounding DINO tiny.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                        metavar="MODEL", help="Имена из --list; sam, sam2 и all — сокращения")
    parser.add_argument("--output", type=Path,
                        help="Папка сохранения; по умолчанию models/segment_anything или рядом с --weights-file")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--weights-file", type=Path, help="Готовые локальные веса одной выбранной модели; файл не копируется")
    reuse.add_argument("--extras-only", action="store_true", help="Только конфигурации/токенизатор, без проверки и скачивания весов")
    parser.add_argument("--list", action="store_true", help="Показать модели без скачивания")
    parser.add_argument("--dry-run", action="store_true", help="Показать файлы и URL без сети")
    parser.add_argument("--force", action="store_true", help="Повторно скачать и заменить выбранные файлы")
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument("--ask-token", action="store_true",
                             help="Принудительно запросить токен; без флага консоль спросит автоматически, если токен не найден")
    credentials.add_argument("--non-interactive", "--no-prompt", action="store_true",
                             help="Не спрашивать токен; без него пропустить закрытые файлы и продолжить другие модели")
    parser.add_argument("--timeout", type=float, default=30, help="Тайм-аут сетевой операции, секунды")
    parser.add_argument("--retries", type=int, default=2, help="Повторы после временного сетевого сбоя (0–5)")
    args = parser.parse_args(argv)
    if args.list:
        for key, model in DOWNLOAD_MODELS.items():
            print(f"{key:20} {model.name}")
        print("Сокращения: sam = sam-vit-b; sam2 = sam2.1-tiny; all = все варианты SAM без DINO.")
        print("По умолчанию: sam, sam2, sam3, grounding-dino-tiny. --models заменяет этот список.")
        return 0
    if not 0 <= args.retries <= 5 or not 0 < args.timeout < float("inf"):
        parser.error("--retries: от 0 до 5; --timeout: положительное конечное число")
    try:
        selected = select_models(args.models)
        if args.weights_file and (len(selected) != 1 or args.force):
            parser.error("--weights-file требует одну модель через --models и не сочетается с --force")
        weights_file = args.weights_file.expanduser().resolve(strict=True) if args.weights_file else None
        if weights_file is not None and not weights_file.is_file():
            raise ValueError(f"Ожидается файл весов: {weights_file}")
        output = (args.output or (weights_file.parent if weights_file else DEFAULT_OUTPUT)).expanduser().resolve()

        def assets_for(key):
            assets = DOWNLOAD_MODELS[key].assets
            if args.extras_only:
                return tuple(asset for asset in assets if not asset.is_weights)
            if weights_file:
                # Keep auxiliary files beside the explicitly selected weights by default.
                return tuple(asset if asset.is_weights else replace(
                    asset, path=Path(asset.path).relative_to(Path(assets[0].path).parent).as_posix()) for asset in assets)
            return assets

        print(f"Папка: {output}")
        print(f"Выбранные модели: {', '.join(selected)}", flush=True)
        if args.dry_run:
            for key in selected:
                for asset in assets_for(key):
                    if weights_file and asset.is_weights:
                        print(f"Готовые веса: {weights_file}")
                    else:
                        print(f"{asset.path}\n  {asset.url}")
            return 0
        manifest_path = output / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if not isinstance(manifest, dict):
            raise ValueError(f"Некорректный манифест: {manifest_path}")
        token = None
        token_resolved = False
        token_asked = False
        interactive = not args.non_interactive and sys.stdin is not None and sys.stdin.isatty()

        def auth_token(repository):
            nonlocal token, token_asked, token_resolved
            if not token_resolved:
                token = resolve_hf_token()
                token_resolved = True
            if (args.ask_token or (interactive and not token)) and not token_asked:
                print(f"Доступ к {repository}: токен с правом чтения; условия репозитория должны быть приняты.", flush=True)
                token_asked = True
                try:
                    with warnings.catch_warnings():
                        # Never fall back to an input mode that echoes credentials.
                        warnings.simplefilter("error", getpass.GetPassWarning)
                        entered = getpass.getpass("Hugging Face token (ввод скрыт; Enter — пропустить закрытые файлы SAM): ")
                    token = entered.strip() or None
                except (EOFError, OSError, getpass.GetPassWarning):
                    token = None
            if not token:
                raise SkipGatedFiles(
                    f"нет токена для {repository}; следующие модели будут обработаны. "
                    "Задайте HF_TOKEN или повторите запуск с --ask-token.")
            return token
        output.mkdir(parents=True, exist_ok=True)
        opener = build_opener(SafeRedirectHandler())
        failed = []
        skipped = []
        for key in selected:
            print(f"\n{DOWNLOAD_MODELS[key].name}", flush=True)
            for asset in assets_for(key):
                for attempt in range(args.retries + 1):
                    try:
                        receipt = download_asset(asset, output, manifest, opener=opener,
                                                 token=lambda: auth_token(gated_repository(asset)),
                                                 force=args.force, timeout=args.timeout,
                                                 existing_path=weights_file if asset.is_weights else None)
                    except SkipGatedFiles as error:
                        print(f"  Пропуск {asset.path}: {error}", file=sys.stderr)
                        if key not in skipped:
                            skipped.append(key)
                        break
                    except HTTPError as error:
                        message = _http_error_message(error, asset)
                        retry = error.code in (408, 429, 500, 502, 503, 504)
                        error.close()
                    except (URLError, TimeoutError, ConnectionError, IncompleteRead):
                        message = "Сетевой сбой или тайм-аут; проверьте подключение и повторите запуск."
                        retry = True
                    except (OSError, ValueError, EOFError) as error:
                        message, retry = str(error), False
                    else:
                        manifest[asset.path] = receipt
                        write_manifest(manifest_path, manifest)
                        break
                    if retry and attempt < args.retries:
                        print(f"  {message} Повтор {attempt + 1}/{args.retries}.", file=sys.stderr)
                        time.sleep(min(attempt + 1, 2))
                    else:
                        print(f"  Ошибка: {message}", file=sys.stderr)
                        if key not in failed:
                            failed.append(key)
                        break
        if failed:
            print(f"\nНе скачаны полностью: {', '.join(failed)}.")
        elif skipped:
            print("\nОбработка завершена; доступные файлы скачаны или проверены.")
        else:
            print("\nСкачивание завершено.")
        if skipped:
            print(f"Пропущены закрытые файлы моделей: {', '.join(skipped)}. Остальные выбранные модели обработаны.")
        print("Веса не устанавливают backend: для применения нужны соответствующие пакеты моделей.")
        if "sam3.1" in selected:
            print("SAM3.1 Multiplex скачивается отдельно: видеобэкенд ещё не подключён к вкладкам применения.")
        return 1 if failed or skipped else 0
    except KeyboardInterrupt:
        print("\nСкачивание остановлено. Готовые файлы сохранены; незавершённый файл удалён.", file=sys.stderr)
        return 130
    except (OSError, ValueError, EOFError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
