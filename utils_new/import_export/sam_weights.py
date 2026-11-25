"""Download official SAM checkpoints without importing any ML libraries.

Run ``python -m utils_new.import_export.sam_weights --help`` from the project root.
"""

import argparse
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
import time
import zipfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


SAM2_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM3_REVISION = "31cb7e624e4fe95301b1be1d4e809111593d2e45"
SAM3_WEIGHTS_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "models" / "segment_anything"
MANIFEST_NAME = "download_manifest.json"
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class Asset:
    path: str
    url: str
    kind: str = "weights"


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
        keys = MODELS if name == "all" else (ALIASES.get(name, name),)
        for key in keys:
            if key not in MODELS:
                raise ValueError(f"Неизвестная модель: {key}. Список: --list")
            if key not in selected:
                selected.append(key)
    return selected


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
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


def validate_asset_file(path, asset):
    """Check containers without loading Torch or executing checkpoint pickle."""
    with path.open("rb") as stream:
        prefix = stream.read(1024)
    if asset.kind == "weights":
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
                pass  # Verify the GZIP footer/CRC too, without retaining the vocabulary.
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
    for directory in dict.fromkeys((canonical.parent, output)):
        if directory.is_dir():
            matches = [path for path in directory.iterdir()
                       if path.is_file() and path.name.casefold() == canonical.name.casefold()]
            if len(matches) > 1:
                raise ValueError(f"Несколько файлов с именем {canonical.name} в {directory}; укажите --weights-file")
            if matches:
                return matches[0]
    return canonical


def download_asset(asset, output, manifest, *, opener, token=None, force=False, timeout=30, existing_path=None):
    """Publish only a complete download; return its size and SHA-256 receipt."""
    receipt = manifest.get(asset.path)
    target = output / asset.path if force else existing_asset_path(asset, output, receipt, existing_path)
    if target.exists() and not force:
        print(f"Проверка: {target}", flush=True)
        if not target.is_file():
            raise ValueError(f"Ожидается файл: {target}")
        if isinstance(receipt, dict) and receipt.get("url") == asset.url:
            if (target.stat().st_size != receipt.get("size") or sha256_file(target) != receipt.get("sha256")):
                raise ValueError(f"{target}: файл повреждён или изменён. Для скачивания укажите --force.")
            print("  Уже скачан, SHA-256 совпадает.", flush=True)
            return dict(receipt, local_path=str(target.resolve()))
        validate_asset_file(target, asset)
        print("  Используется готовый локальный файл; повторное скачивание не требуется.", flush=True)
        return {"url": asset.url, "size": target.stat().st_size, "sha256": sha256_file(target),
                "local_path": str(target.resolve()), "origin": "existing-local"}

    target.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Neuro-threshold-SAM-downloader/1", "Accept-Encoding": "identity"}
    if urlsplit(asset.url).hostname == "huggingface.co":
        credential = token() if callable(token) else token
        if credential and any(character.isspace() or not character.isascii() for character in credential):
            raise ValueError("Некорректный HF_TOKEN: ожидается токен без пробелов и переносов строк")
        if credential:
            headers["Authorization"] = f"Bearer {credential}"
    temporary = None
    print(f"Скачивание: {asset.path}", flush=True)
    try:
        with opener.open(Request(asset.url, headers=headers), timeout=timeout) as response:
            if response.status != 200:
                raise ValueError(f"Ожидался полный файл (HTTP 200), получен HTTP {response.status}")
            length = response.headers.get("Content-Length")
            total = int(length) if length is not None else None
            digest = hashlib.sha256()
            size = 0
            last_update = time.monotonic()
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}-",
                                             suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                while chunk := response.read(CHUNK_SIZE):
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    now = time.monotonic()
                    if now - last_update >= 1:
                        percent = f" / {total / CHUNK_SIZE:.1f} MiB ({100 * size / total:.1f}%)" if total else ""
                        print(f"  {size / CHUNK_SIZE:.1f} MiB{percent}", flush=True)
                        last_update = now
            if size == 0 or (total is not None and size != total):
                raise ValueError(f"Неполный файл: получено {size} байт, ожидалось {total}")
            validate_asset_file(temporary, asset)
            os.replace(temporary, target)
        print(f"  Сохранено: {target} ({size / CHUNK_SIZE:.1f} MiB)", flush=True)
        return {"url": asset.url, "size": size, "sha256": digest.hexdigest(),
                "local_path": str(target.resolve()), "origin": "download"}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _http_error_message(error, asset):
    # Never print a signed CDN URL, headers, bearer token or response body.
    if urlsplit(asset.url).hostname == "huggingface.co" and error.code in (401, 403):
        return (f"HTTP {error.code}: SAM3 требует доступа к https://huggingface.co/facebook/sam3. "
                "Примите условия, дождитесь доступа и задайте HF_TOKEN с правом чтения "
                "этого репозитория либо используйте --ask-token.")
    return f"HTTP {error.code}; источник: {urlsplit(asset.url).hostname}"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Скачать официальные веса SAM/SAM2/SAM3 и файлы для локальной загрузки. "
                    "Без установки PyTorch и моделей. По умолчанию: SAM ViT-B, SAM2.1 Tiny, SAM3.")
    parser.add_argument("--models", nargs="+", default=["sam", "sam2", "sam3"],
                        metavar="MODEL", help="Имена из --list; sam, sam2 и all — сокращения")
    parser.add_argument("--output", type=Path,
                        help="Папка сохранения; по умолчанию models/segment_anything или рядом с --weights-file")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--weights-file", type=Path, help="Готовые локальные веса одной выбранной модели; файл не копируется")
    reuse.add_argument("--extras-only", action="store_true", help="Только конфигурации/токенизатор, без проверки и скачивания весов")
    parser.add_argument("--list", action="store_true", help="Показать модели без скачивания")
    parser.add_argument("--dry-run", action="store_true", help="Показать файлы и URL без сети")
    parser.add_argument("--force", action="store_true", help="Повторно скачать и заменить выбранные файлы")
    parser.add_argument("--ask-token", action="store_true", help="Скрытый ввод токена Hugging Face для SAM3")
    parser.add_argument("--timeout", type=float, default=30, help="Тайм-аут сетевой операции, секунды")
    parser.add_argument("--retries", type=int, default=2, help="Повторы после временного сетевого сбоя (0–5)")
    args = parser.parse_args(argv)
    if args.list:
        for key, model in MODELS.items():
            print(f"{key:20} {model.name}")
        print("Сокращения: sam = sam-vit-b; sam2 = sam2.1-tiny; all = все варианты.")
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
            assets = MODELS[key].assets
            if args.extras_only:
                return tuple(asset for asset in assets if asset.kind != "weights")
            if weights_file:
                # Keep auxiliary files beside the explicitly selected weights by default.
                return tuple(asset if asset.kind == "weights" else replace(
                    asset, path=Path(asset.path).relative_to(Path(assets[0].path).parts[0]).as_posix()) for asset in assets)
            return assets

        print(f"Папка: {output}")
        if args.dry_run:
            for key in selected:
                for asset in assets_for(key):
                    if weights_file and asset.kind == "weights":
                        print(f"Готовые веса: {weights_file}")
                    else:
                        print(f"{asset.path}\n  {asset.url}")
            return 0
        manifest_path = output / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if not isinstance(manifest, dict):
            raise ValueError(f"Некорректный манифест: {manifest_path}")
        token = os.environ.get("HF_TOKEN", "").strip() or None
        token_asked = False

        def auth_token():
            nonlocal token, token_asked
            if args.ask_token and not token_asked:
                token = getpass.getpass("Hugging Face token (ввод скрыт): ").strip() or token
                token_asked = True
            return token
        output.mkdir(parents=True, exist_ok=True)
        opener = build_opener(SafeRedirectHandler())
        failed = []
        for key in selected:
            print(f"\n{MODELS[key].name}", flush=True)
            for asset in assets_for(key):
                for attempt in range(args.retries + 1):
                    try:
                        receipt = download_asset(asset, output, manifest, opener=opener,
                                                 token=auth_token, force=args.force, timeout=args.timeout,
                                                 existing_path=weights_file if asset.kind == "weights" else None)
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
        print("\nСкачивание завершено." if not failed else f"\nНе скачаны полностью: {', '.join(failed)}.")
        print("Веса не устанавливают backend: для применения нужны соответствующие пакеты моделей.")
        return 1 if failed else 0
    except KeyboardInterrupt:
        print("\nСкачивание остановлено. Готовые файлы сохранены; незавершённый файл удалён.", file=sys.stderr)
        return 130
    except (OSError, ValueError, EOFError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
