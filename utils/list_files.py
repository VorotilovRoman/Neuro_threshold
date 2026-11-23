"""Save a recursive UTF-8 inventory: python utils/list_files.py SOURCE OUTPUT."""

import argparse
import os
from pathlib import Path
import tempfile


def save_file_list(source, output):
    """Return (report path, file count, directory count); do not follow directory links."""
    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(f"Не папка: {source}")

    output = Path(output).expanduser()
    if output.is_dir() or not output.suffix:
        output = output / "files_list.txt"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files_count = directories_count = 0
    temporary = None

    def scan_error(error):
        # An inaccessible subtree must not silently produce an incomplete report.
        raise error

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=".files-list-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(f"Папка: {source}\n\n")
            for folder, directories, files in os.walk(
                source, followlinks=False, onerror=scan_error
            ):
                directories.sort(key=lambda name: (name.casefold(), name))
                files.sort(key=lambda name: (name.casefold(), name))
                folder = Path(folder)
                for name in directories:
                    relative = (folder / name).relative_to(source).as_posix()
                    stream.write(f"[DIR] {relative}/\n")
                    directories_count += 1
                # Windows junctions are directory links too, but islink() can be false.
                is_junction = getattr(os.path, "isjunction", lambda _path: False)
                directories[:] = [
                    name for name in directories if not is_junction(folder / name)
                ]
                for name in files:
                    path = folder / name
                    if path == temporary or path == output:
                        continue
                    stream.write(f"[FILE] {path.relative_to(source).as_posix()}\n")
                    files_count += 1
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output, files_count, directories_count


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Рекурсивный список всех файлов и подпапок в UTF-8 TXT. "
        "Ссылки на каталоги перечисляются без обхода их содержимого."
    )
    parser.add_argument("source", help="Папка, которую нужно просканировать")
    parser.add_argument(
        "output",
        help="Файл TXT или папка для files_list.txt; путь без расширения считается папкой",
    )
    args = parser.parse_args(argv)
    try:
        report, files, directories = save_file_list(args.source, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Ошибка: {error}\n")
    print(f"Сохранено: {report}\nФайлов: {files}; папок: {directories}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
