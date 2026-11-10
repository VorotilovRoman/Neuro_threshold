import re
from pathlib import Path

# Ищем: пробел + дата + пробел + время, за которыми идёт
# либо конец имени (перед расширением), либо " (N)"
PATTERN = re.compile(
    r"\s+\d{2}\.\d{2}\.\d{4}\s+\d{2}\.\d{2}\.\d{2}(?=(?:\s*\(\d+\))?\.[^.]+$)"
)

def rename_files(folder):
    """Rename files in an explicitly selected directory; skip collisions."""
    folder = Path(folder)
    renamed = 0
    skipped = 0

    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue

        new_name = PATTERN.sub("", file_path.name)

        if new_name == file_path.name:
            skipped += 1
            continue

        new_path = file_path.with_name(new_name)
        if new_path.exists():
            print(f"[ПРОПУСК] {file_path.name} -> {new_name} (файл уже существует)")
            skipped += 1
            continue

        file_path.rename(new_path)
        print(f"[OK] {file_path.name} -> {new_name}")
        renamed += 1

    print(f"\nГотово. Переименовано: {renamed}, пропущено: {skipped}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Remove timestamps from image filenames")
    parser.add_argument('folder', type=Path, help='Directory to rename (not recursive)')
    args = parser.parse_args()
    rename_files(args.folder)


if __name__ == '__main__':
    main()
