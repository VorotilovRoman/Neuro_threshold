import re
from pathlib import Path

FOLDER = Path(r"C:\Users\prodis\PycharmProjects\Neuro_threshold\models\dataset_bga_seq\images")

# Ищем: пробел + дата + пробел + время, за которыми идёт
# либо конец имени (перед расширением), либо " (N)"
PATTERN = re.compile(
    r"\s+\d{2}\.\d{2}\.\d{4}\s+\d{2}\.\d{2}\.\d{2}(?=(?:\s*\(\d+\))?\.[^.]+$)"
)

renamed = 0
skipped = 0

for file_path in FOLDER.iterdir():
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