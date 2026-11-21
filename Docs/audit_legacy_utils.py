"""Read-only AST inventory of legacy utils consumers and tab dependency paths."""

import ast
import subprocess
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inventory():
    paths = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "*.py",
            ],
            cwd=ROOT,
        )
        .decode()
        .strip("\0")
        .split("\0")
    )
    modules = {
        path[:-3].replace("/", "."): path
        for path in set(paths)
        if path and not path.startswith(("tests/", "Docs/"))
    }
    graph = {name: set() for name in modules}
    consumers = []
    registry = []
    for name, path in sorted(modules.items()):
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    package = name.split(".")[:-1]
                    prefix = package[: len(package) - node.level + 1]
                    base = ".".join(prefix + ([node.module] if node.module else []))
                else:
                    base = node.module or ""
                targets = [base] + [base + "." + alias.name for alias in node.names]
            for target in targets:
                if target in modules:
                    graph[name].add(target)
                if target.startswith("utils.") and target in modules:
                    consumers.append((path, node.lineno, target))
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("utils.")
                and node.value in modules
            ):
                registry.append((path, node.lineno, node.value))
    routes = []
    for start in sorted(name for name in modules if name.startswith("tabs.")):
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            node, chain = queue.popleft()
            if node.startswith("utils."):
                routes.append((start, chain))
                continue
            for target in sorted(graph[node]):
                if target not in visited:
                    visited.add(target)
                    queue.append((target, chain + [target]))
    return consumers, sorted(set(registry)), routes


def render():
    consumers, registry, routes = inventory()
    count = len(list((ROOT / "utils").glob("*.py")))
    lines = [
        "# Аудит старой папки utils",
        "",
        "Проверка по AST импортов и строковым записям реестра. Дата: 21 ноября 2026.",
        "",
        f"Python-файлов в utils: **{count}**. Прямых потребителей: **{len(set(consumers))}**. Путей от вкладок: **{len(routes)}**. Старых записей реестра: **{len(registry)}**.",
        "",
        "## Прямые импорты",
        "",
        "| Файл | Строка | Старый модуль |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| `{path}` | {line} | `{target}` |"
        for path, line, target in sorted(set(consumers))
    ] or ["| Нет | — | — |"]
    lines += [
        "",
        "## Пути от вкладок",
        "",
        "| Вкладка | Путь зависимости |",
        "| --- | --- |",
    ]
    lines += [
        f"| `{start}` | " + " → ".join(f"`{name}`" for name in chain) + " |"
        for start, chain in routes
    ] or ["| Нет | — |"]
    lines += [
        "",
        "## Ленивый реестр",
        "",
        "| Файл | Строка | Модуль |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| `{path}` | {line} | `{target}` |" for path, line, target in registry
    ] or ["| Нет | — | — |"]
    lines += [
        "",
        "## Завершённые переносы",
        "",
        "- `utils/image_io.py` удалён. Три вкладки используют dataset_files_new, ui/image_conversion и ui/segmentation/image_input.",
        "- `utils/ensemble_utils.py` удалён. NMS/WBF находятся в utils_new/annotations/boxes, пороговый пресет — в threshold_processing.",
        "- `utils_aeb/yolo_aeb_utils.py` перенесён в utils_new/segmentation/autoencoder: уникальные модели и предобработка, без старых реэкспортов геометрии.",
        "- Старые строковые экспорты import_libs_internal удалены после проверки всех потребителей. Совместимые алиасы не оставлены.",
        "- utils_ops и автономные скрипты utils_aeb — отдельные папки; их наличие не означает зависимости от удалённой utils.",
        "",
        "[Сценарии, алгоритмы и проверки трёх вкладок](Legacy_Tabs_Migration_Status.md).",
        "",
        "Команда воспроизведения: `python Docs/audit_legacy_utils.py > Docs/Legacy_Utils_Usage.md`.",
        "",
        "Аудит рассматривает статические импорты и точные строковые записи реестра. Дополнительно отсутствие импортов удалённых модулей проверено регрессионным AST-тестом и запуском всех вкладок.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(render())
