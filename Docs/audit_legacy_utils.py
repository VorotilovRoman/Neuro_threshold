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
    lines = [
        "# Оставшиеся потребители старой папки utils",
        "",
        "Проверка по AST импортов и строковым записям реестра. Дата: 21 ноября 2026.",
        "",
        "В utils осталось два Python-файла. Оба используются; удалять их сейчас нельзя.",
        "",
        "## Прямые импорты",
        "",
        "| Файл | Строка | Старый модуль |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| `{path}` | {line} | `{target}` |"
        for path, line, target in sorted(set(consumers))
    ]
    lines += [
        "",
        "## Вкладки и зависимости",
        "",
        "| Вкладка | Путь зависимости |",
        "| --- | --- |",
    ]
    lines += [
        f"| `{start}` | " + " → ".join(f"`{name}`" for name in chain) + " |"
        for start, chain in routes
    ]
    lines += [
        "",
        "Другие вкладки не имеют пути к utils по статическим Python-импортам. Строковые записи ленивого реестра перечислены отдельно: запись ещё не означает загрузку модуля.",
        "",
        "## Ленивый реестр",
        "",
        "| Файл | Строка | Модуль |",
        "| --- | --- | --- |",
    ]
    lines += [f"| `{path}` | {line} | `{target}` |" for path, line, target in registry]
    lines += [
        "",
        "## Следующие переносы",
        "",
        "- `yolo_find_img`: нормализация и чтение снимков → dataset_files_new; отдельно проверить специализированный fallback поиска.",
        "- `yolo_test_aeb`: LazyImageCache/read и Qt-конвертация → dataset_files_new + ui/image_conversion.",
        "- `demo_ensemble_methods`: loader Qt-контракта и ensemble_utils; разделить вычисления ансамбля и GUI, затем проверить веса/координаты/NMS/WBF.",
        "- После переноса актуализировать import_libs_internal и повторно запустить этот скрипт. Импорты utils_ops и utils_aeb относятся к другим папкам и в эту статистику не включены.",
        "",
    ]
    return "\n".join(lines).rstrip()


if __name__ == "__main__":
    print(render())
