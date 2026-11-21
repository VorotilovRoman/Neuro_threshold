"""Print an AST inventory of source controls/dialogs without importing the app."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def assignments(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8-sig"))
    return {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def cell(value):
    return str(value).replace("|", "&#124;").replace("\n", " ")


def calls(node, scope="модуль"):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        scope = scope + "." + node.name
    if isinstance(node, ast.Call):
        yield scope.removeprefix("модуль."), node
    for child in ast.iter_child_nodes(node):
        yield from calls(child, scope)


def build():
    registry = assignments("ui/dataset_source.py")
    tabs = assignments("main_ui.py")["ALL_TABS"]
    actions = registry["ACTIONS"]
    lines = [
        "# Источники данных: воспроизводимый аудит",
        "",
        "[Как пользоваться](Source_Unification_Status.md) · [План](Source_Unification_Plan.md)",
        "",
        "Сгенерировано `python Docs/audit_sources.py > Docs/Source_Inventory.md`. "
        "Скрипт читает AST; не импортирует Qt/Torch и не открывает диалоги. "
        "Перечень вызовов включает скрытые формы и условные ветки; наличие вызова "
        "не доказывает доступность backend. Фактические сценарии и ограничения — в описании результата.",
        "",
        "## Назначение всех вкладок",
        "",
        "| ID | Вкладка | Описание из общего реестра UI |",
        "| --- | --- | --- |",
    ]
    for key, title in tabs:
        lines.append(
            f"| `{key}` | {title} | {cell(registry['TAB_DESCRIPTIONS'][key])} |"
        )
    lines += [
        "",
        "## Общие действия",
        "",
        "| Ключ | Кнопка | Источник / фильтр | Подсказка |",
        "| --- | --- | --- | --- |",
    ]
    for key in registry["ORDER"]:
        caption, _, filters, folder, hint = actions[key]
        lines.append(
            f"| `{key}` | {caption} | {cell('Папка' if folder else filters)} | {cell(hint)} |"
        )
    panels, dialogs, other = [], [], []
    for folder in ("tabs", "ui", "ui_tabs", "utils_new"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for scope, call in calls(tree):
                name = (
                    call.func.attr
                    if isinstance(call.func, ast.Attribute)
                    else getattr(call.func, "id", "")
                )
                link = f"[{rel}:{call.lineno}](../{rel}#L{call.lineno})"
                if name in (
                    "DatasetSourceWidget",
                    "ImageNavigationWidget",
                    "make_source_button",
                ):
                    panels.append((link, scope, ast.unparse(call)))
                if name in ("get_open_file", "get_directory", "choose_coco_sources"):
                    dialogs.append((link, scope, ast.unparse(call)))
                if name in (
                    "getOpenFileName",
                    "getOpenFileNames",
                    "getExistingDirectory",
                ):
                    other.append((link, scope, ast.unparse(call)))
    for title, rows in (
        ("Конфигурации UI и кнопок", panels),
        ("Обработчики общих диалогов", dialogs),
        ("Нативные диалоги: владелец и отдельные пути экспорта", other),
    ):
        lines += [
            "",
            "## " + title,
            "",
            "| Файл | Область | Вызов |",
            "| --- | --- | --- |",
        ]
        for link, scope, code in rows:
            lines.append(f"| {link} | `{scope}` | `{cell(code)}` |")
    lines += [
        "",
        "Последняя таблица содержит также выбор **выходных** каталогов. Эти операции "
        "не являются импортом и не переносятся в панель источника. Все getOpenFileName(s) "
        "принадлежат ui/dataset_source.py. Смена типа данных меняет назначение кнопки labels "
        "на masks; переменная kind в обработчике отражает это состояние.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(build(), end="")
