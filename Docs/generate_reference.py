#!/usr/bin/env python3
"""Rebuild the import/API and exact-duplicate references without importing project code."""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'Docs'


def source_link(path, line=1, label=None):
    return f'[{label or path}](../{path}#L{line})'


def module_link(module):
    return f'[{module}](#{module.replace(".", "-").replace("_", "-")})'


def cell(value):
    return str(value).replace('|', '&#124;').replace('\n', ' ')


def collect():
    records = {}
    for path in sorted(ROOT.rglob('*.py')):
        rel = path.relative_to(ROOT)
        if rel.parts[0] in ('Docs', '.git') or any(p.startswith('.') for p in rel.parts):
            continue
        module = '.'.join(rel.with_suffix('').parts)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(rel))
        records[module] = (rel.as_posix(), tree)
    packages = {m[:-9]: m for m in records if m.endswith('.__init__')}
    imports = defaultdict(list)
    for module, (_, tree) in records.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [(n.name, '', n.asname) for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    prefix = module.split('.')[:-node.level]
                    base = '.'.join(prefix + ([base] if base else []))
                names = [(base, n.name, n.asname) for n in node.names]
            else:
                continue
            for base, name, alias in names:
                child = f'{base}.{name}' if name and name != '*' else ''
                local = child if child in records else (base if base in records else packages.get(base))
                imports[module].append((base, name, alias, node.lineno, local))
    return records, imports


def scope_definitions(node):
    """Include module-level definitions inside if/try, without nested locals."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield child
        else:
            yield from scope_definitions(child)


def build():
    records, imports = collect()
    incoming = defaultdict(set)
    symbol_importers = defaultdict(set)
    for module, entries in imports.items():
        for base, name, alias, line, local in entries:
            if local:
                incoming[local].add(module)
                if name and name != '*':
                    symbol_importers[(local, name)].add(module)
    lines = ['# Справочник модулей, импортов и API', '',
             '[Навигация](README.md) · [Карта архитектуры](Architecture.md) · [Дублирование](Duplication.md)', '',
             'Сгенерировано `python Docs/generate_reference.py` из текущих исходников. '
             'Каталог исключает саму документацию и скрытые каталоги.', '',
             f'Проанализировано Python-модулей: **{len(records)}**. '
             'Импорты учитываются на всех уровнях AST, включая функции и try/except; '
             'наличие записи не означает, что ветка выполняется при запуске. '
             'Звёздочные импорты не разворачиваются в список фактических вызовов. '
             'QProcess, строковые динамические импорты и файловые зависимости описаны отдельно в карте.', '',
             'Для всех модулей перечислены верхнеуровневые функции и классы. '
             'Для всех каталогов утилит дополнительно перечислены сигнатуры методов классов. '
             'Вложенные локальные функции не являются отдельным API и в список не включены. '
             'Приватные имена сохранены для поиска реализации.', '', '## Содержание', '']
    for module in records:
        lines.append(f'- {module_link(module)}')
    lines += ['', '## Внешние импортируемые модули', '',
              'Имена слева — Python imports, а не обязательно имена пакетов PyPI. '
              'Версии и назначение — в [зависимостях](Dependencies.md). '
              'Стандартная библиотека не включена в эту таблицу.', '',
              '| Import | Файлы-потребители (включая условные импорты) |', '| --- | --- |']
    external = defaultdict(set)
    for module, entries in imports.items():
        for base, name, alias, line, local in entries:
            if not local and base.split('.')[0] not in sys.stdlib_module_names and base != '__future__':
                external[base.split('.')[0]].add(module)
    for name, users in sorted(external.items()):
        lines.append(f'| `{name}` | '+', '.join(module_link(u) for u in sorted(users))+' |')
    duplicates = defaultdict(list)
    for module, (path, tree) in records.items():
        lines += ['', f'<a id="{module.replace(".", "-").replace("_", "-")}"></a>', '', f'## {module}', '',
                  f'Исходник: {source_link(path)}.', '',
                  '**Прямые потребители:** '+(', '.join(module_link(u) for u in sorted(incoming[module])) or 'не найдены в import-графе; это не доказательство неиспользования')+'.', '',
                  '**Прямые локальные импорты:**']
        loc = defaultdict(set)
        ext = set()
        for base, name, alias, line, local in imports[module]:
            if local:
                loc[local].add((name or '(модуль)') + (f' as {alias}' if alias else ''))
            else:
                ext.add(base + (f'.{name}' if name else ''))
        for target, names in sorted(loc.items()):
            lines.append(f'- {module_link(target)}: '+', '.join(f'`{n}`' for n in sorted(names)))
        if not loc:
            lines.append('- нет')
        lines += ['', '**Внешние/stdlib импорты:** '+(', '.join(f'`{x}`' for x in sorted(ext)) or 'нет')+'.', '', '**Объявления:**', '']
        utility = module.split('.')[0].startswith('utils')
        for node in scope_definitions(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = cell(f'{node.name}({ast.unparse(node.args)})')
                doc = cell((ast.get_docstring(node) or '').split('\n')[0])
                lines.append(f'- {source_link(path, node.lineno, "`"+sig+"`")}'+(f' — {doc}' if doc else ''))
                users = symbol_importers[(module, node.name)]
                if users:
                    lines.append('  Явно импортируют: '+', '.join(module_link(u) for u in sorted(users))+'.')
            elif isinstance(node, ast.ClassDef):
                bases = ', '.join(ast.unparse(b) for b in node.bases)
                lines.append(f'- Класс {source_link(path, node.lineno, "`"+node.name+"`")}'+(f' (`{bases}`)' if bases else ''))
                if utility:
                    for method in node.body:
                        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            sig = cell(f'{node.name}.{method.name}({ast.unparse(method.args)})')
                            lines.append(f'  - {source_link(path, method.lineno, "`"+sig+"`")}')
        if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in scope_definitions(tree)):
            lines.append('- Нет собственных верхнеуровневых функций/классов; см. импорты и исполняемый код.')
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            length = node.end_lineno - node.lineno + 1
            if length < 9:
                continue
            normalized = ast.parse(ast.unparse(node)).body[0]
            normalized.name = 'FUNCTION'
            if ast.get_docstring(normalized):
                normalized.body = normalized.body[1:]
            key = ast.dump(normalized, include_attributes=False)
            duplicates[key].append((path, node.name, node.lineno, length))
    (DOCS / 'Module_Reference.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    lines = ['# Подтверждённые структурные копии', '', '[Разбор и рекомендации](Duplication.md) · [Справочник](Module_Reference.md)', '',
             'Автоматическое приложение: одинаковые AST функций/методов в разных файлах, '
             'от 9 строк исходника. Имя функции, docstring и номера строк игнорируются; '
             'аргументы и тело сохраняются. Сигнатуры и окружающие импорты проверяйте перед объединением. '
             'Это не поиск похожих алгоритмов и не доказательство взаимозаменяемости классов.', '']
    groups = [v for v in duplicates.values() if len({x[0] for x in v}) > 1]
    groups.sort(key=lambda v: (-max(x[3] for x in v), v[0]))
    lines += [f'Групп совпадений: **{len(groups)}**.', '']
    for index, group in enumerate(groups, 1):
        lines += [f'## {index}. {group[0][1]}', '']
        for path, name, line, length in group:
            lines.append(f'- {source_link(path, line, path+":"+str(line))} — `{name}`, {length} строк')
        lines.append('')
    (DOCS / 'Duplicate_Reference.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'{len(records)} modules; {len(groups)} cross-file duplicate groups')


if __name__ == '__main__':
    build()
