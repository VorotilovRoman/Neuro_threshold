"""Regression checks for dependency boundaries; run with unittest discover."""
import ast
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACADES = {'import_libs_external', 'import_libs_internal', 'import_libs_methods_ui'}


def python_sources():
    # Only project sources, never virtual environments or generated output.
    import subprocess
    paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z', '*.py'], cwd=ROOT, text=True)
    return [ROOT / p for p in sorted(set(paths.rstrip('\0').split('\0'))) if not p.startswith('Docs/')]


def eager_imports(nodes):
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            yield from eager_imports(node.body)
            yield from eager_imports(getattr(node, 'orelse', []))
            yield from eager_imports(getattr(node, 'finalbody', []))
            for handler in getattr(node, 'handlers', []):
                yield from eager_imports(handler.body)


class ArchitectureTests(unittest.TestCase):
    def run_python(self, code):
        with tempfile.TemporaryDirectory() as config:
            env = dict(os.environ, QT_QPA_PLATFORM='offscreen',
                       XDG_CONFIG_HOME=config, XDG_CACHE_HOME=config, MPLCONFIGDIR=config,
                       YOLO_CONFIG_DIR=config, NO_ALBUMENTATIONS_UPDATE='1')
            result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                    env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_explicit_imports(self):
        for path in python_sources():
            with self.subTest(path=path.relative_to(ROOT)):
                for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse(any(a.name == '*' for a in node.names),
                                         f'{path}:{node.lineno}: wildcard import')
                        self.assertNotIn(node.module, FACADES,
                                         f'{path}:{node.lineno}: internal facade use')

    def test_no_eager_project_cycles(self):
        modules = {str(p.relative_to(ROOT).with_suffix('')).replace('/', '.'): p
                   for p in python_sources() if p.stem not in FACADES}
        graph = {}
        for name, path in modules.items():
            deps = set()
            for node in eager_imports(ast.parse(path.read_text(encoding='utf-8')).body):
                if isinstance(node, ast.Import):
                    deps.update(a.name for a in node.names)
                else:
                    base = node.module or ''
                    if node.level:
                        package = name.rpartition('.')[0]
                        base = importlib.util.resolve_name('.' * node.level + base, package)
                    deps.add(base)
                    deps.update(base + '.' + a.name for a in node.names)
            graph[name] = deps & modules.keys()
        visited = set()
        def visit(name, stack):
            self.assertNotIn(name, stack, ' -> '.join(stack + [name]))
            if name in visited:
                return
            for dep in sorted(graph[name]):
                visit(dep, stack + [name])
            visited.add(name)
        for name in graph:
            visit(name, [])

    def test_legacy_scripts_defer_mutation_until_main(self):
        paths = [ROOT / 'rename_data.py', *(ROOT / 'utils_aeb').glob('auto-*.py')]
        forbidden = ('os.makedirs', 'random.seed', 'np.random.seed', 'torch.manual_seed',
                     'torch.cuda.manual_seed_all')
        def calls(nodes):
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    continue
                if isinstance(node, ast.If) and '__name__' in ast.unparse(node.test):
                    continue
                if isinstance(node, ast.Call):
                    yield ast.unparse(node.func)
                yield from calls(list(ast.iter_child_nodes(node)))
        for path in paths:
            with self.subTest(path=path.name):
                used = set(calls(ast.parse(path.read_text()).body))
                self.assertFalse(used & set(forbidden), used & set(forbidden))

    def test_legacy_facades_are_lazy(self):
        self.run_python('''import importlib, sys
for name in ('import_libs_external', 'import_libs_internal', 'import_libs_methods_ui'):
    module = importlib.import_module(name)
    assert 'torch' not in sys.modules
    assert 'cv2' not in sys.modules
    assert 'PyQt5' not in sys.modules
    assert 'YOLO' in dir(module) or name == 'import_libs_methods_ui'
''')

    @unittest.skipUnless(importlib.util.find_spec('cv2') and importlib.util.find_spec('PyQt5'),
                         'GUI runtime dependencies are not installed')
    def test_basic_modules_do_not_import_ml(self):
        self.run_python('''import builtins, sys
blocked = {'torch','torchvision','ultralytics','onnxruntime','albumentations'}
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('Unexpected heavy import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
import utils_new.annotations
import ui.annotation_review_view
import utils_new.annotations.augmentation
import ui.segmentation.preset_bindings
import ui.settings_service
import utils_new.DatasetGenerator_new
import main_ui
import tabs.threshold_methods
import tabs.layout_dataset
import tabs.deep_learning_training
assert not (blocked & sys.modules.keys())
''')


if __name__ == '__main__':
    unittest.main()
