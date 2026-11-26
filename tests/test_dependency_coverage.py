"""Cross-check actual imports, declared profiles and native classical ML backends."""

import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import dependency_health as health
import install

ROOT = Path(__file__).resolve().parents[1]
ALIASES = dict(PIL='Pillow', cv2='opencv-python-headless', skimage='scikit-image',
    sklearn='scikit-learn', yaml='PyYAML', open_clip='open-clip-torch',
    umap='umap-learn', sam2='SAM-2', segmentation_models_pytorch='segmentation-models-pytorch',
    xgboost=health.selected_xgboost())


class DependencyCoverageTests(unittest.TestCase):
    def test_all_external_imports_are_declared_in_base_or_optional_requirements(self):
        declarations = set()
        base = set()
        for path in [ROOT / 'requirements.txt', *sorted((ROOT / 'requirements').glob('optional-*.txt'))]:
            for line in health.requirement_lines(path):
                requirement = Requirement(line)
                if not requirement.marker or requirement.marker.evaluate():
                    declarations.add(canonicalize_name(requirement.name))
                    if path == ROOT / 'requirements.txt':
                        base.add(canonicalize_name(requirement.name))
        required_core = {canonicalize_name(ALIASES.get(module.split('.')[0], module.split('.')[0]))
                         for module in health.CORE_MODULES}
        self.assertFalse(required_core - base, f'Core features declared only as extras: {required_core - base}')
        local = {p.stem for p in ROOT.glob('*.py')} | {p.name for p in ROOT.iterdir() if p.is_dir()}
        paths = list(ROOT.glob('*.py'))
        for folder in ('tabs', 'ui', 'ui_tabs', 'utils_new', 'cluster'):
            paths.extend((ROOT / folder).rglob('*.py'))
        missing = set()
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    modules = [node.module]
                elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                    name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
                    if name in ('import_module', 'find_spec', '__import__') and isinstance(node.args[0].value, str):
                        modules = [node.args[0].value]
                for module in modules:
                    module = module.split('.')[0]
                    if module in local or module in sys.stdlib_module_names:
                        continue
                    if canonicalize_name(ALIASES.get(module, module)) not in declarations:
                        missing.add(f'{module}: {path.relative_to(ROOT)}')
        self.assertFalse(missing, '\n'.join(sorted(missing)))

    def test_pep508_markers_ranges_and_unpinned_packages_are_checked(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'optional.txt'
            path.write_text('missing-feature\nold-feature>=2\nskipped-feature; python_version > "9"\n')
            def version(name):
                if name == 'missing-feature':
                    raise health.metadata.PackageNotFoundError(name)
                return '1.0'
            with patch.object(health.metadata, 'version', side_effect=version):
                errors = health.direct_requirement_errors(path)
            self.assertEqual(len(errors), 2)
            self.assertTrue(any('missing-feature' in value for value in errors))
            self.assertFalse(any('skipped-feature' in value for value in errors))

    def test_pinned_vcs_sources_are_verified_without_importing_models(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'sam.txt'
            path.write_text('SAM-2 @ git+https://github.com/facebookresearch/sam2.git@abcdef\n')
            for revision, expected in (('abcdef', 0), ('wrong', 1)):
                source = dict(url='https://github.com/facebookresearch/sam2.git', vcs_info=dict(commit_id=revision))
                distribution = Mock(read_text=Mock(return_value=json.dumps(source)))
                with patch.object(health.metadata, 'version', return_value='1.0'), \
                        patch.object(health.metadata, 'distribution', return_value=distribution):
                    self.assertEqual(len(health.direct_requirement_errors(path)), expected)

    def test_all_extras_are_checked_and_imported_once_after_core_torch(self):
        module = SimpleNamespace(__version__='test')
        with patch.object(health, 'check_environment', return_value=([], [])) as check, \
                patch.object(health, 'installed_opencv', return_value={}), \
                patch.object(health, 'check_torch_device'), \
                patch.object(health.importlib, 'import_module', return_value=module) as imports, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(health.main(['--load', '--extra', 'all']), 0)
        self.assertEqual(set(check.call_args.kwargs['extras']), set(health.EXTRA_MODULES))
        names = [call.args[0] for call in imports.call_args_list]
        self.assertEqual(names[:2], ['torch', 'torchvision'])
        self.assertEqual(names.count('transformers'), 1)
        self.assertTrue({'xgboost', 'pyamg', 'segment_anything', 'sam2', 'open_clip', 'sahi'} <= set(names))

    def test_installation_verifies_selected_extras_in_the_new_process(self):
        with patch.object(install.sys, 'prefix', '/temporary/venv'), \
                patch.object(install, 'check_installation_environment'), \
                patch.object(install, 'normalize_opencv'), patch.object(install, 'normalize_xgboost'), \
                patch.object(install, 'run_pip'), patch.object(install.subprocess, 'run') as verify:
            self.assertEqual(install.main(['--torch', 'cpu', '--extra', 'training', '--extra', 'audit']), 0)
        command = verify.call_args.args[0]
        self.assertIn('training', command)
        self.assertIn('audit', command)
        self.assertEqual(command.count('--extra'), 2)

    def test_xgboost_shared_files_are_repaired_after_provider_migration(self):
        with patch.object(install, 'installed_xgboost', return_value={'xgboost': '3.1', 'xgboost-cpu': '3.4.1'}), \
                patch.object(install, 'selected_xgboost', return_value='xgboost-cpu'), \
                patch.object(install, 'run_pip') as pip:
            install.normalize_xgboost()
        self.assertEqual(pip.call_args_list[0].args, ('uninstall', '-y', 'xgboost'))
        self.assertEqual(pip.call_args_list[1].args,
                         ('install', '--no-deps', '--force-reinstall', 'xgboost-cpu==3.4.1'))

    def test_real_xgboost_fit_predict_and_random_walker_multigrid(self):
        import numpy as np
        from utils_new.segmentation.backends.traditional_ml import train_xgboost
        from utils_new.segmentation.backends.interactive import random_walker_segmentation
        features = np.array([[i / 20, int(i >= 10)] for i in range(20)], np.float32)
        target = np.array([int(i >= 10) for i in range(20)])
        model = train_xgboost(features, target, n_estimators=5)
        self.assertGreaterEqual(float((model.predict(features) == target).mean()), 0.9)
        image = np.zeros((24, 24, 3), np.uint8)
        image[:, 12:] = 220
        markers = np.zeros((24, 24), np.int32)
        markers[12, 4], markers[12, 20] = 2, 1
        mask = random_walker_segmentation(image, markers, mode='cg_mg', use_color=False)
        self.assertEqual(mask[12, 4], 0)
        self.assertEqual(mask[12, 20], 255)


@unittest.skipUnless(os.environ.get('NEURO_RUN_GUI_TESTS') == '1', 'opt-in Qt')
class MissingXGBoostGuiTests(unittest.TestCase):
    def test_missing_package_disables_only_xgboost_and_explains_installation(self):
        from PyQt5.QtWidgets import QApplication
        from tabs.traditional_ml_methods import TraditionalMLWindow
        app = QApplication.instance() or QApplication([])
        with patch('ui_tabs.traditional_ml_methods_ui.find_spec', return_value=None):
            owner = TraditionalMLWindow()
        try:
            item = owner.model_combo.model().item(owner.model_combo.findText('XGBoost'))
            self.assertFalse(item.isEnabled())
            self.assertIn('install.py', item.toolTip())
            self.assertTrue(owner.model_combo.model().item(0).isEnabled())
        finally:
            owner.close()
            app.processEvents()
