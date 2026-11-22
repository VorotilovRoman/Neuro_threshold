"""Checks for the installer policy without mutating the test environment."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dependency_health
import install


class InstallationTests(unittest.TestCase):
    def test_removing_desktop_always_restores_headless_files(self):
        with patch.object(install, 'installed_opencv', return_value={
                'opencv-python': '4.13.0.92', 'opencv-python-headless': '4.13.0.92'}), \
                patch.object(install, 'run_pip') as pip:
            install.normalize_opencv()
        self.assertEqual(pip.call_args_list[0].args,
                         ('uninstall', '-y', 'opencv-python'))
        self.assertEqual(pip.call_args_list[1].args,
                         ('install', '--no-deps', '--force-reinstall', 'opencv-python-headless==4.13.0.92'))

    def test_single_provider_is_not_reinstalled(self):
        with patch.object(install, 'installed_opencv', return_value={'opencv-python-headless': '4.13.0.92'}), \
                patch.object(install, 'run_pip') as pip:
            install.normalize_opencv()
        pip.assert_not_called()

    def test_explicit_repair_restores_single_provider_files(self):
        with patch.object(install, 'installed_opencv', return_value={'opencv-python-headless': '4.13.0.92'}), \
                patch.object(install, 'run_pip') as pip:
            install.normalize_opencv(force=True)
        pip.assert_called_once_with('install', '--no-deps', '--force-reinstall',
                                    'opencv-python-headless==4.13.0.92', dry_run=False)

    def test_single_provider_with_wrong_version_is_replaced(self):
        with patch.object(install, 'installed_opencv', return_value={'opencv-python-headless': '5.0.0.93'}), \
                patch.object(install, 'run_pip') as pip:
            install.normalize_opencv()
        pip.assert_called_once_with('install', '--no-deps', '--force-reinstall',
                                    'opencv-python-headless==4.13.0.92', dry_run=False)

    def test_known_metadata_exception_does_not_hide_other_errors(self):
        output = ('ultralytics 8.4.36 requires opencv-python, which is not installed.\n'
                  'example 1 requires missing-package, which is not installed.\n')
        result = subprocess.CompletedProcess([], 1, output, '')
        with patch.object(dependency_health, 'installed_opencv',
                          return_value={'opencv-python-headless': '4.13.0.92'}), \
                patch.object(dependency_health, 'direct_requirement_errors', return_value=[]), \
                patch.object(dependency_health.subprocess, 'run', return_value=result):
            errors, notes = dependency_health.check_environment()
        self.assertEqual(len(errors), 1)
        self.assertIn('example', errors[0])
        self.assertEqual(len(notes), 1)

    def test_installer_stops_after_pip_failure(self):
        with patch.object(install, 'run_pip', side_effect=subprocess.CalledProcessError(1, 'pip')), \
                patch.object(install.sys, 'prefix', '/temporary/venv'), \
                patch.object(install, 'normalize_opencv') as normalize:
            self.assertEqual(install.main([]), 1)
        normalize.assert_not_called()

    def test_direct_pin_check_accepts_local_torch_builds(self):
        with tempfile.TemporaryDirectory() as directory:
            requirements = Path(directory) / 'requirements.txt'
            requirements.write_text('torch==2.11.0\n')
            with patch.object(dependency_health.metadata, 'version', return_value='2.11.0+cpu'):
                self.assertEqual(dependency_health.direct_requirement_errors(requirements), [])
            with patch.object(dependency_health.metadata, 'version', return_value='2.12.0+cpu'):
                self.assertEqual(len(dependency_health.direct_requirement_errors(requirements)), 1)

    def test_exact_cuda_pins_reject_cpu_or_other_cuda_builds(self):
        with tempfile.TemporaryDirectory() as directory:
            requirements = Path(directory) / 'requirements.txt'
            requirements.write_text('torch==2.11.0+cu126\ntorchvision==0.26.0+cu126\n')
            for suffix in ('cpu', 'cu130', 'cu126'):
                with self.subTest(suffix=suffix), patch.object(
                        dependency_health.metadata, 'version',
                        side_effect=lambda name: ('2.11.0+' if name == 'torch' else '0.26.0+') + suffix):
                    errors = dependency_health.direct_requirement_errors(requirements)
                    self.assertEqual(len(errors), 0 if suffix == 'cu126' else 2)

    def test_cu126_profile_is_passed_to_installer(self):
        with patch.object(install, 'run_pip') as pip, \
                patch.object(install, 'normalize_opencv'):
            self.assertEqual(install.main(['--torch', 'cu126', '--dry-run']), 0)
        self.assertEqual(pip.call_args_list[0].args,
                         ('install', '-r', str(install.ROOT / 'requirements/torch-cu126.txt')))
        self.assertTrue(pip.call_args_list[0].kwargs['dry_run'])

    def test_extras_resolve_together_with_exact_cuda_profile(self):
        with patch.object(install, 'run_pip') as pip, \
                patch.object(install, 'normalize_opencv'):
            self.assertEqual(install.main(['--torch', 'cu126', '--extra', 'training', '--dry-run']), 0)
        pip.assert_called_once_with(
            'install', '-r', str(install.ROOT / 'requirements/torch-cu126.txt'),
            '-r', str(install.ROOT / 'requirements/optional-training.txt'), dry_run=True)

    def test_installed_profile_uses_metadata_without_guessing_unknown_builds(self):
        for profile in install.TORCH_PROFILES:
            with self.subTest(profile=profile), patch.object(
                    dependency_health.metadata, 'version', return_value='2.11.0+' + profile):
                self.assertEqual(dependency_health.installed_torch_profile(), profile)
        with patch.object(dependency_health.metadata, 'version',
                          side_effect=dependency_health.metadata.PackageNotFoundError('torch')):
            self.assertIsNone(dependency_health.installed_torch_profile())
        for version in ('2.11.0', '2.11.0+cu128'):
            with self.subTest(version=version), patch.object(
                    dependency_health.metadata, 'version', return_value=version):
                with self.assertRaisesRegex(ValueError, 'Select --torch'):
                    dependency_health.installed_torch_profile()

    def test_adding_extras_preserves_existing_cuda_profile(self):
        with patch.object(install, 'installed_torch_profile', return_value='cu126'), \
                patch.object(install, 'run_pip') as pip, \
                patch.object(install, 'normalize_opencv'):
            self.assertEqual(install.main(['--extra', 'training', '--dry-run']), 0)
        self.assertIn(str(install.ROOT / 'requirements/torch-cu126.txt'), pip.call_args.args)

    def test_fresh_environment_defaults_to_cpu(self):
        with patch.object(install, 'installed_torch_profile', return_value=None), \
                patch.object(install, 'run_pip') as pip, \
                patch.object(install, 'normalize_opencv'):
            self.assertEqual(install.main(['--dry-run']), 0)
        self.assertIn(str(install.ROOT / 'requirements/torch-cpu.txt'), pip.call_args.args)

    def test_ambiguous_installed_build_stops_before_changing_packages(self):
        with patch.object(install, 'installed_torch_profile', side_effect=ValueError('Select --torch')), \
                patch.object(install, 'run_pip') as pip, \
                patch.object(install, 'normalize_opencv') as normalize:
            self.assertEqual(install.main(['--extra', 'training', '--dry-run']), 1)
        pip.assert_not_called()
        normalize.assert_not_called()

    def test_explicit_profile_overrides_existing_build_without_detection(self):
        with patch.object(install, 'installed_torch_profile') as detect, \
                patch.object(install, 'run_pip') as pip, \
                patch.object(install, 'normalize_opencv'):
            self.assertEqual(install.main(['--torch', 'cpu', '--dry-run']), 0)
        detect.assert_not_called()
        self.assertIn(str(install.ROOT / 'requirements/torch-cpu.txt'), pip.call_args.args)

    def test_gpu_installation_requires_successful_real_cuda_verification(self):
        with patch.object(install, 'run_pip'), \
                patch.object(install, 'normalize_opencv'), \
                patch.object(install.sys, 'prefix', '/temporary/venv'), \
                patch.object(install.subprocess, 'run',
                             side_effect=subprocess.CalledProcessError(1, 'CUDA unavailable')) as verify:
            self.assertEqual(install.main(['--torch', 'cu126']), 1)
        command = verify.call_args.args[0]
        self.assertEqual(command[-4:], ['--load', '--torch-profile', 'cu126', '--cuda-test'])

    def test_cpu_installation_does_not_require_a_gpu(self):
        with patch.object(install, 'run_pip'), \
                patch.object(install, 'normalize_opencv'), \
                patch.object(install.sys, 'prefix', '/temporary/venv'), \
                patch.object(install.subprocess, 'run') as verify:
            self.assertEqual(install.main(['--torch', 'cpu']), 0)
        command = verify.call_args.args[0]
        self.assertIn('--torch-profile', command)
        self.assertNotIn('--cuda-test', command)

    def test_no_gpu_and_wrong_runtime_are_not_reported_as_success(self):
        for profile, runtime, message in [('cu126', None, 'runtime 12.6'),
                                          ('cu126', '13.0', 'runtime 12.6'),
                                          ('cu126', '12.6', 'CUDA is unavailable'),
                                          ('cpu', '12.6', 'CPU profile')]:
            torch = SimpleNamespace(version=SimpleNamespace(cuda=runtime),
                                    cuda=SimpleNamespace(is_available=lambda: False))
            with self.subTest(profile=profile, runtime=runtime), self.assertRaisesRegex(RuntimeError, message):
                dependency_health.check_torch_device(torch, None, profile=profile, cuda_test=True)

    def test_metadata_failure_prevents_native_imports(self):
        with patch.object(dependency_health, 'check_environment', return_value=(['wrong CUDA wheel'], [])), \
                patch.object(dependency_health, 'installed_opencv', return_value={}), \
                patch.object(dependency_health.importlib, 'import_module') as loader:
            self.assertEqual(dependency_health.main(['--cuda-test', '--torch-profile', 'cu126']), 1)
        loader.assert_not_called()

    def test_cuda_test_loads_torch_before_other_native_libraries_and_propagates_failure(self):
        module = SimpleNamespace(__version__='test')
        with patch.object(dependency_health, 'check_environment', return_value=([], [])), \
                patch.object(dependency_health, 'installed_opencv', return_value={}), \
                patch.object(dependency_health.importlib, 'import_module', return_value=module) as loader, \
                patch.object(dependency_health, 'check_torch_device',
                             side_effect=RuntimeError('CUDA NMS unavailable')) as device:
            self.assertEqual(dependency_health.main(['--cuda-test', '--torch-profile', 'cu126']), 1)
        names = [call.args[0] for call in loader.call_args_list]
        self.assertEqual(names[:2], ['torch', 'torchvision'])
        device.assert_called_once_with(module, module, profile='cu126', cuda_test=True)

    def test_cpu_and_gpu_pairs_are_explicit(self):
        for backend in install.TORCH_PROFILES:
            text = (install.ROOT / 'requirements' / f'torch-{backend}.txt').read_text()
            self.assertIn(f'torch==2.11.0+{backend}', text)
            self.assertIn(f'torchvision==0.26.0+{backend}', text)
            self.assertIn(f'https://download.pytorch.org/whl/{backend}', text)
            self.assertIn('-r ../requirements.txt', text)
            self.assertIn('--extra-index-url ', text)


if __name__ == '__main__':
    unittest.main()
