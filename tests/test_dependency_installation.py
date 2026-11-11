"""Checks for the installer policy without mutating the test environment."""
import subprocess
import tempfile
import unittest
from pathlib import Path
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

    def test_cpu_and_gpu_pairs_are_explicit(self):
        for backend in install.TORCH_PROFILES:
            text = (install.ROOT / 'requirements' / f'torch-{backend}.txt').read_text()
            self.assertIn(f'torch==2.11.0+{backend}', text)
            self.assertIn(f'torchvision==0.26.0+{backend}', text)
            self.assertIn(f'https://download.pytorch.org/whl/{backend}', text)


if __name__ == '__main__':
    unittest.main()
