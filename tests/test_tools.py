import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

class RepositoryTests(unittest.TestCase):
    def test_launcher_resolves_models_and_boards(self):
        run = module('run')
        script, models, env = run.prepare('autonomy', ROOT / 'robot.example.json')
        self.assertEqual(script, ROOT / 'applications/field_tasks.py')
        self.assertEqual(env['ASABE_ESP1_PORT'], '/dev/esp32_01')
        self.assertEqual(env['ASABE_BALL_VERIFY_ENGINE'], str(models / 'highcam_platform.engine'))

    def test_launcher_rejects_path_escape(self):
        with self.assertRaises(ValueError):
            module('run').prepare('../outside.py', ROOT / 'robot.example.json')

    def test_checksum_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.onnx'
            path.write_bytes(b'correct')
            record = {'bytes': 7, 'sha256': hashlib.sha256(b'correct').hexdigest()}
            module('models').verify(path, record)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                module('models').verify(path, record)

    def test_models_manifest_has_only_unique_safe_names(self):
        files = json.loads((ROOT / 'models/manifest.json').read_text())['files']
        self.assertEqual(len(files), len({item['filename'] for item in files}))
        for item in files:
            self.assertEqual(Path(item['filename']).name, item['filename'])
            self.assertRegex(item['sha256'], r'^[a-f0-9]{64}$')

    def test_config_overrides_shell_device(self):
        with patch.dict('os.environ', {'ASABE_ESP1_PORT': '/dev/wrong'}):
            _, _, env = module('run').prepare('dashboard', ROOT / 'robot.example.json')
            self.assertEqual(env['ASABE_ESP1_PORT'], '/dev/esp32_01')

    def test_explicit_engine_override_is_preserved(self):
        with patch.dict('os.environ', {'ASABE_LANE_ENGINE': '/custom/rail.engine'}):
            _, _, env = module('run').prepare('autonomy', ROOT / 'robot.example.json')
            self.assertEqual(env['ASABE_LANE_ENGINE'], '/custom/rail.engine')

    def test_documentation_relative_file_links(self):
        import re
        from urllib.parse import unquote
        for path in ROOT.rglob('*.md'):
            for link in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
                link = link.strip('<>').split('#')[0]
                if not link or '://' in link:
                    continue
                self.assertTrue((path.parent / unquote(link)).exists(), f'{path.relative_to(ROOT)}: {link}')

    def test_five_primary_applications(self):
        runner = module('run')
        for name in ('dashboard', 'identify', 'navigate', 'field', 'lane'):
            target, _, _ = runner.prepare(name, ROOT / 'robot.example.json')
            self.assertEqual(target.parent, ROOT / 'applications')

    def test_verification_dry_run_switches(self):
        import subprocess
        import sys
        for flag, expected in [('on', '1'), ('off', '0')]:
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/run.py'), '--config',
                                     str(ROOT / 'robot.example.json'), '--dry-run',
                                     '--verification', flag, 'field'], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)['environment']['ASABE_ENABLE_BALL_VERIFY'], expected)

    def test_verification_rejected_for_navigation(self):
        import subprocess
        import sys
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/run.py'), '--config',
                                 str(ROOT / 'robot.example.json'), '--dry-run',
                                 '--verification', 'on', 'navigate'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def test_field_plan_includes_verification_only_when_enabled(self):
        import ast
        tree = ast.parse((ROOT / 'applications/field_tasks.py').read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_front_tof_steps')
        constants = {node.id: 1 for node in ast.walk(function) if isinstance(node, ast.Name) and node.id.isupper()}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<field plan>', 'exec'), constants)
        for enabled in (False, True):
            constants['ENABLE_BALL_VERIFY'] = enabled
            plan = constants['build_front_tof_steps']('R', 'L', 'left')
            actions = next(step[2] for step in plan if step[1] == 'placeholder_actions')
            self.assertEqual(any(action[0] == 'verify' for action in actions), enabled)

    def test_device_deployment_commands(self):
        spec = importlib.util.spec_from_file_location('launch_vision', ROOT / 'deployment/launch_vision.py')
        deploy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(deploy)
        for target, backend, suffix in [('pi5', 'ncnn', '.param'), ('pi5-hailo8', 'hailo', '.hef'), ('orin-nano', 'tensorrt', '.engine')]:
            command = deploy.command_for(target, Path('model' + suffix), Path('dataset'), 'highcam_platform')
            self.assertEqual(command[command.index('--backend') + 1], backend)
            self.assertEqual(Path(command[1]), ROOT / 'deployment/tools/evaluate.py')
        with self.assertRaises(ValueError):
            deploy.command_for('pi5-hailo8', Path('wrong.engine'), Path('data'), 'highcam_platform')

    def test_hardware_release_assets(self):
        expected = {
            'Argus_Omnimotus_Master_Multibody.SLDPRT',
            'Assembly_Drawing.jpg',
            'Circuit_Diagram.png',
            'BOM.csv',
            'Pin_Map.md',
            'README.md',
        }
        self.assertEqual({path.name for path in (ROOT / 'hardware').iterdir() if path.is_file()}, expected)
        self.assertGreater((ROOT / 'hardware/Argus_Omnimotus_Master_Multibody.SLDPRT').stat().st_size, 100_000_000)

if __name__ == '__main__':
    unittest.main()
