import ast
import importlib.machinery
from pathlib import Path
import unittest
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
ROOT = Path(__file__).resolve().parents[1]
class PackageNamespaceTest(unittest.TestCase):
    def test_current_distribution(self):
        with (ROOT / 'pyproject.toml').open('rb') as handle:
            config = tomllib.load(handle)
        self.assertEqual(config['project']['name'], 'phytoreg')
        self.assertEqual(config['project']['scripts'], {
            'phytoreg': 'phytoreg.cli:main', 'phytoreg-inspect': 'phytoreg.cli:inspect_main'})
        self.assertEqual(config['tool']['setuptools']['packages']['find']['include'], ['phytoreg', 'phytoreg.*'])
        self.assertIsNotNone(importlib.machinery.PathFinder.find_spec('phytoreg', [str(ROOT / 'src')]))
    def test_runtime_has_no_historical_imports(self):
        for path in (ROOT / 'src/phytoreg').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                           else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                for module in modules:
                    self.assertFalse(module.startswith(('plantreglm', 'n2_runtime', 'track_query')), str(path))
if __name__ == '__main__': unittest.main()
