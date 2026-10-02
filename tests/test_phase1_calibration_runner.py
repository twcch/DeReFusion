import unittest

from reproduction.batches.run_phase1_calibration import (
    CONFIG_PATH,
    MANIFEST_PATH,
    REGISTRY_PATH,
    build_command,
    load_csv,
    run_id,
)
import json


class Phase1CalibrationRunnerTest(unittest.TestCase):
    def test_frozen_grid_has_unique_complete_commands(self):
        config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        rows = load_csv(MANIFEST_PATH)
        registry = {row['asset']: row for row in load_csv(REGISTRY_PATH)}
        self.assertEqual(len(rows), 15)
        identifiers = set()
        for index, row in enumerate(rows, start=1):
            identifier = run_id(index, row)
            self.assertNotIn(identifier, identifiers)
            identifiers.add(identifier)
            command, _, _ = build_command(row, registry, config, attempt=1, device='cpu')
            joined = ' '.join(command)
            self.assertIn('--deterministic', command)
            self.assertIn('--no_use_gpu', command)
            self.assertIn('--result_log reproduction/logs/phase1/calibration_result_log.txt', joined)
            self.assertIn(f"--model {row['model']}", joined)
            self.assertIn(f"--rand_seed {row['seed']}", joined)

    def test_resource_guard_defaults_are_explicit(self):
        source = (CONFIG_PATH.parents[1] / 'batches' / 'run_phase1_calibration.py').read_text(encoding='utf-8')
        self.assertIn('default=300', source)
        self.assertIn('subprocess.TimeoutExpired', source)
        self.assertIn('--exclude-model', source)
        self.assertIn('resource_blocked_models()', source)
        self.assertIn('--retry-blocked-model', source)


if __name__ == '__main__':
    unittest.main()
