"""Shape gate for the five frozen Phase 1 models.

Doctrine L1.8: *a check on a frozen configuration reads the frozen
configuration.* This test previously restated the configuration by hand. The
restatement carried ``d_ff=128`` while the frozen protocol sets ``d_ff=2048``,
which makes TimesNet 4,699,969 parameters -- within 0.007% of the TSLib
reference and 15.96x smaller than the model the protocol actually runs. The
test was green throughout, and asserted nothing about production.

Every model configuration value below now comes from a source of truth:

* ``reproduction/configs/phase1_modern_baselines.json`` -- the frozen ``common``
  block, for everything that sets capacity;
* the calibration manifest, for ``pred_len``;
* ``run.py``'s argparse defaults, for the keys the runner does not pass.

Values that are neither capacity-bearing nor declared in the frozen config
(``task_name``, ``freq``, ``num_class``) are plumbing constants and are
transcribed with a note, not silently.
"""

import csv
import importlib
import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG = REPO_ROOT / "reproduction" / "configs" / "phase1_modern_baselines.json"
CALIBRATION_MANIFEST = (
    REPO_ROOT / "reproduction" / "results" / "phase1" / "A_calibration.manifest.csv"
)
RUN_PY = REPO_ROOT / "run.py"

# The frozen model names map to module paths. The mapping itself cannot be read
# from the config, so the test asserts that its keys still equal the frozen
# model list -- a stale mapping would otherwise silently drop a model from the
# gate rather than fail it.
MODULE_BY_MODEL = {
    "revin-DLinear": "models.revin-DLinear",
    "DeReFusion": "models.derefusion.DeReFusion",
    "revin-PatchTST": "models.revin-PatchTST",
    "revin-iTransformer": "models.revin-iTransformer",
    "revin-TimesNet": "models.revin-TimesNet",
}

# Keys the runner passes to run.py, so their values must come from the frozen
# config and nowhere else.
RUNNER_SUPPLIED = (
    "seq_len", "label_len", "enc_in", "dec_in", "c_out", "d_model", "n_heads",
    "e_layers", "d_layers", "d_ff", "moving_avg", "factor", "dropout", "embed",
)

# Keys the runner does *not* pass. Production therefore takes run.py's argparse
# defaults, and this test must take the same defaults rather than restate them.
RUN_PY_SUPPLIED = ("top_k", "num_kernels", "activation")


def _argparse_defaults() -> dict[str, str]:
    """Read run.py's argparse defaults without importing it.

    run.py builds its parser inside ``if __name__ == '__main__'``, so importing
    it would execute training. The source is parsed instead; a default that
    moves is then a test failure rather than a silent divergence.
    """
    pattern = re.compile(r"add_argument\(\s*'--(?P<name>\w+)'[^)]*?default=(?P<value>[^,)]+)")
    defaults: dict[str, str] = {}
    for match in pattern.finditer(RUN_PY.read_text(encoding="utf-8")):
        defaults[match.group("name")] = match.group("value").strip().strip("'\"")
    return defaults


def _frozen_config() -> dict:
    return json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))


def _single_calibration_horizon() -> int:
    with CALIBRATION_MANIFEST.open(newline="", encoding="utf-8") as handle:
        horizons = {int(row["horizon"]) for row in csv.DictReader(handle)}
    if len(horizons) != 1:
        raise AssertionError(
            f"calibration manifest declares horizons {sorted(horizons)}; this gate "
            "assumes exactly one, so it must be extended before the manifest changes"
        )
    return horizons.pop()


class Phase1ModelShapeTest(unittest.TestCase):
    def test_module_map_covers_the_frozen_model_list(self):
        frozen = set(_frozen_config()["available_models"])
        self.assertEqual(
            set(MODULE_BY_MODEL), frozen,
            "the module map and the frozen available_models list have diverged; "
            "one of them is stale and the gate is not covering what it claims",
        )

    def test_runner_supplied_keys_are_all_in_the_frozen_config(self):
        common = _frozen_config()["common"]
        for key in RUNNER_SUPPLIED:
            with self.subTest(key=key):
                self.assertIn(key, common)

    def test_run_py_defaults_still_match_what_the_models_are_tested_with(self):
        """Guard the channel the runner does not cover.

        The runner passes no ``--top_k``, ``--num_kernels`` or ``--activation``,
        so production silently uses run.py's defaults. If a default moves, the
        frozen protocol changes without any recorded event; this assertion is
        what makes that loud.
        """
        defaults = _argparse_defaults()
        expected = {"top_k": "5", "num_kernels": "6", "activation": "gelu"}
        for key in RUN_PY_SUPPLIED:
            with self.subTest(key=key):
                self.assertIn(key, defaults, f"run.py no longer defines --{key}")
                self.assertEqual(
                    defaults[key], expected[key],
                    f"run.py's --{key} default moved to {defaults[key]!r}; the frozen "
                    "protocol did not change, so production is now running a "
                    "configuration no receipt records",
                )

    def test_all_frozen_models_accept_ms_protocol_shape(self):
        config = _frozen_config()
        common = config["common"]
        defaults = _argparse_defaults()

        values = {key: common[key] for key in RUNNER_SUPPLIED}
        values["pred_len"] = _single_calibration_horizon()
        values["top_k"] = int(defaults["top_k"])
        values["num_kernels"] = int(defaults["num_kernels"])
        values["activation"] = defaults["activation"]
        # Plumbing constants, not declared in the frozen config and not
        # capacity-bearing. `task_name` and `freq` mirror
        # `run_phase1_calibration.build_command`; `num_class` is unused by these
        # five forecast models and is present only because some model modules
        # read it during construction.
        values.update(
            task_name="long_term_forecast",
            freq="b",
            num_class=2,
        )
        configs = SimpleNamespace(**values)

        self.assertEqual(configs.d_ff, 2048, "the frozen d_ff moved; L1.2-L1.4 apply")

        x_enc = torch.randn(2, configs.seq_len, configs.enc_in)
        for model_name, module_name in MODULE_BY_MODEL.items():
            with self.subTest(model=model_name):
                model = importlib.import_module(module_name).Model(configs).eval()
                with torch.no_grad():
                    output = model(x_enc, None, None, None)
                # These five modules emit `enc_in` channels; the experiment
                # wrapper slices to the target (`f_dim = -1` for features=MS).
                # That is why the shape is asserted against enc_in and not
                # against the frozen c_out=1.
                self.assertEqual(
                    tuple(output.shape), (2, configs.pred_len, configs.enc_in)
                )
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
