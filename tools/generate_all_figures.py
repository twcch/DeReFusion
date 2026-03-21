"""
One-time script: scan all test_results/<Model>/<setting>/ paths,
find matching results/<Model>/<setting>/ with .npy files,
and generate visualization figures into test_results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.visualization import generate_figures

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
TEST_RESULTS_DIR = os.path.join(ROOT, "test_results")


def main():
    if not os.path.isdir(TEST_RESULTS_DIR):
        print(f"test_results/ not found: {TEST_RESULTS_DIR}")
        return

    total = 0
    success = 0
    skipped = 0

    for model_name in sorted(os.listdir(TEST_RESULTS_DIR)):
        model_test_dir = os.path.join(TEST_RESULTS_DIR, model_name)
        if not os.path.isdir(model_test_dir):
            continue

        for setting_name in sorted(os.listdir(model_test_dir)):
            setting_test_dir = os.path.join(model_test_dir, setting_name)
            if not os.path.isdir(setting_test_dir):
                continue

            # Corresponding results folder with .npy files
            setting_results_dir = os.path.join(RESULTS_DIR, model_name, setting_name)
            pred_path = os.path.join(setting_results_dir, "pred.npy")

            if not os.path.exists(pred_path):
                print(f"[SKIP] No pred.npy: {model_name}/{setting_name}")
                skipped += 1
                continue

            # Check if figures already exist
            expected_figs = [
                "fig_prediction_curves.png",
                "fig_error_analysis.png",
                "fig_error_heatmap.png",
                "fig_pred_true.png",
                "fig_dashboard.png",
            ]
            existing = sum(1 for f in expected_figs if os.path.exists(os.path.join(setting_test_dir, f)))
            if existing >= len(expected_figs):
                print(f"[DONE] Already has figures: {model_name}/{setting_name}")
                skipped += 1
                continue

            total += 1
            print(f"\n{'='*60}")
            print(f"[{total}] {model_name}/{setting_name}")
            print(f"  input:  {setting_results_dir}")
            print(f"  output: {setting_test_dir}")
            try:
                generate_figures(
                    input_path=setting_results_dir,
                    output_path=setting_test_dir,
                )
                success += 1
            except Exception as e:
                print(f"  [ERROR] {e}")

    print(f"\n{'='*60}")
    print(f"Done. Generated: {success}/{total}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
