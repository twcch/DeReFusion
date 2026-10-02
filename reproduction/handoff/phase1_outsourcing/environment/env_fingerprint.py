"""Emit and validate the Phase 1 GPU environment fingerprint."""

from __future__ import annotations

import json
import platform
import subprocess
import sys

import numpy
import pandas
import sklearn
import torch


EXPECTED = {
    "python": "3.11.15",
    "torch": "2.5.1+cu121",
    "torch_cuda": "12.1",
    "numpy": "2.1.2",
    "pandas": "2.3.3",
    "scikit_learn": "1.7.2",
    "cuda_available": True,
}


def main() -> int:
    observed = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "platform": platform.platform(),
        "pip_freeze": subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        ).splitlines(),
    }
    mismatches = {
        key: {"expected": expected, "observed": observed.get(key)}
        for key, expected in EXPECTED.items()
        if observed.get(key) != expected
    }
    print(json.dumps({"observed": observed, "mismatches": mismatches}, indent=2, sort_keys=True))
    return 2 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
