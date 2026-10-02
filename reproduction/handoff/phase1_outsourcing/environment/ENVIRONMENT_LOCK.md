# GPU environment lock

Hard preflight requirements:

| Component | Required |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.5.1+cu121 |
| PyTorch CUDA runtime | 12.1 |
| NumPy | 2.1.2 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| CUDA available | true |
| Base image | pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel |

The first eligible remote machine must record the immutable container digest,
GPU model, driver, cuDNN version, OS and installed package inventory. Those
observed values become the full environment fingerprint for later workers.
Do not guess or pre-fill the cuDNN patch version from the CPU host.
