# Dependency groups

Use Python 3.11. The project no longer presents the foundation-model stack as a
minimal dependency.

```bash
# CUDA 12.1 PyTorch; install a matching CPU/MPS wheel instead when appropriate.
pip install -r requirements/torch-cu121.txt

# DeReFusion, RevIN baselines, analysis and plotting.
pip install -r requirements/core.txt

# Optional: Chronos, Moirai, TimesFM and related zero-shot models.
pip install -r requirements/foundation.txt

# Optional and platform-specific: Linux x86_64, Python 3.11, CUDA 12, torch 2.5.
pip install -r requirements/mamba-linux-cu12.txt
```

The pinned versions record the audited environment; they are not a guarantee
that every optional foundation model can coexist on every platform. Create a
separate environment for optional backends when their transitive requirements
conflict.
