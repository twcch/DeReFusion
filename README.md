# DeReFusion

**Official code for the paper**
**_DeReFusion: A Decomposition–Residual Fusion Forecaster for Non-Stationary Financial Time Series_**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Built on Time-Series-Library](https://img.shields.io/badge/Built%20on-Time--Series--Library-8A2BE2.svg)](https://github.com/thuml/Time-Series-Library)

This repository contains everything needed to reproduce the experiments in the paper: the proposed **DeReFusion** model with its **ablation** and **fusion-gate** variants, the **RevIN-wrapped baseline models** and **pretrained foundation models** they are compared against, the **daily OHLC stock/index/crypto/forex datasets**, and the **shared training and evaluation pipeline** that makes the comparison fair.

---

## Table of Contents

- [Overview](#overview)
- [What makes the benchmark _fair_](#what-makes-the-benchmark-fair)
- [DeReFusion — the architecture](#derefusion--the-architecture)
- [Benchmark models](#benchmark-models)
- [Datasets](#datasets)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Reproducing the benchmark](#reproducing-the-benchmark)
- [Evaluation and outputs](#evaluation-and-outputs)
- [Adding a model](#adding-a-model)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contributing](#contributing)

---

## Overview

Daily financial prices — equities, indices, crypto, and forex alike — are strongly **non-stationary**: their mean and variance drift over time and the raw `Close` series carries a unit root (see [`utils/ADFtest.py`](./utils/ADFtest.py)). One way to attack this is to keep stacking capacity — deeper Transformers, recurrent–attention hybrids, learned fusion gates. This paper asks the opposite question:

> **On non-stationary financial series, how far does a deliberately simple recipe go — a linear decomposition base, a learned residual, and plain additive fusion — and does anything more elaborate actually buy accuracy?**

To answer it, the repo provides two things:

1. A **fair benchmark** — a single data pipeline, normalization scheme, training protocol, and metric set shared by every model, so differences in results reflect the **backbone**, not the harness.
2. **DeReFusion plus its probes** — the proposed model splits the forecast into a **DLinear base branch** and a **hybrid (LSTM → Transformer) residual branch**, combined by **direct additive fusion**, all wrapped in RevIN. It ships with **ablation variants** that remove one branch or one residual component at a time, and **fusion-gate variants** that replace the plain `base + residual` with a learned gate. Sliding DeReFusion across these two axes isolates *which branches matter* and *whether a smarter fusion is worth its parameters*.

The repository is a focused fork of [thuml/Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library): it keeps TSLib's experiment harness and reuses its layer library, but trims the model set down to exactly what the paper studies.

---

## What makes the benchmark _fair_

Every model in this repository is evaluated under an **identical protocol**, so accuracy differences are attributable to the architecture and not to incidental advantages.

| Dimension | Shared setting |
|---|---|
| **Data loader** | `Dataset_Custom` ([`data_provider/data_loader.py`](./data_provider/data_loader.py)) for all models |
| **Splits** | Chronological **70% / 10% / 20%** train / validation / test; validation and test windows are extended back by `seq_len` for lookback context |
| **Scaling** | `StandardScaler` **fit on the training split only**, then applied to all splits |
| **Normalization** | **RevIN** (Reversible Instance Normalization) applied to _every_ trained model — DeReFusion and all RevIN baselines — so the comparison isolates backbone complexity rather than the normalization scheme |
| **Inputs / target** | `--features MS`: all four OHLC columns as input, single `Close` target (`--target Close`) |
| **Training** | Adam optimizer, MSE loss, early stopping on validation loss (`--patience`), learning-rate schedule (`--lradj`) |
| **Seeds** | Controlled by `--rand_seed` (default `2021`); the seed is embedded in each run's `setting` string so multi-seed runs are tracked separately |
| **Metrics** | MAE, MSE, RMSE, MAPE, MSPE, R² (optionally DTW), plus parameter count, train time, inference ms/sample, and GPU peak memory |

> **Reproducibility note.** `run.py` seeds Python `random`, NumPy, and PyTorch (`torch.manual_seed`). It does **not** set `torch.backends.cudnn.deterministic`, so results on CUDA are reproducible up to cuDNN non-determinism. Run several seeds (e.g. `2020`–`2024`) and report mean ± std, as the paper does.

---

## DeReFusion — the architecture

**File:** [`models/derefusion/DeReFusion.py`](./models/derefusion/DeReFusion.py) · **`--model DeReFusion`**

DeReFusion is a **decomposition–residual dual-branch** forecaster. A linear **DLinear base branch** captures the trend/seasonal structure of the lookback window; a **hybrid residual branch** (per-feature projection → LSTM → Transformer encoder → projection) models what the linear base leaves behind. The two are combined by **direct additive fusion**, and the whole forecast is produced in RevIN-normalized space and denormalized back to the original price scale.

**Forward pass** (`B` = batch, `D` = `d_model`, `C` = `enc_in` = 4):

```
x_enc [B, seq_len, 4]                              # OHLC input window
  │
  ├─ RevIN(norm) ──► x_norm [B, seq_len, 4]        # per-instance normalize (stats reused for denorm)
  │
  ├─ DLinear base branch ───────────────────────────────────────────
  │     seasonal, trend = x_norm − MA(x_norm),  MA(x_norm)   # moving-avg kernel = --moving_avg
  │     base = Linearₛ(seq_len→pred_len)·seasonalᵀ
  │              + Linearₜ(seq_len→pred_len)·trendᵀ          ──► base [B, pred_len, 4]
  │
  ├─ Hybrid residual branch ────────────────────────────────────────
  │     Linear(4→D) → LSTM(D→D) → TransformerEncoder(D)
  │       → Linear(seq_len→pred_len) over time
  │       → Linear(D→4) over channels                        ──► residual [B, pred_len, 4]
  │
  ├─ fused = base + residual                        # direct additive fusion
  └─ RevIN(denorm) ──────► forecast [B, pred_len, 4]   (target = Close column)
```

The two design choices that define the model — and that the variants probe — are:

- **Decomposition–residual split:** the linear base and the learned residual are produced by separate branches over the *same* normalized window, rather than by one monolithic backbone.
- **Direct additive fusion:** the branches are combined as plain `base + residual` — a parameter-free fusion. The gate variants below replace it with a learned gate to test whether that simplicity costs accuracy.

RevIN is the standalone Reversible Instance Normalization layer at [`layers/RevIN.py`](./layers/RevIN.py), shared with every RevIN baseline.

### Ablation variants

**Folder:** [`models/derefusion/ablation_variant/`](./models/derefusion/ablation_variant/). Each is a valid `--model` name (filename without `.py`).

| `--model` | Removes / changes | Resulting behavior |
|---|---|---|
| `DeReFusion-woDy` | Hybrid residual branch + fusion | DLinear base only — tests whether the residual branch contributes at all |
| `DeReFusion-woLSTM` | LSTM from the residual branch | `base +` Transformer-only residual — isolates the LSTM's contribution |
| `DeReFusion-woTransformer` | Transformer from the residual branch | `base +` LSTM-only residual — isolates the Transformer's contribution |

### Fusion-gate variants

**Folder:** [`models/derefusion/gate_variant/`](./models/derefusion/gate_variant/). These keep both branches but replace the additive `base + residual` with a **gated fusion** `(1 − gate) · base + gate · residual`, asking whether a learned gate beats plain addition on small, non-stationary financial datasets.

| `--model` | Gate | Mechanism |
|---|---|---|
| `DeReFusion-gatev1-volatilityaware` | **VolatilityAwareGate** | Gate driven by the **raw** input's per-channel std (computed before RevIN); a single volatility scalar must set per-timestep, per-channel weights |
| `DeReFusion-gatev2-learnable` | **LearnableGate** | Per-channel learnable scalar `gate = sigmoid(α)`; **static** across samples and timesteps |
| `DeReFusion-gatev3-inputconditioned` | **InputConditionedGate** | Sample-adaptive bottleneck net (`seq_len → --gate_bottleneck → pred_len`) producing per-sample / per-channel / per-timestep weights |

---

## Benchmark models

DeReFusion is compared against two families of baselines, all run through the same pipeline.

### (a) RevIN-wrapped baselines — `models/`

Ten TSLib backbones, each wrapped with the shared [`layers/RevIN.py`](./layers/RevIN.py) normalize/denormalize so they sit on a clean **complexity ladder** alongside DeReFusion. Names are case-sensitive and the hyphen is part of the model name (e.g. `--model revin-DLinear`).

| Complexity tier | `--model` | Backbone |
|---|---|---|
| **Linear / MLP** | `revin-DLinear` | Series decomposition + per-component linear maps |
| | `revin-LightTS` | Lightweight MLP with interval/continuous sampling |
| **CNN** | `revin-TimesNet` | 2D temporal blocks over multi-period reshapes |
| **Transformer** | `revin-Informer` | ProbSparse attention + distilling |
| | `revin-Reformer` | LSH-attention efficient Transformer |
| | `revin-iTransformer` | Inverted Transformer (attention across variates) |
| | `revin-PatchTST` | Patching + channel-independent Transformer |
| | `revin-Autoformer` | Decomposition + AutoCorrelation attention |
| | `revin-FEDformer` | Frequency-enhanced decomposition Transformer |
| | `revin-ETSformer` | Exponential-smoothing attention |

### (b) Pretrained foundation models — `models/`

Run **zero-shot** (no training) through the `zero_shot_forecast` task with `--is_training 0`.

| `--model` | File |
|---|---|
| `Chronos` | [`models/Chronos.py`](./models/Chronos.py) |
| `Moirai` | [`models/Moirai.py`](./models/Moirai.py) |
| `TimesFM` | [`models/TimesFM.py`](./models/TimesFM.py) |

---

## Datasets

Ten daily **OHLC** series under [`dataset/`](./dataset/), spanning **2016-01-01 → 2025-12-31**. All files share the header `date,Open,High,Low,Close` (`date` is `YYYY-MM-DD`; values are split/adjusted floats). **There is no Volume column.** Row counts differ because each market follows its own trading calendar — crypto trades on weekends, equities/indices/forex do not.

| Ticker | Instrument | Type | Rows |
|---|---|---|---|
| [BABA](./dataset/BABA-2016-2025.csv) | Alibaba | Stock | 2514 |
| [NVO](./dataset/NVO-2016-2025.csv) | Novo Nordisk | Stock | 2514 |
| [TM](./dataset/TM-2016-2025.csv) | Toyota Motor | Stock | 2514 |
| [GSPC](./dataset/GSPC-2016-2025.csv) | S&P 500 | Index | 2514 |
| [DJI](./dataset/DJI-2016-2025.csv) | Dow Jones Industrial Average | Index | 2514 |
| [SOX](./dataset/SOX-2016-2025.csv) | PHLX Semiconductor | Index | 2514 |
| [EURUSD](./dataset/EURUSD-2016-2025.csv) | Euro / US Dollar | Forex | 2602 |
| [USDJPY](./dataset/USDJPY-2016-2025.csv) | US Dollar / Japanese Yen | Forex | 2602 |
| [BTCUSD](./dataset/BTCUSD-2016-2025.csv) | Bitcoin | Crypto | 3653 |
| [ETHUSD](./dataset/ETHUSD-2016-2025.csv) | Ethereum | Crypto | 2975 |

**Non-stationarity.** [`utils/ADFtest.py`](./utils/ADFtest.py) runs the Augmented Dickey–Fuller test (via `statsmodels` and `arch`) to quantify the unit-root behavior of each series — the empirical motivation for applying RevIN to every trained model in the benchmark.

**Loading convention.** CSVs are read through the `custom` provider with multivariate-input / single-target settings:

| Flag | Value | Meaning |
|---|---|---|
| `--data` | `custom` | use `Dataset_Custom` |
| `--features` | `MS` | all OHLC columns in, single target out |
| `--target` | `Close` | predict `Close` (moved to the last column internally) |
| `--freq` | `b` | business-day frequency for the temporal embedding |
| `--enc_in` / `--dec_in` / `--c_out` | `4` / `4` / `1` | 4 OHLC inputs, 1 target output |

> `run.py` inherits TSLib's defaults (`--data ETTh1`, `--features M`, `--target OT`, `--freq h`), so the financial configuration must be passed explicitly, as shown below.

---

## Repository structure

```text
DeReFusion/
├── run.py                              # Single entry point — CLI, seeding, device & task dispatch
├── run_batch_long_term_forecast.py     # Parallel batch sweep runner (DeReFusion + RevIN baselines)
├── run_batch_zero_shot_forecast.py     # Batch sweep runner (foundation models, zero-shot)
├── models/
│   ├── revin-Autoformer.py             # 10 RevIN-wrapped baselines (revin-*.py)
│   ├── revin-DLinear.py
│   ├── revin-ETSformer.py
│   ├── revin-FEDformer.py
│   ├── revin-Informer.py
│   ├── revin-iTransformer.py
│   ├── revin-LightTS.py
│   ├── revin-PatchTST.py
│   ├── revin-Reformer.py
│   ├── revin-TimesNet.py
│   ├── Chronos.py · Moirai.py · TimesFM.py   # pretrained foundation models (zero-shot)
│   └── derefusion/
│       ├── DeReFusion.py                # proposed model
│       ├── ablation_variant/            # 3 ablation variants (woDy / woLSTM / woTransformer)
│       └── gate_variant/                # 3 fusion-gate variants (gatev1 / gatev2 / gatev3)
├── exp/                                # Task pipelines + Exp_Basic model registry
│   ├── exp_basic.py                    # auto-discovers models under models/ (lazy import)
│   ├── exp_long_term_forecasting.py    # the pipeline used by the paper
│   └── exp_zero_shot_forecasting.py    # foundation-model zero-shot pipeline
├── layers/                             # Reusable layers (RevIN, attention, embeddings, …)
├── data_provider/                      # data_factory.py, data_loader.py (Dataset_Custom)
├── dataset/                            # 10 OHLC CSVs (2016–2025)
├── utils/                              # metrics.py, visualization.py, ADFtest.py, tools.py
├── scripts/                            # Inherited TSLib reproduction scripts (standard benchmarks — see note)
├── requirements/                       # Ordered install files reqs_1..4.txt
├── Dockerfile                          # CUDA 12.1 / PyTorch 2.5.1 image
└── docker-compose.yml                  # dev service with GPU passthrough
```

> **Note on `scripts/`.** These shell scripts are inherited from upstream TSLib and target the _standard_ academic benchmarks (ETT, ECL, Weather, M4, anomaly/classification datasets). They are **not** the financial experiments of this paper — the benchmark is run through `run.py` and the batch runners as documented below.

---

## Installation

**Recommended Python: 3.11.** Dependencies are split into four ordered files in [`requirements/`](./requirements/).

### Minimal install (DeReFusion + RevIN baselines)

The proposed model and the RevIN baselines depend only on PyTorch and the standard scientific/attention stack — **not** on the Mamba or foundation-model libraries.

```bash
# 1. PyTorch (CUDA 12.1 build — torch==2.5.1)
pip install -r requirements/reqs_1.txt

# 2. Core scientific + attention stack
#    (numpy, scipy, scikit-learn, pandas, matplotlib, einops, reformer-pytorch,
#     sktime, sympy, PyWavelets, tqdm, …)
pip install -r requirements/reqs_2.txt
```

> **CPU / Apple Silicon:** `requirements/reqs_1.txt` pins the CUDA 12.1 wheel. If you do not have an NVIDIA GPU, install the matching CPU or MPS build of `torch==2.5.1` from [pytorch.org](https://pytorch.org/get-started/locally/) instead of step 1, then run step 2.

### Foundation-model + Mamba extras (only for the zero-shot baselines)

`requirements/reqs_2.txt` also pulls the foundation-model stack (`transformers`, `chronos-forecasting`, `timesfm`, `tirex-ts`, `gluonts`, `lightning`, `jax`, …) needed to run `Chronos` / `Moirai` / `TimesFM`. The pinned Mamba state-space wheel and `uni2ts` live in the remaining files and are **not** imported by DeReFusion or the RevIN baselines, so you can skip them unless you need those backends.

```bash
pip install -r requirements/reqs_3.txt          # mamba_ssm — Linux x86_64 + CUDA 12 + Python 3.11 + torch 2.5 ONLY
pip install -r requirements/reqs_4.txt           # uni2ts and friends
```

> `reqs_3.txt` pins a `mamba_ssm` wheel built for `cu12 / torch2.5 / cp311 / linux_x86_64` only; it will not install on macOS, Windows, ARM, or other Python versions.

### Docker (optional)

A CUDA environment is provided for full-stack reproduction:

```bash
# The Dockerfile installs from a single consolidated requirements.txt:
cat requirements/reqs_*.txt > requirements.txt
docker compose up -d --build
docker compose exec dev_tslib bash
```

The image builds on `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel`, runs with GPU passthrough (`NVIDIA_VISIBLE_DEVICES=all`), `shm_size: 8gb`, and a `/workspace` volume. It targets the _complete_ stack including the optional Mamba / foundation dependencies.

---

## Reproducing the benchmark

All experiments run through [`run.py`](./run.py), which seeds the RNGs, selects the device, and dispatches to the appropriate task pipeline.

### Train and evaluate the proposed model

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id GSPC_96_24 \
  --model DeReFusion \
  --data custom \
  --root_path ./dataset/ \
  --data_path GSPC-2016-2025.csv \
  --features MS --target Close --freq b \
  --seq_len 96 --label_len 48 --pred_len 24 \
  --enc_in 4 --dec_in 4 --c_out 1 \
  --d_model 32 --moving_avg 25 \
  --train_epochs 30 --batch_size 32 --learning_rate 0.0001 \
  --patience 5 --lradj cosine \
  --rand_seed 2021
```

### Run a baseline, ablation, or fusion-gate variant

Swap `--model` for any name from the [benchmark set](#benchmark-models), the [ablation table](#ablation-variants), or the [gate table](#fusion-gate-variants); every other flag stays the same:

```bash
python run.py --model revin-DLinear                     ... # a linear baseline
python run.py --model revin-PatchTST                    ... # a Transformer baseline
python run.py --model DeReFusion-woDy                   ... # an ablation (base branch only)
python run.py --model DeReFusion-gatev3-inputconditioned ... # a fusion-gate variant
```

### Zero-shot foundation models

Run a pretrained foundation model with no training (`--is_training 0`, `--task_name zero_shot_forecast`):

```bash
python run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --model_id BTCUSD_zeroshot \
  --model TimesFM \
  --data custom \
  --root_path ./dataset/ \
  --data_path BTCUSD-2016-2025.csv \
  --features MS --target Close --freq b \
  --seq_len 96 --label_len 48 --pred_len 7
```

> Swap `--model TimesFM` for `Chronos` or `Moirai` to try the other foundation models.

### Device selection

- **CUDA** is auto-detected and used by default.
- **Apple Silicon:** add `--gpu_type mps`.
- **CPU:** add `--no_use_gpu`.

### Full sweep

The paper reports results over **all 10 instruments × multiple forecast horizons × several seeds**. The two batch runners automate this:

```bash
# DeReFusion + RevIN baselines (trained); parallel, MAX_PARALLEL=6, seeds 2020–2024
python run_batch_long_term_forecast.py

# Foundation models (zero-shot): TimesFM / Chronos / Moirai
python run_batch_zero_shot_forecast.py
```

Both runners shell out to `run.py`, key each completed experiment by an MD5 of its command, record progress in `run_batch_progress.log` and failures in `run_batch_failed.log`, write per-experiment stdout/stderr into `run_batch_logs/`, and **resume** where they left off. The dataset list, horizon grid (`pred_len`), seed list, and the active `--model` set are edited at the top of each script — comment blocks there hold the DeReFusion presets (the 2016–2025 instruments and horizons `[1, 7, 12, 24, 36]`).

---

## Evaluation and outputs

The long-term-forecasting pipeline ([`exp/exp_long_term_forecasting.py`](./exp/exp_long_term_forecasting.py)) reports an extended set of metrics and diagnostics.

**Metrics** ([`utils/metrics.py`](./utils/metrics.py)): `MAE, MSE, RMSE, MAPE, MSPE, R²` — and optionally **DTW** with `--use_dtw` (off by default; time-consuming).

**Diagnostics** (printed and appended to the result log):
- **GPU peak memory** — `gpu_mem_peak_mb` via `torch.cuda.max_memory_allocated` (CUDA only).
- **Training wall time** and **inference speed** — `inference_speed_ms` (ms/sample, guarded by `torch.cuda.synchronize()`).
- **Parameter counts** — total and trainable.

**Output layout** (relative to the working directory; all are git-ignored):

| Path | Contents |
|---|---|
| `results/<setting>/` | `pred.npy`, `true.npy`, `metrics.npy` (the 6 core metrics) |
| `test_results/<setting>/` | auto-generated figures + per-window preview PDFs |
| `checkpoints/<setting>/checkpoint.pth` | best model by validation loss |
| `result_long_term_forecast.txt` | appended one-line summary per run (all metrics + diagnostics) |

**Publication figures** ([`utils/visualization.py`](./utils/visualization.py), 300 DPI, serif/journal style, PNG):

| File | Content |
|---|---|
| `fig_prediction_curves.png` | Sample ground-truth vs. prediction windows with error band |
| `fig_error_analysis.png` | MSE-per-horizon-step bars + error-distribution histogram |
| `fig_metrics_radar.png` | Radar over MAE / MSE / RMSE / MAPE / MSPE / R² |
| `fig_error_heatmap.png` | Absolute-error heatmap (sample × horizon) |
| `fig_pred_true.png` | Continuous ground-truth vs. prediction curve |
| `fig_dashboard.png` | 4-panel summary |

Figures can be regenerated standalone from saved arrays:

```bash
python -m utils.visualization --input results/<setting>/ --output test_results/<setting>/
```

---

## Adding a model

The registry ([`exp/exp_basic.py`](./exp/exp_basic.py)) **auto-discovers** models: drop a `.py` file anywhere under [`models/`](./models/) that defines a class named `Model`, and its filename (without `.py`) becomes the `--model` string — no manual registration, and the module is lazily imported only when selected. This is exactly how the RevIN baselines (`revin-*.py`) and the DeReFusion family are wired in. An unknown `--model` name raises a `ValueError` listing the discovered models.

<!-- ---

## Citation

If you use this code or build on the benchmark, please cite the paper:

```bibtex
@article{hsieh2026derefusion,
  title   = {DeReFusion: A Decomposition--Residual Fusion Forecaster for
             Non-Stationary Financial Time Series},
  author  = {Hsieh, Chih-Chien},
  year    = {2026}
}
```

> Bibliographic details (journal, volume, DOI) will be completed upon publication. -->

---

## Acknowledgements

This project is a focused fork of the [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) by THUML @ Tsinghua University. It reuses TSLib's experiment harness and layer library, and adds the financial datasets, the RevIN-wrapped baselines, the foundation-model integrations, and the DeReFusion model family studied in the paper.

---

## License

Released under the **MIT License** — see [`LICENSE`](./LICENSE).

- Copyright © 2026 Chih-Chien Hsieh
- Copyright © 2021 THUML @ Tsinghua University (Time-Series-Library)

---

## Contributing

This is a **personal research repository** accompanying a publication, so external pull requests are not accepted (see [`CONTRIBUTING.md`](./CONTRIBUTING.md)). It is open source under MIT and **fork-friendly** — you are welcome to fork it, adapt it, and build on the benchmark. Bug reports and questions can be raised as Issues.
