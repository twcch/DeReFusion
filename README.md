# DeReFusion

## Decomposition–Residual Fusion for Financial Time Series Forecasting

DeReFusion is a hybrid forecasting model that combines a linear **decomposition branch** (DLinear-style seasonal + trend projection) with a non-linear **residual branch** (LSTM → Transformer), fused under reversible instance normalisation (RevIN). It is designed for the noisy, non-stationary, low-sample regime characteristic of financial market data, where pure Transformer models tend to overfit and pure linear models under-fit structural breaks.

This repository also serves as a research harness: it ships a broad suite of baselines (Autoformer, Informer, FEDformer, PatchTST, iTransformer, TimeMixer, TimesNet, Mamba, Koopa, TimeXer, foundation models such as Chronos / TimesFM / Moirai / TiRex / TimeMoE, and more) and a unified task interface for long-/short-term forecasting, imputation, anomaly detection and classification, so that ablations can be run against strong reference implementations on the same data splits and metrics.

## Highlights

- **Hybrid architecture** — linear decomposition base + hybrid LSTM/Transformer residual, combined via additive fusion or a learned gate.
- **RevIN end-to-end** — distribution-shift-robust normalisation/de-normalisation wrapping both branches.
- **Financial-first datasets** — ready-to-use daily OHLCV for `^DJI`, `^GSPC`, `^SOX`, and TSMC (`2330.TW` / `TSM`) from 2016–2025, plus the standard TSLib benchmarks.
- **Ablation- and gate-variant ready** — drop-in models for studying each architectural choice (`woLSTM`, `woTransformer`, `woDy`, gate v1/v2/v3).
- **Unified runner** — a single `run.py` covers six task types; Bayesian hyper-parameter search via `run_bayesian_opt.py`; batch sweeps via `run_batch.py`.
- **Reproducible environment** — pinned `requirements.txt`, Dockerfile, and `docker-compose.yml` for CUDA 12.1 / PyTorch 2.5.1.

## Model Architecture

```text
         ┌──────────────── RevIN (norm) ────────────────┐
         │                                              │
  x_enc ─┤                                              ├─▶ x_norm
         │                                              │
         └──────────────────────────────────────────────┘
                                │
                ┌───────────────┴────────────────┐
                ▼                                ▼
     ┌────────────────────┐            ┌─────────────────────┐
     │  DLinear Branch    │            │  Residual Branch    │
     │  (base)            │            │  (non-linear)       │
     │  Moving-avg decomp │            │  Linear → LSTM →    │
     │  → seasonal linear │            │  Transformer encoder│
     │  → trend linear    │            │  → time / channel   │
     │                    │            │    projection       │
     └────────────────────┘            └─────────────────────┘
                │                                │
                └────────────────┬───────────────┘
                                 ▼
                       ┌──────────────────┐
                       │  Fusion          │
                       │  (additive or    │
                       │   gated v1/v2/v3)│
                       └──────────────────┘
                                 │
                        RevIN (denorm)
                                 │
                                 ▼
                             forecast
```

The default model (`models/derefusion_model/DeReFusion.py`) uses **direct additive fusion** (`base + residual`). Three gated variants live under [`models/derefusion_model/gate_variant/`](models/derefusion_model/gate_variant/):

| Variant | File | Gate design |
| --- | --- | --- |
| v1 — Volatility-aware | [`DeReFusion_gatev1_volatilityaware.py`](models/derefusion_model/gate_variant/DeReFusion_gatev1_volatilityaware.py) | Gate driven by rolling input volatility statistics |
| v2 — Learnable scalar | [`DeReFusion_gatev2_learnable.py`](models/derefusion_model/gate_variant/DeReFusion_gatev2_learnable.py) | Per-channel learnable mixing weight |
| v3 — Input-conditioned | [`DeReFusion_gatev3_inputconditioned.py`](models/derefusion_model/gate_variant/DeReFusion_gatev3_inputconditioned.py) | Bottleneck MLP producing per-sample / per-step gate |

Ablation models (remove one component at a time) are under [`models/derefusion_model/ablation_variant/`](models/derefusion_model/ablation_variant/): `DeReFusion_woLSTM`, `DeReFusion_woTransformer`, `DeReFusion_woDy`.

## Supported Tasks

`run.py --task_name` accepts:

- `long_term_forecast`
- `short_term_forecast`
- `imputation`
- `anomaly_detection`
- `classification`
- `zero_shot_forecast`

Task-specific scripts live in [`scripts/`](scripts/):

```text
scripts/
├── long_term_forecast/       # ETT, ECL, Traffic, Weather, Exchange, ILI, …
├── short_term_forecast/
├── exogenous_forecast/
├── imputation/
├── classification/
└── anomaly_detection/
```

## Datasets

Financial datasets (daily, 2016–2025) are shipped under [`dataset/yfinance/`](dataset/yfinance/):

| Ticker | File |
| --- | --- |
| `^DJI` | `DJI-2016-2025.csv` |
| `^GSPC` | `GSPC-2016-2025.csv` |
| `^SOX` | `SOX-2016-2025.csv` |
| TSMC | `TSMC-2016-2025.csv` |

Regenerate or extend them with [`tools/fetch_yfinance_data.py`](tools/fetch_yfinance_data.py).

Standard TSLib benchmarks (ETT{h1,h2,m1,m2}, ECL, Traffic, Weather, Exchange, ILI, M4, UEA, SMD/MSL/SMAP/SWaT/PSM) are supported via the same `data_provider` and should be placed under `./data/` following the [Time-Series-Library](https://github.com/thuml/Time-Series-Library) layout.

## Supported Models

A non-exhaustive list of baselines available under [`models/`](models/):

- **Transformer family** — Transformer, Informer, Autoformer, FEDformer, Reformer, Pyraformer, Nonstationary Transformer, ETSformer, Crossformer, PatchTST, iTransformer, TimeXer, MultiPatchFormer, Temporal Fusion Transformer
- **MLP / linear** — DLinear, TiDE, LightTS, FreTS, TSMixer, WPMixer, PAttn
- **Convolution / decomposition** — TimesNet, MICN, SCINet, MSGNet, FiLM, Koopa, SegRNN
- **State-space** — Mamba, MambaSimple, MambaSingleLayer, TimeFilter, TimeMixer
- **Foundation models (zero-shot)** — Chronos, Chronos2, TimesFM, Moirai, TiRex, TimeMoE, Sundial, KANAD
- **DeReFusion** — main model plus ablation and gate variants

Custom hybrids: `LSTMAttention`, `TransLSTM`, `RevINTransformer`, `RevINTransLSTM`, `RevTransLSTM-AR`.

## Getting Started

### 1. Clone

```bash
git clone https://github.com/twcch/DyVolFusion.git
cd DyVolFusion
```

### 2. Install PyTorch (CUDA 12.1 example)

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Mamba-SSM (required by the Mamba baselines)

Pick the wheel that matches your CUDA / PyTorch / Python ABI. Example for CUDA 12, PyTorch 2.5, Python 3.11, Linux x86_64:

```bash
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

### 5. Install `uni2ts` without its dependencies (required by the Moirai baseline)

```bash
pip install uni2ts --no-deps
```

### (Optional) Docker

A CUDA 12.1 / PyTorch 2.5.1 image is provided:

```bash
docker compose build dev_tslib
docker compose run --rm dev_tslib
```

## Running Experiments

Training and evaluation are dispatched through [`run.py`](run.py). The canonical entry point is:

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id DeReFusion_ETTh1_96_96 \
  --model DeReFusion \
  --data ETTh1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 128 --n_heads 4 --e_layers 2 --d_layers 1 --d_ff 256 \
  --des 'Exp' --itr 1
```

For dataset- and model-specific reference configurations, see the shell scripts in [`scripts/`](scripts/). Results, checkpoints and visualisations are written to `./results/`, `./checkpoints/`, and `./test_results/` respectively.

### Batch sweeps and Bayesian search

- `python run_batch.py` — run pre-configured batches of experiments across models/datasets.
- `python run_bayesian_opt.py` — hyper-parameter search driven by Bayesian optimisation.

### Inference only

Set `--is_training 0` to reuse an existing checkpoint for evaluation.

## Project Structure

```text
.
├── run.py / run_batch.py / run_bayesian_opt.py   # Entry points
├── exp/                                          # Task experiments (long/short/impute/anom/class/zeroshot)
├── data_provider/                                # Datasets, loaders, factory
├── dataset/yfinance/                             # Financial data (DJI, GSPC, SOX, TSMC)
├── models/
│   ├── derefusion_model/                         # DeReFusion + ablation + gate variants
│   └── tslib/                                    # Baselines (Autoformer, Informer, PatchTST, …)
├── layers/                                       # Reusable blocks (RevIN, Embed, decomp, attention, …)
├── scripts/                                      # Shell scripts per task / dataset
├── tools/                                        # Utilities (figure generation, yfinance fetcher)
├── utils/                                        # Metrics, losses, viz, DTW, time features, …
├── tutorial/                                     # Notebooks / walkthroughs
├── results/ · test_results/ · checkpoints/       # Experiment outputs
├── Dockerfile · docker-compose.yml               # Reproducible environment
└── requirements.txt
```

## Reproducibility

- Random seed is controlled by `--rand_seed` (default `2021`) and is propagated to `random`, `numpy`, and `torch`.
- All baseline implementations track the upstream [Time-Series-Library](https://github.com/thuml/Time-Series-Library); dataset splits and evaluation protocols follow the same conventions.
- Dependencies are pinned in `requirements.txt`; the Docker image builds deterministically on top of `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel`.

## Contributing

This repository is maintained strictly for the author's personal research. **External pull requests are not accepted.** The project is, however, fully open-source under MIT — you are welcome to fork, adapt, and build on it. Issues for genuine bugs are welcome; please understand that support may be best-effort. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Released under the MIT License.

- Copyright © 2026 Chih-Chien Hsieh
- Copyright © 2021 THUML @ Tsinghua University

See [LICENSE](LICENSE) for the full text.

## Acknowledgements

DeReFusion builds on [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) and reuses or re-implements ideas from DLinear, Autoformer, FEDformer, PatchTST, iTransformer, TimeMixer, TimesNet, Mamba, Chronos, TimesFM, Moirai, and many others. Sincere thanks to the authors of those works and to the broader open-source time-series community.
