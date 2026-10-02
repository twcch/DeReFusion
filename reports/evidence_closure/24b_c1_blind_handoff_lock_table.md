# 24b · C1 blind-handoff lock-table excerpt

**Purpose:** this is a deliberately minimal, result-free handoff excerpt for a newly appointed
independent C1 auditor. It contains only the locked cohort identity and file-integrity data needed
to begin the first stage of `24a_c1_blind_recomputation_spec.md`.

**It does not supersede** the C1 preregistration or the blind-recomputation specification. It
contains no predictor values, interactions, correlations, verdicts, or analysis outputs. The
formal blind auditor must read `24a` for the prescribed order of operations, but must not need to
open `23_c1_preregistration.md` before its independent numerical recomputation is complete.

**Source fidelity:** mechanically transcribed from the locked cohort table in
`23_c1_preregistration.md` §8 at commit `587fbf47179f06ead1c62536125e71501317f7df`. At handoff,
the executor must provide the original bytes of every listed file and a new per-file SHA-256
manifest. Any mismatch is a stop condition under `24a`.

| # | Required filename | Rows | First date | Last date | Locked SHA-256 |
|---:|---|---:|---|---|---|
| 1 | `AAPL-2016-2025.csv` | 2514 | 2016-01-04 | 2025-12-31 | `2d0ec4a90463b9a6bae915a3038d871c061b3d18d813d6572d50d75f4b185916` |
| 2 | `MSFT-2016-2025.csv` | 2514 | 2016-01-04 | 2025-12-31 | `48139f47b21a71466a60f7db045a20c85c3a75dcaba7afc20d175646f8b0d931` |
| 3 | `AMZN-2016-2025.csv` | 2514 | 2016-01-04 | 2025-12-31 | `28f348100dc6d056b7bd70aa472ff29923d3776b3eaf9a1385254e08b01e6771` |
| 4 | `META-2016-2025.csv` | 2514 | 2016-01-04 | 2025-12-31 | `fe0effbe8a541895f60e19c664e7f06791408fd46de31db0396f61b2590fa19d` |
| 5 | `TSLA-2016-2025.csv` | 2514 | 2016-01-04 | 2025-12-31 | `4993de1e6a0997785a6c40f5397374dcbe9254b5f5bd3f0633edf92c951e5b9d` |
| 6 | `JPM-2016-2025.csv` | 2514 | 2016-01-04 | 2025-12-31 | `450e3fec5302e7464e1a773e18a4d7f0132b58a926cac014f0e09e2083b1dc5e` |
| 7 | `XOM-2016-2025.csv` | 2514 | 2016-01-04 | 2025-12-31 | `74ecd45a77d1edb9692cadc61b853a381de1b028f50a290df4057f37f9b9c928` |
| 8 | `WMT-2016-2025.csv` | 2514 | 2016-01-04 | 2025-12-31 | `927d268a64b42742e6ae0b29c60a178beadf98506c0e6e1bc778c2e08f885481` |
| 9 | `N225-2016-2025.csv` | 2444 | 2016-01-04 | 2025-12-30 | `7250ac1596365ef894b61bb257789e23fe56a386a90e378901ce651fce4924a3` |
| 10 | `GDAXI-2016-2025.csv` | 2537 | 2016-01-04 | 2025-12-30 | `6c8cc51c65fa002dc3129fd12b2278c05ae8b9a3e1d33ce3e3bcb076f57dbee5` |
| 11 | `HSI-2016-2025.csv` | 2459 | 2016-01-04 | 2025-12-31 | `2a04256095a96d541142a426d289c5f173946b287ceefbd0ee2887c10cc25cec` |
| 12 | `FTSE-2016-2025.csv` | 2525 | 2016-01-04 | 2025-12-31 | `7c69f5dd2b1d6e8f78c228cd607646c1289a688e83e3c6eeef5735e2f276d13e` |
| 13 | `RUT-2016-2025.csv` | 2514 | 2016-01-04 | 2025-12-31 | `27f21f002ef041815e4a1db4acb8c6a0b2492832b43dfd2e79e0ad016bf1f718` |
| 14 | `GBPUSD-2016-2025.csv` | 2602 | 2016-01-01 | 2025-12-31 | `90e00d04d481138da937e5316ac8274728d622b30a117e755e33e9057c833c7d` |
| 15 | `AUDUSD-2016-2025.csv` | 2602 | 2016-01-01 | 2025-12-31 | `cd12448f0624128676d89c3bc7c0d6a12fc34a8b37309e689834804fa15d2a9a` |
| 16 | `USDCAD-2016-2025.csv` | 2602 | 2016-01-01 | 2025-12-31 | `4f346a1849e213ec83446c6d7cdcd450e4e13f15532ce9787cb26a05e09727ee` |
| 17 | `GOLD-2016-2025.csv` | 2513 | 2016-01-04 | 2025-12-31 | `76e42c67bcf264a2ce36a12e162a5f61f8807bde3954823ed074e23b77f8f4e5` |
| 18 | `WTI-2016-2025.csv` | 2514 | 2016-01-04 | 2025-12-31 | `0b7361bb11333d1f6fb889429d8e4869789f260eb6d5c457c1ecec6b936527f1` |
| 19 | `GLD-2016-2025.csv` | 2514 | 2016-01-04 | 2025-12-31 | `0b77dfa686fce94a851cf19ad1f0f23a45028a85423a5a9f218594b12f69b231` |
| 20 | `TLT-2016-2025.csv` | 2514 | 2016-01-04 | 2025-12-31 | `4da4b28f0adc0b50a763a4eba39f6e1f0893df85b9f3bac2769576cfb7671da1` |

## Handoff gate for the formal auditor

The first-stage sealed package must contain this excerpt, `24a`, the 20 original CSVs, every raw
`pred.npy` and `true.npy`, `per_run_manifest.csv`, the two frozen analysis scripts, a full frozen
commit hash, and a SHA-256 manifest covering every individual file. Analyst result files are a
separate, second-stage package and may be opened only after the independent recomputation and
verdict have been completed.
