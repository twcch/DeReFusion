# 10 · Structural Validation — BLOCKED at asset acquisition (+ internal robustness §9)

**Status:** `BLOCKED` — the protocol's step 2 (select ≥2 clearly-low and ≥2 clearly-high
`|ACF1|` **new** assets, pre-registered before any operator run) cannot be executed on this
host right now because **no new market data can be acquired**.

**Nothing was substituted silently.** Gate A is **not issued**, because issuing it requires the
new assets. No NS / transport / diffusion / gating / router / MoE work was started.

## 1. Data-acquisition attempts (all four channels tested 2026-09-13)

| Channel | Result | Exact failure |
|---|---|---|
| Yahoo Finance chart API via `urllib` | ❌ | `URLError: <urlopen error _ssl.c:1011: The handshake operation timed out>` |
| `curl_cffi` browser-fingerprint path (the method that worked for the original ten datasets) | ❌ | `pip install curl_cffi` → *No matching distribution found* (venv is Python 3.13; no wheel on the configured mirror); module therefore absent |
| Stooq CSV endpoint (`stooq.com/q/d/l/?s=…&i=d`) | ❌ | request hangs; no payload returned |
| OpenClaw `web_fetch` on the same Stooq URL | ❌ | *Readability, provider fallback and basic HTML cleanup returned no content* |
| Local disk search for substitute market data (other projects / caches) | ❌ | only `matplotlib` sample data; no usable OHLC series |

Consequence: the candidate pool (`reproduction/data/fetch_candidate_pool.py`, 25 assets across
indices / equities / FX / crypto / commodities / funds) is **ready to run the moment a working
network path exists**; the pre-registration document will be written immediately after the pool
is fetched — i.e. strictly before any operator experiment, exactly as the protocol requires.

## 2. What *was* completed: protocol §9 three-way check on existing data

The protocol requires distinguishing three explanations for ρ(|ACF1|, Δ_interaction) = +0.733:
**(A)** a genuine cross-asset relation, **(B)** leverage by a few assets, **(C)** a
capacity-sensitive artefact. Using only the existing ten assets
(`reproduction/analysis/association_robustness_check.py`):

| View | ρ(\|ACF1\|) | ρ(realized vol) |
|---|---|---|
| all 10 assets | **+0.733** | +0.661 |
| drop NVO (highest \|ACF1\|, 0.156) | +0.633 | +0.683 |
| drop SOX (0.115) | +0.683 | +0.633 |
| drop BTCUSD (0.084) | +0.683 | +0.667 |
| drop GSPC (the only 2-seed asset) | **+0.817** | +0.600 |
| drop ETHUSD (lowest \|ACF1\|, 0.053) | +0.783 | +0.900 |
| **range across all single-asset drops** | **[+0.633, +0.817], no sign flip** | [+0.533, +0.900], no sign flip |

**Capacity view** (only the three assets that have the width sweep): ρ(|ACF1|, mean ΔMSE) is
+1.000 at width 64 and +1.000 at width 128 — **but n = 3**, so this is a degenerate rank
correlation and is reported as *directionally consistent*, not as evidence.

**Direction-of-preference check** against the association's own prediction:

| Asset | \|ACF1\| | Δ_interaction | mean ΔMSE @128 | Prediction | Outcome |
|---|---|---|---|---|---|
| ETHUSD | 0.053 | −0.00943 | **−0.0320** | low \|ACF1\| → nonlinear-favoured | ✅ matches |
| BTCUSD | 0.084 | +0.01502 | **+0.2998** | high \|ACF1\| → linear-favoured | ✅ matches |
| GSPC | 0.076 | −0.01726 | +0.0061 | mid \|ACF1\| → near parity | ✅ matches |

**Reading.** On existing data, explanation **(B) is rejected** (no leverage-driven sign flip; the
association survives dropping every single asset, and *strengthens* when GSPC is dropped) and
explanation **(C) is not supported by the available capacity data** (both widths give the same
direction). That leaves **(A)** as the working hypothesis — but this is **internal robustness
only**: it reuses the same ten assets that generated the association and therefore cannot
validate it externally. The protocol explicitly requires new assets for that, so:

```
Gate A: NOT ISSUED (blocked)
NS benchmark eligibility: unchanged (CONDITIONAL, from the previous gate)
```

## 3. Options to unblock (operator decision required)

1. **Restore the data channel** (proxy/VPN or an allow-listed route), then the pool fetch runs
   unchanged — recommended, because it is the only route that yields genuine new assets.
2. **Supply the data out-of-band** (fetch on another machine/peer and drop the CSVs into
   `dataset/`); the pre-registration + experiments then proceed as specified.
3. **Approve a narrowed design** (e.g. temporal out-of-sample split of the existing ten assets as
   a *substitute* validation) — requires explicit approval, because it deviates from the
   protocol's "new assets" requirement and will be labelled as such in the record.
4. **Hold** until the network situation changes; the evidence-closure state remains valid.

## 4. Discipline notes

- No definition was changed (preprocessing, split, horizon, window, volatility, features,
  thresholds, statistics, seed protocol, capacity settings all untouched).
- The volatility conclusion remains **INCONSISTENT** and was **not** re-framed to lean on the
  structural association.
- The new script and this record are committed; no claim in this file relies on data that is not
  in the repository.
