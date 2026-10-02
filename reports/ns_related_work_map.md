# NS-Frame Related Work Map (companion to `ns_hypothesis_charter.md`)

**Status:** knowledge-based compilation. **Citations are NOT verified online in this session** —
`web_search` is disabled on this host and the arXiv/Semantic Scholar APIs were unreachable in an
earlier attempt. Treat author–year attributions as leads to verify, not as bibliography.
**Purpose:** answer, for the "NS frame", the question *what already exists*, so the frame's
genuinely new part can be isolated before any experiment is proposed.

---

## 1. Naming disambiguation (do this first, it causes confusion)

| Word | In ML it usually means | In the NS frame it would mean | Risk |
|---|---|---|---|
| **diffusion** | generative diffusion models (denoising score matching) | physical diffusion `ν∇²u` (smoothing / mixing) | writing "diffusion" invites the wrong literature and the wrong baseline |
| **transport** | optimal transport of distributions (Wasserstein) | advection of a field `(u·∇)u` | "transport" alone is ambiguous; say **advection** for the field sense |
| **flow** | normalising flows (density transformation) | fluid flow / velocity field | same ambiguity |
| **operator** | neural operator (DeepONet/FNO) | a parameterised map in a dynamical system | we use "operator" in the *linear vs nonlinear branch* sense |

**Rule for all later writing:** say *state-dependent advection-like shift* and *state-dependent
diffusion-like smoothing*, never bare "transport/diffusion/flow".

## 2. Operator families in ML that already encode physics-like structure

| Family | Core idea | Why it matters here |
|---|---|---|
| **Neural ODEs / continuous-depth** (Chen et al., 2018) | learn `dX/dt = f(X, t)` and integrate | the NS task's `dX/dt = L(X) + N_NS(X)` is a *special case*; the general version is already standard — so the claim cannot be "we use a continuous-time formulation" |
| **Hamiltonian / Lagrangian NNs** (Greydanus et al., 2019; Cranmer et al., 2020) | impose conservation/energy structure | the "conservation" frame is already instantiated; in markets there is no conserved quantity, so this family is likely the *first to abandon* (charter §2 failure mode) |
| **Invariant / conserving networks** (e.g. conserving architectures, Lie-symmetry nets) | preserve specified quantities by construction | only useful if a real invariant exists; our data offers none |
| **PINNs** (Raissi et al., 2019) | add a PDE residual as a loss term | the original NS task explicitly rejects this route ("not as an extra loss") — keep that |
| **Neural operators: DeepONet / FNO** (Lu et al., 2021; Li et al., 2021) | learn maps between function spaces; **advection–diffusion is a standard benchmark** | the single strongest "already done" family: if the proposal is "an operator that respects advection–diffusion structure", this is the literature to beat |
| **State-space models / long convolutions** (S4/S5/Mamba-style) | continuous-time linear dynamics + selection | "learned, input-dependent time-scaling" already exists here — a state-dependent advection proposal must show it is not just selective gating |
| **Attention as particle interaction** (particle/Lagrangian readings of transformers) | tokens transported and mixed | the particle frame is already instantiated by attention; charter §2 flags this |
| **Spectral / Fourier mixing** | global mixing in frequency domain | diffusion-like smoothing is partially covered |
| **Optimal transport in ML** | distribution shift, Wasserstein objectives | different estimand (distribution-level), not a drop-in branch |
| **Generative diffusion models** | denoising | *unrelated* to the physical frame; do not cite as prior art |

**Implication:** three of the charter's candidate frames (transport, diffusion, splitting) are
*crowded*. The defensible claim cannot be "structured nonlinear operator"; it must be narrower —
e.g. **a specific state-dependent form that a capacity-matched MLP, SSM and attention block cannot
reproduce at equal budget**, which is precisely charter §7 question 2.

## 3. Finance-specific precedents

| Line of work | Content | Relation to this project |
|---|---|---|
| **Stochastic volatility / SDE models** (Heston-type) | returns as diffusion processes with state-dependent coefficients | the *classical* finance instantiation of "diffusion"; our data-driven analogue would be a learned coefficient function |
| **Fokker–Planck for return distributions** (econophysics) | evolve the density of returns | distribution-level, not branch-level |
| **PINN for option pricing / Black–Scholes PDE** | physics-informed training against a known PDE | shows the machinery works *when the governing PDE is known*; in forecasting no such PDE exists |
| **Mean-field games in finance** | interacting-agent dynamics | analogy source, not a forecasting operator |
| **Econophysics hydrodynamic analogies** | "markets as fluids" narratives | historically rich, predictively thin — the reason our protocol bans the phrase as a factual claim |
| **Regime-switching state-space models** (Hamilton-type) | discrete state dynamics on top of a linear model | the *best-founded* classical analogue of "state-dependent behaviour", and the natural baseline our routing analysis already echoes |
| **RevIN / distribution-shift normalisation** | per-window normalisation | already inside our framework; addresses level/scale drift that could masquerade as "transport" (charter §2 failure mode) |
| **Our own negative results** | capacity sensitivity, asset dependence, no sample-level routing | constrain any proposal: a universal gain is already excluded |

## 4. Mapping: frame → prior art density → falsification cost

| Charter frame | Prior-art density | Cheapest falsification test (capacity-matched) | Notes |
|---|---|---|---|
| State-dependent **advection-like shift** | high (SSM, deformable conv, warping) | learned shift field vs fixed-depth convolution, same params | likely to reduce to input-dependent time-scaling |
| State-dependent **diffusion-like smoothing** | medium | learned, state-conditioned kernel vs fixed moving average (25) | directly comparable to our `moving_avg` 25 baseline |
| **Operator splitting** (advection ⊕ diffusion) | high (compositional blocks) | composition vs single block at equal budget | risk: expensive re-parameterisation of an MLP |
| **Conservation / invariants** | high but inappropriate | — | no financial invariant; abandon up front |
| **Pressure / constraint term** | low in ML, high in CFD | — | original task deferred it; our evidence finds no necessity for a pressure-like constraint |
| **Multi-scale cascade** | high (hierarchical/seasonal models) | — | not NS-specific |
| **Particle / Lagrangian** | high (attention) | — | must show what attention cannot do |
| **Optimal transport of distributions** | medium | — | different estimand; RevIN partially covers it |

## 5. What this means for the charter (edits proposed, not applied)

1. Add the naming-disambiguation table (§1) as a mandatory vocabulary rule — prevents the
   "generative diffusion" and "optimal transport" confusions.
2. Mark **three** frames as *already crowded* (advection/SSM, diffusion/spectral, splitting) so the
   novelty test is explicit rather than assumed.
3. Mark **two** frames as *abandon up front* on our own evidence: conservation/invariants (no
   invariant) and pressure/constraint (our findings show no necessity).
4. Keep the priority rule "cheapest to falsify first": **state-dependent diffusion-like smoothing**
   is the cheapest (single kernel, directly comparable to the existing fixed moving-average), then
   **state-dependent advection-like shift**, then splitting.

## 6. Verification backlog (needs real literature access)

- [ ] Verify the neural-ODE / Hamiltonian / PINN / DeepONet–FNO attributions and years.
- [ ] Find whether a *state-dependent* advection–diffusion operator has been applied to
      financial forecasting specifically (not PDE benchmarks).
- [ ] Find finance precedents for learned state-dependent diffusion coefficients in a forecasting
      (not pricing) setting.
- [ ] Check whether our "state-dependent smoothing vs fixed moving average" test has been published
      as a negative result already.

*If literature access is restored (or an external assistant has it), this file is the checklist to
run against it; until then the citations above are leads, not evidence.*
