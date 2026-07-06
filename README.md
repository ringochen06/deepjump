# DeepJump-lite

A plain-PyTorch reproduction of the core trainable interface from
**DeepJump** (arXiv:2509.13294): a learned conformational **jump operator**

$$p(X_{t+\delta} \mid X_t,\ \text{sequence},\ \delta)$$

that predicts a protein's conformation $\delta$ nanoseconds ahead in one shot,
trained on the public **mdCATH** MD dataset. This is a *lite* research prototype
(runs on a laptop / Apple-Silicon MPS), not a full reproduction of the paper's
fast-folder results.

## What is implemented (stage 1)

- **Representation** `X = (P, V)` from Ophiuchus: `P ∈ R^{N×3}` (Cα coords),
  `V ∈ R^{N×13×3}` (heavy-atom offsets from Cα, canonical order, zero-padded).
- **mdCATH dataloader**: samples aligned state pairs `(X_t, X_{t+δ})`, δ = 1 ns.
  Rigid-body tumbling is removed by **Kabsch-aligning** `X_{t+δ}` onto `X_t`, so
  the target is a purely internal conformational change (~1 Å, not ~10 Å).
- **SE(3)-equivariant two-stage model** (GVP/EGNN-style, no e3nn tensor products):
  - *Conditioner* (6 layers): `(X_t, sequence, δ) → H_t`
  - *Transport* (6 layers): `(X^τ, τ, H_t) → X̂_1` — predicts Cα, and optionally
    the heavy-atom offsets `V̂_1` (`predict_heavy`, residual `V_t + dV`).
  - equivariant self-attention + GVP feed-forward + vector LayerNorm.
- **AlphaFlow-style x₁ prediction**: linear interpolant `X^τ = (1−τ)(X_t+ε) + τX_{t+δ}`,
  ODE drift `b = (X̂_1 − X^τ)/(1−τ)` for sampling.
- **Losses**: pairwise vector Huber on Cα–Cα difference vectors (Ophiuchus
  Vector-Map), plus an optional heavy-atom offset Huber term (`w_offset`), both
  padding/atom-masked.
- **Tests**: rotation equivariance of the representation *and* the full model,
  masking/padding invariance, shapes — the silent-bug gate for equivariant nets.

Stage-2 in progress: heavy-atom offset output + offset loss (`configs/full_delta1.yaml`)
done. Still deferred: 25 Å all-atom pairwise loss, ODE rollout, δ = 10/100 ns,
TICA/MSM distributional evaluation.

## Layout

```
src/deepjump/
  atom_constants.py     canonical heavy-atom ordering per residue
  representation.py     build (P,V); Kabsch alignment; rotation-covariant by construction
  data/mdcath.py        HDF5 reader + (X_t, X_{t+δ}) pair dataset
  model/                embeddings, equivariant layers, conditioner, transport, wrapper
  losses.py  metrics.py  utils.py
scripts/
  download_mdcath.py    fetch N smallest mdCATH domains from HuggingFace
  inspect_h5.py         dump one domain's HDF5 structure
  train.py  eval.py
configs/ca_delta1.yaml  Cα, δ=1 ns, H=32
tests/                  pytest suite (equivariance / masking / shapes)
```

## Setup

```bash
mamba create -n deepjump python=3.11 -y
conda activate deepjump
pip install -e .
```

## Data

mdCATH is public. Download a small subset (one HDF5 per domain, ~0.2–1.6 GB each):

```bash
python scripts/download_mdcath.py --n 5 --max-gb 0.6      # 5 smallest domains
python scripts/inspect_h5.py ~/hkucds/data/mdcath/data/mdcath_dataset_*.h5
```

mdCATH HDF5 layout: `domain → temperature{320,348,379,413,450} → replica{0..4} →
coords[F,A,3]` (Å, 1 ns/frame). Atom names are recovered from the embedded PSF
(protein atoms only), combined with the per-atom `resid`/`resname` arrays.

## Train & evaluate

```bash
python scripts/train.py --config configs/ca_delta1.yaml --fast-dev   # overfit 1 batch -> loss~0
python scripts/train.py --config configs/ca_delta1.yaml              # Ca, δ=1 ns
python scripts/train.py --config configs/full_delta1.yaml            # Ca + heavy-atom offsets
python scripts/train.py --config configs/ca_delta10.yaml             # δ=10 ns
python scripts/eval.py         --ckpt runs/ca_delta1/last.ckpt       # single-step vs baselines
python scripts/diagnose_tau.py --ckpt runs/ca_delta1/last.ckpt       # RMSD vs τ sweep
python scripts/train.py --config configs/full_delta1_unroll3.yaml    # 3-step self-conditioning (stable rollout)
python scripts/train.py --config configs/full_delta1_allatom.yaml    # 25 A all-atom Vector-Map loss
python scripts/train.py --config configs/full_delta1_h64.yaml        # H=64 capacity sweep
python scripts/rollout_eval.py --ckpt runs/full_delta1_unroll3/last.ckpt --mode mean --gate  # rollout
python scripts/tica_eval.py    --ckpt runs/full_delta1_unroll/last.ckpt   # TICA distributional JSD
python scripts/plot_summary.py && python scripts/plot_stability.py   # docs/*.png
pytest -q                                                            # correctness gate (14/14)
```

See **[REPORT.md](REPORT.md)** for the consolidated reproduction report (scope, results,
differences from the paper, honest findings, next steps).

## Stage-1 result (honest)

Trained H=32 (114k params), Cα, δ=1 ns, on a 30-domain mdCATH subset (24 train /
6 held-out val domains, ~10.6k train pairs, MPS). `pytest` 11/11; fast-dev overfit
collapses loss 6.1 → 0.003. On held-out data, CA RMSD as a function of the latent
time τ (`scripts/diagnose_tau.py`), with the earlier 5-domain run for comparison:

| query | 5 domains | 30 domains |
|---|---|---|
| no-op (`X̂ = X_t`) | 1.60 | 1.58 |
| one-shot x̂₁ @ τ=0 | 1.67 | 1.65 |
| one-shot x̂₁ @ τ=0.25 | 0.87 | 0.70 |
| one-shot x̂₁ @ τ=0.50 | 0.58 | 0.52 |
| one-shot x̂₁ @ τ=0.75 | 0.49 | 0.45 |
| one-shot x̂₁ @ τ=0.90 | 0.46 | 0.42 |
| ODE sample (20 Euler steps) | 2.12 | 1.87 |

Scaling 5→30 domains improved the transport field at every τ>0 and the ODE sample
(2.12→1.87, on unseen val domains), but left the τ=0 / no-op story unchanged —
confirming that is a modeling/eval limitation, not a data-scale one.

**Stage-2 (heavy-atom output, `configs/full_delta1.yaml`)**: the transport head also
predicts heavy-atom offsets `V̂_1`. They follow the same τ-curve as Cα — heavy-atom
offset MAE (Å): no-op (`V̂=V_t`) 0.43 · τ=0 0.47 · τ=0.5 0.17 · τ=0.9 0.15 — and adding
the offset target does not hurt Cα (τ=0.9 RMSD 0.42, same as Cα-only). Model stays
SE(3)-equivariant (test covers both `P̂_1` and `V̂_1`); `pytest` 14/14.

**Multi-step rollout** (`scripts/rollout_eval.py`, chain the jump along a real
trajectory). Real dynamics move only ~2 Å from the start over 10 ns; the model
rollout is **unstable** — error and Cα–Cα bond length compound off the training
distribution:

| step (ns) | ODE RMSD (Å) | ODE bond (Å) | mean RMSD (Å) | mean bond (Å) | true drift (Å) |
|---|---|---|---|---|---|
| 1 | 2.4 | 4.2 | 1.8 | 3.5 | 1.8 |
| 5 | 71 | 92 | 4.9 | 4.6 | 2.2 |
| 10 | 8495 | 11604 | 18.8 | 19.4 | 2.1 |

The **ODE sampler explodes immediately** (each jump inflates the structure, feeding
back compounds it). The **deterministic mean predictor** (`mode="mean"`, x̂₁ at τ=0)
holds valid geometry (bond ≈3.8 Å, contacts) for ~7 steps then also diverges. This is
the classic **rollout-instability / distribution-shift** problem: a single-step model
with no correction accumulates error.

**Stability fix — geometry acceptance gate** (`--gate`, `rollout(..., gate=True)`). A
proposed jump is accepted only if its Cα–Cα geometry stays physical; otherwise the
previous frame is kept. This is the energy-free spirit of Timewarp's accept/reject:

| step-10 CA RMSD (Å) | no gate | with gate |
|---|---|---|
| ODE | 8495 | 2.25 |
| mean | 18.8 | 2.49 |

The gate **bounds the rollout** but at a **low acceptance rate** (mean 0.20, ODE 0.02):
once the model would drift off-manifold it mostly rejects and freezes — stable but
conservative. That tension (stability vs acceptance) is exactly Timewarp's finding.

**Stability fix #2 — input-augmentation training** (`input_aug_sigma`, `configs/full_delta1_aug.yaml`).
The gate is inference-only; the real cure is making the *model* robust. We perturb the
conditioner's input structure `X_t` with per-sample noise during training, so the model
learns to recover from imperfect inputs — exactly the off-distribution structures it feeds
itself during a rollout. This is a genuine improvement, not a band-aid:

| rollout (mean, ungated) | baseline | **aug-trained** |
|---|---|---|
| CA RMSD @ step 10 | 18.8 Å | **4.5 Å** |
| CA–CA bond @ step 10 | 19.4 Å | **4.4 Å** |
| stable horizon | ~7 steps | **~15 steps** |
| single-step RMSD @ τ=0.9 | 0.42 | **0.37** (improved) |

**Stability fix #3 — 2-step unrolled (self-conditioning) training** (`configs/full_delta1_unroll.yaml`,
`data.unroll: 2`, `train.w_unroll`). Instead of gaussian noise, train on the model's *own*
prediction: supervise `f(f(X_t)) ≈ X_{t+2δ}`, feeding the detached step-1 output as the step-2
input. This gives the **best single-step accuracy** (τ=0.9 **0.362**, τ=0 **1.59**) and the
**tightest near-term rollout geometry** (step-10 bond 3.49 Å vs aug's 4.40, ideal 3.8).

**Summary of rollout behavior (mean mode):**

| approach | ungated stable horizon | step-20 RMSD (Å) | step-20 bond (Å) |
|---|---|---|---|
| baseline ODE | 0 (explodes) | 8495 | 11604 |
| baseline mean | ~7 steps | 18.8 | 19.4 |
| aug-trained | ~15 steps | 62 | 35 |
| 2-step unroll | ~13 steps | 70 | 90 |
| **3-step unroll** | **~20 steps** | **6.4** | **4.5** |
| **3-step unroll + gate** | **20+ (stable)** | **2.9** | **3.9** (FNC 0.81) |

**Deeper unrolling is the real cure for the horizon**: 3-step self-conditioning keeps the
*ungated* rollout bounded and physical over the full 20 steps (step-20 6.4 Å / bond 4.5 Å,
vs 2-step's 70 Å / bond 90). Adding the gate on top holds it fully stable (RMSD ~2.9 Å, bond
~3.9 Å, contacts 0.81) at 0.22 acceptance. Figure: `docs/stability.png`. The trend is clear —
each extra unroll step extends the horizon; full closure would continue with deeper unrolling,
energy-based MH, or an SDE / two-sided interpolant (EquiJump).

**Model capacity & loss.** H=64 (vs H=32) improves every single-step metric — τ=0 1.67→1.58
(matches no-op), τ=0.9 0.42→0.35, ODE sample 2.44→1.84. The faithful **25 Å all-atom
Vector-Map loss** (`configs/full_delta1_allatom.yaml`) trains cleanly and modestly beats the
Cα+offset loss (ODE 2.44→1.96).

**Distributional evaluation (TICA).** `scripts/tica_eval.py` fits TICA on a real trajectory
(SE(3)-invariant Cα-pairwise features) and compares the model's rollout ensemble to real MD in
TIC space. Honest result: **JSD(model, real) = 0.58 > start-only 0.38** — even the *stable*
gated rollout does **not** reproduce the equilibrium landscape; it under-explores and drifts to a
biased region (`docs/tica.png`). This is the DeepJump evaluation philosophy at lite scale, and it
quantifies the gap that a single-step model without a proper unbiased sampler cannot close.

**Jump size δ=1 vs δ=10 ns** (`configs/ca_delta10.yaml`, same H=32 net). A 10 ns jump
is bigger and harder — CA RMSD (Å):

| query | δ=1 ns | δ=10 ns |
|---|---|---|
| no-op | 1.58 | 2.25 |
| x̂₁ @ τ=0 | 1.65 | 2.28 |
| @ τ=0.5 | 0.52 | 0.64 |
| @ τ=0.9 | 0.42 | 0.53 |
| ODE sample | 1.87 | 2.68 |

Error is uniformly higher at δ=10 but the τ-curve story is identical. 10× the elapsed
time only moves the structure ~1.4× as far (no-op 1.58→2.25) — protein motion is
bounded/sub-diffusive, which is exactly why learning large-δ jumps buys real MD
acceleration. This is DeepJump's acceleration/accuracy tradeoff at lite scale.

**Reading**: the model clearly *learned the transport field* — accuracy improves
monotonically as the interpolant input approaches the answer (1.67 → 0.46). But
the two generative-critical regimes do **not** beat no-op yet:

- **τ=0 one-shot** (predict `X_{t+δ}` from `X_t` alone). A deterministic x₁-predictor
  under MSE approximates the conditional mean `E[X_{t+δ}|X_t]`, which for diffusive
  1 ns dynamics is ≈ `X_t`; so being near no-op here is expected, and single-step
  RMSD vs no-op is a **weak** success signal (the honest eval queries τ=0, not
  random τ, which would leak the answer).
- **ODE sampling** diverges (2.12) because the miscalibrated τ≈0 drift compounds
  over Euler steps.

This precisely scopes stage-2: better τ≈0 calibration / sampler (more steps, SDE
or two-sided stochastic interpolant à la EquiJump), more data/steps, and above all
a **distributional** evaluation (sampled-ensemble contact/pair-distance stats,
TICA/MSM) — which is how DeepJump itself is judged, not by single-step RMSD.
