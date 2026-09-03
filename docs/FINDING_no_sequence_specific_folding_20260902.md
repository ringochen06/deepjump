# The model compacts chains; it does not fold them

- date: 2026-09-02
- checkpoint: `artifacts/formal500k/20260726T164217Z/ckpt_500000.pt`
  (sha256 `d0e7ae08f1a9e4f3ae11fa73c45f4e6005e9eac66754070b5b92fcaab91348e6`)
- probe: `scripts/free_rollout_probe.py`
- companion: [`MEASUREMENT_FAILURES_20260902.md`](MEASUREMENT_FAILURES_20260902.md)

## Verdict

`[MEASURED]` With the sampler defect fixed, the formal 500k model takes a fully
extended chain and contracts it to native-scale compactness, holds legal geometry
for well over a thousand jumps, and produces no steric clashes. None of that is
folding. Across three domains, two sampler seeds, and a shuffled-sequence
control, the recovered native-contact fraction is **not distinguishable from what
a random compact chain achieves by geometry alone**.

`STATUS.md`'s existing statement — no evidence of stable folding from a fully
extended state to the native state — stands, and is now supported by a direct
test rather than by the absence of one.

## Why a native-contact fraction needs a floor

Any chain of the right length packed into the right volume recovers some native
contacts by geometry alone. Without knowing how many, an FNC of 0.6 cannot be
called folding. `random_compact_chain()` builds a self-avoiding chain with correct
CA-CA bond lengths confined to a sphere at 1.3x the native radius of gyration, and
scores it against the native structure. That is the floor.

It is strongly domain-dependent, which is why a fixed threshold would mislead:

| domain | random-chain floor FNC |
|---|---:|
| 1hw7A02 | 0.485 |
| 1pyvA00 | 0.463 |
| 3dmqA01 | 0.176 |

## `[MEASURED]` The test

Extended start, 600 jumps, `ode_20`, native-contact fraction at the final step:

| domain | floor | real seq, seed 0 | real seq, seed 1 | shuffled seq |
|---|---:|---:|---:|---:|
| 1hw7A02 | 0.485 | **0.640** | 0.600 | 0.520 |
| 1pyvA00 | 0.463 | 0.477 | **0.331** | 0.385 |
| 3dmqA01 | 0.176 | 0.160 | 0.176 | 0.144 |

Three readings, each fatal to a folding claim on its own:

1. **It does not hold across domains.** Only `1hw7A02` clears its floor.
   `1pyvA00` ties it (0.477 vs 0.463) and `3dmqA01` falls below it (0.160 vs
   0.176).
2. **It does not hold across seeds.** `1pyvA00` gives 0.477 and 0.331 on two
   sampler seeds — a spread as large as the entire effect being claimed.
   `1hw7A02` seed 1 ends at Rg 27.47 A, never reaching native scale at all.
3. **Shuffling the sequence barely matters.** Permuting `res_index` keeps
   composition and length while destroying sequence-specific signal. FNC drops by
   0.12 / 0.09 / 0.02. On `1pyvA00` the shuffled run (0.385) *beats* the real
   sequence on seed 1 (0.331): the sequence effect is smaller than seed noise.

## `[MEASURED]` What the model genuinely does

Real capability, worth recording:

| observation | value |
|---|---|
| contraction from extended | Rg 51.95 -> ~11 A, matching the native 10.9-12.2 A |
| geometric horizon | 1293 steps (native start) / 1755 steps (extended start) before NaN |
| CA-CA bond throughout | 3.79-3.85 A against a real 3.8 A |
| steric clashes | 0.0000 in every run but one (`3dmqA01` seed 0 at 0.0008) |

For reference, real MD on `1hw7A02` at 320 K drifts to RMSD 13.98 A / FNC 0.613
within 50 ns and stays in FNC 0.52-0.67 thereafter. The model started from the
native frame drifts *less* than the real trajectory does over the same elapsed
time (RMSD 10.51, FNC 0.667 at 200 ns), so the compaction behaviour is not an
artefact of an unstable rollout.

## Bearing on ab initio folding

| layer | status |
|---|---|
| measurement instrument | fixed this session |
| geometric stability | ~1.3-1.8e3 jumps, no clashes, correct bond lengths |
| compaction to native scale | works, from a fully extended chain |
| **sequence-specific folding** | **not distinguishable from zero** |
| long horizon (paper uses 3e5 jumps) | untested above ~1.5e3 |

The last two are what remain, and the fourth is not a budget problem. A model
that is not using sequence to decide *which* fold to form will not get there by
training longer. Any restart under `GOAL.md`'s preregistration requirement should
carry a sequence-specificity criterion and a random-compact-chain control from
the first experiment.

## Method note

This conclusion was reached after three wrong readings, each corrected by adding
a control that should have existed first:

| reading | refuted by |
|---|---|
| sigma=0.1 blocks off-equilibrium input entirely | Rg contracted 52 -> 11 A |
| the chain will collapse to a ball, FNC < 0.3 | FNC reached 0.64 |
| FNC 0.64 beats the 0.560 baseline, so it folds | wrong baseline; a random compact chain scores 0.485 |

The error was constant: interpreting a number without the control that makes it
readable. The random-compact-chain floor should have been the first thing built,
not the fourth.
