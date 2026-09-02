# Root cause: the sampler corrupted V's zero-padded atom slots

- date: 2026-09-02
- checkpoint under test: `artifacts/formal500k/20260726T164217Z/ckpt_500000.pt`
  (sha256 `d0e7ae08f1a9e4f3ae11fa73c45f4e6005e9eac66754070b5b92fcaab91348e6`, step 500000)
- data: local mdCATH at `/Users/ringochen/hkucds/data/mdcath`
- evidence labels follow `CLAUDE.md`

## Verdict

The formal 500k model is **not** collapsed and **not** undertrained. The defect
that made rollout, folding, and every distributional result unusable was in the
**sampler**: `project_v_atom_mask` defaulted to `False`, so the endpoint ODE wrote
nonzero values into `V`'s zero-padded heavy-atom slots. Those values re-entered
the conditioner on the next iteration as inputs training never produced, and they
compounded until the structure disintegrated.

Fixed in this change set. The invariant is now on by default and pinned by tests.

**Scope limit:** this is a correctness fix, not a quality result. With the
measurement instrument repaired, the model's conformational sampling becomes
measurable for the first time — and on the TICA metric it is still not
distributionally faithful. See "What the fix does not buy".

## Evidence

### 1. `[MEASURED]` Half of V is padding, and sampling filled it with garbage

`V` is `[N, 13, 3]`; residues carry different heavy-atom counts, so 283 of 650
slots on domain `1a92A00` must stay exactly zero. Ground-truth magnitude there is
`0.000e+00`.

| ODE steps | project_v | padded-slot \|V\| max | real-slot \|V\| max |
|---:|---|---:|---:|
| 20 | off | 3.063e+01 | 14.399 |
| 20 | on | **0.000e+00** | **5.911** |
| 50 | off | 1.010e+02 | 24.455 |
| 50 | on | **0.000e+00** | **5.836** |
| 150 | off | 3.175e+03 | 2084.048 |
| 150 | on | **0.000e+00** | **5.774** |

With re-masking the real slots are step-count independent. Without it, both the
padding and the real coordinates diverge.

### 2. `[MEASURED]` Single-jump geometry broke between 30 and 50 Euler steps

Domain `1a92A00`, N=50, delta=1 ns, K=8 draws, `tau_max=1.0`, no re-masking.
Reference: true 1 ns CA displacement 2.794 A; real CA-CA bond 3.8 A.

| Euler steps | spread (A) | spread/ref | CA-CA bond (A) |
|---:|---:|---:|---:|
| 20 | 1.167 | 0.42 | 3.72 |
| 30 | 1.719 | 0.62 | 3.84 |
| 50 | 6.105 | 2.19 | **6.53** |
| 150 | 38.865 | 13.91 | **40.81** |

The former default (`steps=20`) sat just below the cliff, which is why single
jumps looked acceptable while anything longer did not.

### 3. `[MEASURED]` Only V re-masking rescued the rollout

20-step (20 ns) autoregressive rollout, `1a92A00`. Real MD over the same span:
bond 3.8 A, drift 8.4 A.

| setting | bond @20 (A) | drift @20 (A) |
|---|---:|---:|
| former default, ode 20, tau_max 1.0 | diverged by step 5 | diverged |
| `mode="mean"` | 3.6 | 5.0 |
| tau_max 0.9 + terminal denoise, 150 | diverged | diverged |
| **project_v every step, 150** | **3.8** | **11.2** |
| **both combined, 150** | **3.9** | **9.2** |
| real MD reference | 3.8 | 8.4 |

A truncated endpoint alone fixes a single jump but not a rollout. Re-masking is
the necessary ingredient; both together track real MD drift most closely.

### 4. `[CODE-VERIFIED]` The first-party release re-masks every step

`reviews/EVIDENCE_official_pypi_deepjump_0.1.0_20260718.md` records that the
public `tensorclouds` flow matcher "centralizes and remasks every step". The
reproduction exposed the same behaviour as an opt-in flag defaulting off.

### 5. `[MEASURED]` The new default holds end to end on the real checkpoint

No explicit argument passed, `tau_max=1.0` (the previously worst case):

| ODE steps | padded \|V\| max | CA-CA bond (A) |
|---:|---:|---:|
| 20 | 0.000e+00 | 3.87 |
| 50 | 0.000e+00 | 3.89 |
| 150 | 0.000e+00 | 3.90 |

## What the fix does not buy

### `[MEASURED]` Rollout TICA is domain-dependent, not a clean win

`--gen rollout`, 20 starts, 20 steps. Lower is better. A = `mode="mean"` + gate
(previously hardcoded); C = tau_max 0.9 + terminal denoise + project_v, ode 50.

| domain | A | C | no-dynamics reference |
|---|---:|---:|---:|
| 1a92A00 | 0.450 | **0.313** | 0.335 |
| 1avyB00 | 0.338 | 0.407 | 0.337 |

C wins on one domain and loses on the other. A plausible reason: C explores much
further (20-step drift 9-11 A vs mean-mode's 5.0 A), while the no-dynamics
reference is built from real MD start frames that already tile the landscape. The
metric therefore penalises any model that moves unless it moves correctly. Note
also that the reference value itself shifts with start count (0.335 at 20 starts
vs 0.411 at 12), so only same-start-count comparisons are meaningful.

### `[MEASURED]` Sampling-time sigma cannot test the release source law

Fixed sampler, `1a92A00`, K=8. Trained sigma 0.1; release ships
`var_coords=1.5`, `var_features=1.0`.

| sigma | spread (A) | spread/ref | CA-CA bond (A) |
|---:|---:|---:|---:|
| 0.1 | 1.647 | 0.59 | 3.86 |
| 0.3 | 8.929 | 3.20 | 3.95 |
| 0.5 | 10.762 | 3.85 | 4.20 |
| 1.0 | 17.657 | 6.32 | 10.66 |
| 1.5 | 56.402 | 20.19 | 34.92 |

A model trained at sigma=0.1 has never seen inputs perturbed at 1.5, so raising
sigma at sampling time only pushes it off-distribution. **The earlier hypothesis
that a too-small sigma caused an ensemble collapse is withdrawn**: at its trained
sigma with a working sampler the model produces spread/ref = 0.59 with valid
geometry, which is a real conditional distribution, somewhat under-dispersed but
not collapsed. The only valid test of the release source law is retraining;
`configs/v100_tensorcloud01_full_d1_first_party_source_law1000.yaml` encodes it
and has still never been run.

## Consequence for existing records

`mode="mean"` is the deterministic conditional mean at tau=0. It has **exactly
zero** ensemble spread (measured 0.000 A across 12 seeds), and for diffusive
dynamics the conditional mean is approximately the identity. It appears in ten
evaluation scripts:

```
endpoint_panel_eval.py               endpoint_grid_eval.py
guarded_endpoint_panel_eval.py       external_endpoint_root_cause.py
external_step2000_outlier_replay.py  transition_robustness_eval.py
robustness_eval.py                   rollout_robustness_eval.py
rollout_eval.py                      tica_eval.py
```

The endpoint family calls it as `steps=1, mode="mean"`. Any "no-op margin" study
built on those calls measured a quantity that is near zero by construction, and
the adjudication machinery in `reviews/` rests on those numbers.

`REPORT.md` section 4.5 attributes the distributional gap to scale. **That
conclusion is stale** — it predates the formal run. The 500k checkpoint already
matches the paper's printed recipe on domains (5,218 vs 5,398), temperatures,
replicas, steps (500k), width (H=128), depth (6+6), LR schedule (0.005 -> 0.003,
warmup 200, horizon 500k), crop (256), effective batch (2 x 8 x 8 = 128), and the
25 A all-atom Vector-Map loss. Scale is no longer the explanation. Section 4.5
and the `docs/tica_panel.png` figure, which was generated at `--steps 20` with no
re-masking, both need regeneration.

## Changes landed

| file | change |
|---|---|
| `src/deepjump/model/deepjump.py` | `project_v_atom_mask` defaults `True`; docstring records the mechanism and warns that `mode="mean"` has zero spread |
| `scripts/tica_panel.py` | supply the missing `atom_mask` |
| `tests/test_sampler_atom_mask_invariant.py` | new; pins the invariant, the default, the loud error, and rollout end to end |
| `tests/test_sampling_integrators.py` | the test pinning the old default now pins the new one |
| `tests/test_shapes.py` | fixture carries an `atom_mask` |

`331 passed` (was 320). `ruff check src scripts tests` adds no new findings.

The new test file includes a reverse guard asserting that padding *is* polluted
when re-masking is disabled, so the invariant test cannot silently become vacuous
if a future sampler stops touching V.

## Open items

- Regenerate `REPORT.md` 4.5 and `docs/tica_panel.png` with the fixed sampler.
- Move distributional and kinetic evaluations off `mode="mean"`; keep it only for
  deterministic identity checks and say so where it is used.
- Run the first-party source-law config; it is the only way to test sigma.
- Architecture still differs from the paper (hand-rolled GVP/EGNN at l=1 vs e3nn
  tensor products to l=2), so a fixed sampler is not expected to close the gap to
  the published numbers on its own.
