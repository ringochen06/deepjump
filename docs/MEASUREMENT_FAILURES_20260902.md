# Five measurement failures, and what they had in common

- date: 2026-09-02
- scope: the DeepJump-lite reproduction, formal 500k checkpoint
- companion: [`FINDING_sampler_root_cause_20260902.md`](FINDING_sampler_root_cause_20260902.md)

A single session's investigation turned up five separate defects. None of them
were in the model. Every one was in the apparatus used to judge the model, and
every one produced output that looked entirely reasonable. This note records the
pattern, because the pattern is more useful than any of the individual fixes.

## The five

### 1. The sampler polluted structural zero padding

`V` is `[N, 13, 3]`; residues carry different heavy-atom counts, so 283 of 650
slots on `1a92A00` must stay exactly zero. The endpoint ODE wrote into all of
them. The garbage re-entered the conditioner as an input training never produced
and compounded: padded-slot magnitude reached `3.2e3` by 150 Euler steps, taking
CA-CA bond from 3.8 to 40.8 A.

**Why it hid:** the default step count (20) sat just below the divergence cliff
(~30-50). Single jumps looked fine. Only long rollouts broke, and those were
attributed to the model.

### 2. The TICA JSD rewarded divergence

`hist2d_jsd` sized its histogram to the *union* of reference and model samples.
A diverging model stretched the extent until real MD occupied one or two bins,
and the divergence measure fell. Measured on synthetic data: a model sampling the
reference law exactly scores 0.174; a wrong model with 20 of 400 samples pushed
to ~100 scores 0.156, and 5 samples at ~1000 scores 0.004.

**Why it hid:** lower is better, and the numbers went down as the project scaled.
The `REPORT.md` "scale ladder" (0.564 -> 0.347) read as steady progress. It was
partly tracking how far each model diverged.

### 3. `mode="mean"` was used where a distribution was needed

The deterministic conditional mean has exactly zero ensemble spread (measured
0.000 A across 12 seeds), and for diffusive dynamics it approximates the
identity. `tica_eval`'s rollout branch was pinned to it, so the project's
distributional test was not a distributional test.

**Why it hid:** it is stable and produces physically valid structures. It fails
by being *too* well behaved.

### 4. The figure asserted provenance it never checked

The panel title read "held-out mdCATH domains" unconditionally, while rows came
from `split_domains()` over whatever shards sat on local disk. The formal
contract records 5,218 training domains out of mdCATH's 5,398. Checked against
the run's own list, all four domains in the committed figure were training
domains.

**Why it hid:** the claim was in a title string, not in a computation, so nothing
could contradict it.

### 5. The check written to catch (4) was inverted

The first fix added `--held-out-list` and treated its contents as the held-out
set. But the artifact a run produces is the *training* list. Passing it — the
only list that exists — would have labelled every genuinely held-out domain
"trained-on".

**Why it hid:** it never ran against real data before being committed. It was
caught by reading it back, one commit later.

## What they share

Each defect **produced plausible output**. None crashed, none returned NaN in the
common path, none looked anomalous. In three cases the broken measurement was
*more* flattering than the correct one, which is the dangerous direction: a
measurement that makes results look bad gets investigated, one that makes them
look good gets published.

Three of the five were in code written to *check* something — a metric, a
provenance label, a fix for a provenance label. Verification code gets less
scrutiny than the code it verifies, while carrying the same ability to be wrong.

The one conclusion that survived the session unchanged is the padded-slot
mechanism, and it survived because it is a directly measured physical quantity:
"these 283 array positions must equal zero, and they equal 3.2e3." It depends on
no aggregate, no baseline, and no summary statistic.

## Practice adopted here

- **Every fix ships with a test that fails if the fix is inverted.** Not just a
  test that the fixed behaviour works — one that fails if the logic is reversed.
  `test_sampler_atom_mask_invariant.py` asserts padding *is* polluted when
  re-masking is off, so the invariant test cannot go vacuous. `test_panel_
  provenance.py` asserts the code reads membership rather than its negation.
- **Metrics are validated against synthetic cases with known answers** before
  being trusted on real data. The JSD defect is a five-line demonstration; it
  went unnoticed for the life of the project.
- **Provenance is derived, never asserted.** A claim in a title string is not
  evidence. If it cannot be computed from a checksummed artifact, the figure says
  "unverified".
- **Callers declare their expectations and the script cross-checks them.** The
  generalisation scan takes `--held-out` and `--trained-on` and exits if either
  disagrees with the training list, so a mistyped domain cannot silently produce
  a wrongly grouped table.
- **Prefer directly measured quantities over aggregates** when establishing that
  something is broken. Aggregates are for deciding whether it is fixed.
- **State the prediction before running the test.** The generalisation scan was
  run to confirm that a landscape-difficulty confound explained the held-out
  advantage. It did not: training status correlated more strongly (+0.769) than
  difficulty (+0.566). Having written the prediction down first made that a
  result rather than something to quietly drop.

## Bearing on the reproduction's conclusions

`REPORT.md` carries a correction notice (section 0a). The distributional results
predating this session are void, not merely imprecise. Re-measured with the
sampler and metric repaired, the model reaches TICA JSD 0.38-0.58 across thirteen
domains, which covers roughly the right region of conformational space without
reproducing real MD's basin structure.

That is a weaker result than the project previously believed it had, and it is
the first one measured with an instrument that has been checked.
