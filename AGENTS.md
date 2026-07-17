# DeepJump Codex Instructions

## Mission

Own the DeepJump reproduction workflow end to end: implementation correctness,
scientific comparison with the paper, Huawei Cloud execution, reproducible
records, evaluation, and concise project reporting.

Do not conflate:

1. implementation correctness;
2. a closer-to-paper reproduction;
3. an exact reproduction of the published method.

State the category supported by the available evidence. The repository's current
implementation is not automatically equivalent to the paper's original model.

## Project context

- Primary branch: `cloud-fullscale`
- Paper:
  `/Users/ringochen/Desktop/HKUCDS/02_DeepJump_Reproduction/DeepJump - Accelerating Protein Molecular Dynamics Simulation with DeepJump - Costa et al. 2025.pdf`
- Project notes:
  `/Users/ringochen/Library/Mobile Documents/iCloud~md~obsidian/Documents/hkucds/DeepJump`
- Current status:
  `/Users/ringochen/Library/Mobile Documents/iCloud~md~obsidian/Documents/hkucds/DeepJump/STATUS.md`

Read `STATUS.md`, relevant configs, tests, and `git status` before substantial
work. Update the status and run log after each cloud or experiment stage.

## Code and scientific correctness

- Verify diagnoses against code and runtime evidence before editing.
- Make the smallest complete change and preserve existing APIs and configs unless
  explicitly authorized otherwise.
- Add regression tests for correctness bugs.
- Distinguish measured results, inferences, and estimates.
- Never invent benchmarks, metrics, costs, test outcomes, or paper equivalence.
- For numerical changes, check stability, mixed precision behavior, gradient
  synchronization, checkpoint compatibility, and relevant baselines.
- Do not commit or push unless explicitly requested in the current task.

## Cloud and data safety

- Obtain explicit approval before purchasing, deleting, resizing, restarting, or
  releasing cloud resources; starting formal training; or downloading more than
  10 GB.
- Never read, print, copy, store, or commit AK/SK values, tokens, private keys, or
  credential files. Credential entry is performed by the user.
- Do not delete raw data or checkpoints until counts, hashes/readability, OBS
  upload, and OBS readback have been verified.
- Use local EVS for training data and OBS for persistent staging/checkpoints.
- Before expensive GPU work, report the expected duration, expected cost, stop
  condition, checkpoint plan, and recovery procedure.
- Long-running jobs must survive SSH disconnects. Record the PID/session, log
  path, status command, safe stop command, and resume command.

## Required stage gates

1. Local audit: branch, commit, clean/dirty tree, configs, tests.
2. Data audit: subset identity, file count, manifest count, total bytes, SHA256,
   HDF5 samples, and zero unresolved failures.
3. Infrastructure audit: instance identity, GPU count, driver/CUDA, mounts, free
   space, OBS access, and deployed repository commit.
4. Eight-GPU smoke: world size, effective batch, NCCL, finite losses, memory,
   throughput, validation, atomic checkpoint, and checkpoint readback.
5. Short bounded calibration run before formal training.
6. Formal run only after explicit user approval.
7. Evaluation against TICA/JSD, RMSD, and no-op or other stated baselines.

## Records

After each stage update:

- `DeepJump/STATUS.md` for the single current state and next action;
- `DeepJump/RUN_LOG.md` for commands, code/config versions, resources, timing,
  costs, metrics, artifacts, validation, and recovery;
- `DeepJump/DECISIONS.md` for consequential scientific or infrastructure choices;
- `DeepJump/REPORT_NOTES.md` for evidence suitable for reports.

Use English for code, identifiers, comments, and commands. Use concise Chinese
for user-facing progress and reports unless the user writes in English.
