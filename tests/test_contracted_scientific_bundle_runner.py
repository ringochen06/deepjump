from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "cloud" / "huawei" / "run_contracted_scientific_bundle.sh"


def test_runner_is_bounded_fail_closed_and_never_trains():
    text = RUNNER.read_text(encoding="utf-8")
    assert "HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-240}" in text
    assert '[[ "$HARD_STOP_MINUTES" == 240 ]]' in text
    assert "/usr/bin/systemctl poweroff" in text
    assert "systemctl is-active --quiet" in text
    assert "SHUTDOWN_ON_EXIT must be 1" in text
    assert "git status --porcelain=v1 --untracked-files=all" in text
    assert "export PYTHONNOUSERSITE=1" in text
    assert "DATA_ROOT mount must be read-only" in text
    assert "deepjump-mdcath-download.service" in text
    assert "deepjump-mdcath-hash.service" in text
    assert "deepjump-mdcath-copy.service" in text
    assert '[[ "$gpu_count" == 8 ]]' in text
    assert "obs_prefix_preflight" in text
    assert "verify_obsutil_empty_prefix.py" in text
    assert "train_ddp.py --config" not in text
    assert "torchrun" not in text.lower()


def test_runner_preserves_authorization_until_both_kernels_exist():
    text = RUNNER.read_text(encoding="utf-8")
    evaluator_status = text.index(
        "contracted_scientific_bundle_eval.py --implementation-status"
    )
    evaluator_run = text.index(
        "scripts/contracted_scientific_bundle_eval.py \\", evaluator_status + 1
    )
    assert evaluator_status < evaluator_run
    assert 'numerical_kernel_implemented") is not True' in text
    assert 'independent_numerical_recomputation_implemented") is not True' in text


def test_runner_freezes_child_budgets_and_readback():
    text = RUNNER.read_text(encoding="utf-8")
    assert "12m" in text
    assert "150m" in text
    assert "10m" in text
    assert text.count("timeout 8m obsutil") >= 5
    assert "--sha256-file \"$READBACK_ONE/raw.json\"" in text
    assert "--sha256-file \"$READBACK_TWO/decision.json\"" in text
    assert "--sha256-file \"$READBACK_THREE/state_archive.npz\"" in text
    assert "sha256sum" not in text
    assert text.count('"formal_training_authorized": false') >= 2


def test_runner_hash_loads_session_and_binds_all_authority_paths():
    text = RUNNER.read_text(encoding="utf-8")
    assert "load_session(session_path, session_sha)" in text
    assert "load_bundle_prerequisites_for_mode(" in text
    assert '"runtime_probe_output": str(Path(runtime_output).resolve())' in text
    assert '"raw_output": str(Path(raw_output).resolve())' in text
    assert '"decision_output": str(Path(decision_output).resolve())' in text
    assert '"state_archive_output": str(Path(state_archive_output).resolve())' in text
    assert '"obs_prefix": obs_prefix' in text
    assert 'RAW_OUTPUT="$RUN_DIR/raw.json"' in text
    assert 'DECISION_OUTPUT="$RUN_DIR/decision.json"' in text
    assert "print(json.load(open(sys.argv[1]" not in text


def test_runner_pins_evaluator_source_and_only_qualifies_via_measure_clis():
    text = RUNNER.read_text(encoding="utf-8")
    assert "EVALUATOR_SOURCE_SHA256=${EVALUATOR_SOURCE_SHA256:?" in text
    assert '[[ "$actual_evaluator_sha" == "$EVALUATOR_SOURCE_SHA256" ]]' in text
    assert 'getattr(os, "O_NOFOLLOW"' in text
    assert 'QUALIFICATION_MODE=${QUALIFICATION_MODE:-bundle}' in text
    assert "--measure-delta1-oracle" in text
    assert "--measure-runtime" in text
    assert "--produce-oracle-decision" not in text
    assert "--produce-runtime-decision" not in text
    assert text.count('[[ ! -e "$RAW_OUTPUT" && ! -e "$DECISION_OUTPUT" ]]') == 2
    assert "load_bundle_prerequisites_for_mode(" in text
    assert 'None if runtime is None else runtime["sha256"]' in text


def test_untouched_global_once_requires_conditional_create_and_readback():
    text = RUNNER.read_text(encoding="utf-8")
    evaluator = (ROOT / "scripts" / "contracted_scientific_bundle_eval.py").read_text(
        encoding="utf-8"
    )
    assert "Untouched global conditional-create is enforced inside the evaluator core" in text
    assert "establish_global_obs_claim(protocol, session)" in evaluator
    assert "OBS_CONDITIONAL_CREATE_HELPER_SHA256" in evaluator
    assert 'protocol["untouched_global_claim"]["helper_sha256"]' in evaluator
    assert "global_claim_readback_path" in evaluator
    assert "payload_raw = (json.dumps(payload" in evaluator
    assert "subprocess.run" in evaluator
    assert "write_text(" not in text
