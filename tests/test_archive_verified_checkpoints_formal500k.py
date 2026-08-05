import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path


ROOT = Path(__file__).parents[1]
ARCHIVER = ROOT / "cloud/huawei/archive_verified_checkpoints_formal500k.sh"


def _fake_obsutil(tmp_path: Path) -> Path:
    executable = tmp_path / "obsutil"
    executable.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            command=$1
            shift
            map_path() {
              case "$1" in
                obs://*) printf '%s/%s' "$FAKE_OBS_ROOT" "${1#obs://}" ;;
                *) printf '%s' "$1" ;;
              esac
            }
            if [ "$command" = stat ]; then
              [ -f "$(map_path "$1")" ] || {
                echo 'StatusCode: 404 NoSuchKey' >&2
                exit 1
              }
              exit 0
            fi
            [ "$command" = cp ] || exit 90
            source=$(map_path "$1")
            destination=$(map_path "$2")
            mkdir -p "$(dirname "$destination")"
            cp "$source" "$destination"
            printf 'obs_cp %s %s\\n' "$1" "$2" >> "$EVENT_LOG"
            if [ "${FAKE_CORRUPT_CHECKPOINT_READBACK:-0}" = 1 ] &&
               [[ "$1" = obs://*/checkpoints/ckpt_*.pt ]]; then
              printf corrupt >> "$destination"
            fi
            """
        )
    )
    executable.chmod(0o755)
    return executable


def _fake_validator(tmp_path: Path) -> Path:
    validator = tmp_path / "validator.py"
    validator.write_text(
        textwrap.dedent(
            """\
            import argparse
            import json
            import os
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--checkpoint", required=True)
            parser.add_argument("--history", required=True)
            parser.add_argument("--expected-step", required=True, type=int)
            parser.add_argument("--history-mode", required=True)
            parser.add_argument("--output", required=True)
            args, _ = parser.parse_known_args()
            history = json.loads(Path(args.history).read_text())
            if args.history_mode == "final":
                if not history or history[-1].get("step") != args.expected_step:
                    raise SystemExit("history does not end at expected step")
            elif any(record.get("step", 0) > args.expected_step for record in history):
                raise SystemExit("history contains a future record")
            with open(os.environ["EVENT_LOG"], "a") as handle:
                handle.write(f"validate {args.expected_step} {args.checkpoint}\\n")
            Path(args.output).write_text(json.dumps({
                "status": "PASS",
                "checkpoint_step": args.expected_step,
            }) + "\\n")
            """
        )
    )
    return validator


def _environment(tmp_path: Path, run_dir: Path, *, keep: int = 3) -> dict[str, str]:
    obs_root = tmp_path / "obs"
    config = tmp_path / "formal.yaml"
    config.write_text("train: {max_steps: 500000}\\n")
    verification = tmp_path / "contract-verification.json"
    verification.write_text('{"status":"PASS_FULL_TRAINING_DATA_CONTRACT"}\\n')
    event_log = tmp_path / "events.log"
    event_log.touch(exist_ok=True)
    env = os.environ.copy()
    if shutil.which("flock") is None:
        fake_bin = tmp_path / "fake-bin"
        fake_bin.mkdir(exist_ok=True)
        fake_flock = fake_bin / "flock"
        fake_flock.write_text("#!/usr/bin/env bash\nexit 0\n")
        fake_flock.chmod(0o755)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
    obsutil = _fake_obsutil(tmp_path)
    python = Path(sys.executable)
    env.update(
        {
            "RUN_DIR": str(run_dir),
            "OBS_DST": "obs://bucket/formal/run-1",
            "FORMAL_CONFIG": str(config),
            "CONTRACT_VERIFICATION": str(verification),
            "CONTRACT_VERIFICATION_SHA256": "a" * 64,
            "VALIDATOR": str(_fake_validator(tmp_path)),
            "OBSUTIL": str(obsutil),
            "OBSUTIL_SHA256": hashlib.sha256(obsutil.read_bytes()).hexdigest(),
            "PYTHON": str(python),
            "PYTHON_SHA256": hashlib.sha256(python.read_bytes()).hexdigest(),
            "FAKE_OBS_ROOT": str(obs_root),
            "EVENT_LOG": str(event_log),
            "ARCHIVE_ONCE": "1",
            "KEEP_LOCAL_VERIFIED": str(keep),
        }
    )
    return env


def _write_checkpoint(run_dir: Path, step: int) -> None:
    (run_dir / f"ckpt_{step}.pt").write_bytes(f"checkpoint-{step}".encode())
    history = []
    for existing in sorted(run_dir.glob("ckpt_*.pt")):
        value = int(existing.stem.split("_")[1])
        history.append(
            {"step": value, "val_loss": 1.0, "val_rmsd": 2.0, "noop_rmsd": 3.0}
        )
    (run_dir / "history.json").write_text(json.dumps(history))


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(ARCHIVER)], env=env, text=True, capture_output=True
    )


def test_archiver_orders_strict_validation_before_marker_and_latest(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_checkpoint(run_dir, 10_000)
    env = _environment(tmp_path, run_dir)
    result = _run(env)
    assert result.returncode == 0, result.stderr

    obs_prefix = tmp_path / "obs/bucket/formal/run-1"
    marker = json.loads(
        (obs_prefix / "verified/ckpt_10000.pt.readback.json").read_text()
    )
    assert marker["status"] == "PASS_STRICT_CHECKPOINT_OBS_READBACK"
    assert marker["formal_training_authorized"] is False
    assert marker["resume_semantics"] == "state_consistent_non_bitwise_crop_and_noise"
    assert (obs_prefix / "validation/local_10000.json").is_file()
    assert (obs_prefix / "validation/remote_readback_10000.json").is_file()
    assert (obs_prefix / "LATEST_VERIFIED.json").read_bytes() == (
        obs_prefix / "verified/ckpt_10000.pt.readback.json"
    ).read_bytes()

    events = (tmp_path / "events.log").read_text().splitlines()
    local_validation = next(i for i, line in enumerate(events) if line.startswith("validate 10000"))
    marker_upload = next(
        i for i, line in enumerate(events) if "verified/ckpt_10000.pt.readback.json" in line
    )
    latest_upload = next(
        i for i, line in enumerate(events) if line.endswith("obs://bucket/formal/run-1/LATEST_VERIFIED.json")
    )
    assert local_validation < marker_upload < latest_upload


def test_archiver_publishes_durable_readiness_after_owning_lock(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_checkpoint(run_dir, 1_000)
    env = _environment(tmp_path, run_dir)
    ready = tmp_path / "attempt/archiver.ready"
    ready.parent.mkdir()
    env["ARCHIVER_READY_FILE"] = str(ready)
    env["ARCHIVER_RUN_ID"] = "run-1"
    env["ARCHIVER_ATTEMPT_DIR"] = str(ready.parent)
    result = _run(env)
    assert result.returncode == 0, result.stderr
    payload = json.loads(ready.read_text())
    assert payload["schema"] == "deepjump.formal500k_archiver_ready.v1"
    assert payload["status"] == "ARCHIVER_READY_AFTER_INITIAL_ROUND"
    assert isinstance(payload["pid"], int)
    assert payload["run_id"] == "run-1"
    assert payload["attempt_dir"] == str(ready.parent.resolve())
    assert (tmp_path / "obs/bucket/formal/run-1/LATEST_VERIFIED.json").is_file()


def test_archiver_stale_lock_file_does_not_block_restart(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_checkpoint(run_dir, 1_000)
    state_dir = run_dir / ".formal500k_archive"
    state_dir.mkdir()
    (state_dir / "archiver.lock").write_text("stale inode\n")
    env = _environment(tmp_path, run_dir)
    result = _run(env)
    assert result.returncode == 0, result.stderr


def test_archiver_live_flock_blocks_second_owner(tmp_path):
    if shutil.which("flock") is None:
        import pytest

        pytest.skip("util-linux flock is verified on the Linux target")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_checkpoint(run_dir, 1_000)
    env = _environment(tmp_path, run_dir)
    env["ARCHIVE_ONCE"] = "0"
    env["POLL_SECONDS"] = "60"
    first = subprocess.Popen(
        ["bash", str(ARCHIVER)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        lock_path = run_dir / ".formal500k_archive/archiver.lock"
        deadline = time.monotonic() + 5
        while not lock_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert lock_path.exists()
        second = _run({**env, "ARCHIVE_ONCE": "1"})
        assert second.returncode != 0
        assert "another checkpoint archiver owns" in second.stderr
    finally:
        first.terminate()
        first.wait(timeout=5)


def test_archiver_ready_pid_matches_owned_process_and_term_releases_flock(tmp_path):
    setsid = shutil.which("setsid")
    if shutil.which("flock") is None or setsid is None:
        import pytest

        pytest.skip("util-linux flock/setsid are verified on the Linux target")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_checkpoint(run_dir, 1_000)
    ready = tmp_path / "attempt/archiver.ready"
    ready.parent.mkdir()
    env = _environment(tmp_path, run_dir)
    env.update(
        {
            "ARCHIVE_ONCE": "0",
            "POLL_SECONDS": "60",
            "ARCHIVER_READY_FILE": str(ready),
            "ARCHIVER_RUN_ID": "run-1",
            "ARCHIVER_ATTEMPT_DIR": str(ready.parent),
        }
    )
    fake_bin = tmp_path / "term-ignoring-bin"
    fake_bin.mkdir()
    fake_sleep = fake_bin / "sleep"
    fake_sleep.write_text("#!/usr/bin/env bash\ntrap '' TERM\nexec /bin/sleep \"$@\"\n")
    fake_sleep.chmod(0o755)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    process = subprocess.Popen(
        [setsid, str(ARCHIVER)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), process.stderr.read() if process.poll() is not None else ""
        assert json.loads(ready.read_text())["pid"] == process.pid
        child_deadline = time.monotonic() + 5
        child_pid = ""
        while not child_pid and time.monotonic() < child_deadline:
            child_pid = subprocess.run(
                ["pgrep", "-P", str(process.pid), "sleep"],
                text=True,
                capture_output=True,
                check=False,
            ).stdout.strip()
            if not child_pid:
                time.sleep(0.02)
        assert child_pid, "archiver did not reach its active sleep phase"
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        time.sleep(0.2)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    assert subprocess.run(
        ["pgrep", "-g", str(process.pid)],
        capture_output=True,
        check=False,
    ).returncode != 0

    one_shot = _run({**env, "ARCHIVE_ONCE": "1", "ARCHIVER_READY_FILE": ""})
    assert one_shot.returncode == 0, one_shot.stderr


def test_archiver_processes_only_new_unverified_checkpoints(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_checkpoint(run_dir, 10_000)
    env = _environment(tmp_path, run_dir)
    assert _run(env).returncode == 0
    first_validations = (tmp_path / "events.log").read_text().count("validate 10000")

    _write_checkpoint(run_dir, 20_000)
    result = _run(env)
    assert result.returncode == 0, result.stderr
    events = (tmp_path / "events.log").read_text()
    assert events.count("validate 10000") == first_validations
    assert events.count("validate 20000") == 2


def test_recovering_an_old_local_marker_cannot_regress_latest(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_checkpoint(run_dir, 10_000)
    _write_checkpoint(run_dir, 20_000)
    env = _environment(tmp_path, run_dir)
    assert _run(env).returncode == 0
    (run_dir / ".formal500k_archive/verified/ckpt_10000.pt.readback.json").unlink()

    result = _run(env)
    assert result.returncode == 0, result.stderr
    latest = json.loads(
        (tmp_path / "obs/bucket/formal/run-1/LATEST_VERIFIED.json").read_text()
    )
    assert latest["step"] == 20_000


def test_archiver_accepts_trainer_owned_empty_history_before_first_validation(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_checkpoint(run_dir, 1_000)
    (run_dir / "history.json").write_text("[]\n")
    env = _environment(tmp_path, run_dir)
    result = _run(env)
    assert result.returncode == 0, result.stderr
    archived = (
        tmp_path / "obs/bucket/formal/run-1/history/history_1000.json"
    )
    assert json.loads(archived.read_text()) == []


def test_corrupt_checkpoint_readback_withholds_marker_and_latest(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_checkpoint(run_dir, 10_000)
    env = _environment(tmp_path, run_dir)
    env["FAKE_CORRUPT_CHECKPOINT_READBACK"] = "1"
    result = _run(env)
    assert result.returncode != 0
    assert "readback SHA" in result.stderr or "conflicts" in result.stderr
    obs_prefix = tmp_path / "obs/bucket/formal/run-1"
    assert not (obs_prefix / "verified/ckpt_10000.pt.readback.json").exists()
    assert not (obs_prefix / "LATEST_VERIFIED.json").exists()


def test_local_retention_refuses_without_current_remote_marker_proof(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for step in (10_000, 20_000, 30_000):
        _write_checkpoint(run_dir, step)
    env = _environment(tmp_path, run_dir, keep=3)
    assert _run(env).returncode == 0
    remote_marker = (
        tmp_path / "obs/bucket/formal/run-1/verified/ckpt_10000.pt.readback.json"
    )
    remote_marker.unlink()

    env["KEEP_LOCAL_VERIFIED"] = "1"
    result = _run(env)
    assert result.returncode != 0
    assert "remote marker proof" in result.stderr
    assert (run_dir / "ckpt_10000.pt").is_file()
