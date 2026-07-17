"""Fail-closed helpers for reproducible DeepJump evaluation panels."""

from __future__ import annotations

import hashlib
import hmac
from numbers import Integral
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def domain_id(path: str | Path) -> str:
    """Return the mdCATH domain identifier encoded in a dataset filename."""
    name = Path(path).name
    prefix = "mdcath_dataset_"
    suffix = ".h5"
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"not an mdCATH dataset filename: {name}")
    value = name[len(prefix):-len(suffix)]
    if not value:
        raise ValueError(f"empty mdCATH domain identifier in {name}")
    return value


def require_single_delta(value: object) -> int:
    """Return one positive integer delta, rejecting silent multi-delta mixing."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                "evaluation requires exactly one delta_frames value; "
                f"got {value!r}. Evaluate each delta in a separate report."
            )
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"delta_frames must be a positive integer, got {value!r}")
    delta = int(value)
    if delta <= 0:
        raise ValueError(f"delta_frames must be positive, got {delta}")
    return delta


def load_frozen_domain_ids(path: str | Path, expected_sha256: str) -> tuple[list[str], str]:
    """Load an exact domain panel and verify its byte-level SHA256 identity."""
    domain_path = Path(path).expanduser()
    raw = domain_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("expected domain-list SHA256 must be 64 lowercase hex characters")
    if not hmac.compare_digest(digest, expected):
        raise ValueError(
            f"domain-list SHA256 mismatch for {domain_path}: {digest} != {expected}"
        )
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"domain list is not UTF-8: {domain_path}") from exc
    ids = [line.strip() for line in lines if line.strip()]
    if not ids:
        raise ValueError(f"domain list is empty: {domain_path}")
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"domain list contains duplicate identifiers: {duplicates}")
    return ids, digest


def resolve_frozen_domains(
    discovered: Iterable[str | Path], domain_ids: Sequence[str]
) -> list[Path]:
    """Resolve a frozen ordered ID list against discovered HDF5 files."""
    by_id: dict[str, Path] = {}
    duplicate_files: list[str] = []
    for candidate in discovered:
        path = Path(candidate)
        identifier = domain_id(path)
        if identifier in by_id:
            duplicate_files.append(identifier)
        by_id[identifier] = path
    if duplicate_files:
        raise ValueError(
            "multiple HDF5 files resolve to the same domain identifiers: "
            f"{sorted(set(duplicate_files))}"
        )
    missing = [identifier for identifier in domain_ids if identifier not in by_id]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} frozen evaluation domains are missing: {missing[:10]}"
        )
    return [by_id[identifier] for identifier in domain_ids]


def reference_transition_deltas(values, delta: int):
    """Return observed increments separated by the checkpoint's physical delta."""
    delta = require_single_delta(delta)
    if len(values) <= delta:
        raise ValueError(
            f"need more than {delta} reference frames for delta={delta}; got {len(values)}"
        )
    return values[delta:] - values[:-delta]


def fit_kmeans(
    values: np.ndarray,
    n_clusters: int,
    *,
    seed: int = 0,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit deterministic k-means++ without adding a heavy evaluation dependency."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("k-means values must be a finite 2D array")
    if not 1 <= n_clusters <= len(values):
        raise ValueError("n_clusters must be in [1, n_samples]")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")

    rng = np.random.default_rng(seed)
    centers = np.empty((n_clusters, values.shape[1]), dtype=np.float64)
    chosen = np.zeros(len(values), dtype=bool)
    first = int(rng.integers(len(values)))
    centers[0] = values[first]
    chosen[first] = True
    closest_sq = ((values - centers[0]) ** 2).sum(axis=1)
    for index in range(1, n_clusters):
        weights = closest_sq.copy()
        weights[chosen] = 0.0
        total = float(weights.sum())
        if total > 0:
            candidate = int(rng.choice(len(values), p=weights / total))
        else:
            candidate = int(np.flatnonzero(~chosen)[0])
        centers[index] = values[candidate]
        chosen[candidate] = True
        closest_sq = np.minimum(closest_sq, ((values - centers[index]) ** 2).sum(axis=1))

    labels = assign_clusters(values, centers)
    for _ in range(max_iter):
        updated = centers.copy()
        distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        nearest_distance = distances[np.arange(len(values)), labels]
        reserved: set[int] = set()
        for index in range(n_clusters):
            members = values[labels == index]
            if len(members):
                updated[index] = members.mean(axis=0)
            else:
                order = np.argsort(nearest_distance)[::-1]
                replacement = next(int(i) for i in order if int(i) not in reserved)
                reserved.add(replacement)
                updated[index] = values[replacement]
        new_labels = assign_clusters(values, updated)
        centers = updated
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
    return centers, labels


def assign_clusters(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign each row to its nearest Euclidean cluster center."""
    values = np.asarray(values, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    if values.ndim != 2 or centers.ndim != 2 or values.shape[1] != centers.shape[1]:
        raise ValueError("values and centers must be compatible 2D arrays")
    distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def transition_matrix(
    origin: np.ndarray,
    destination: np.ndarray | None = None,
    *,
    n_states: int,
    lag: int = 1,
    pseudocount: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a row-stochastic transition matrix and raw origin counts."""
    origin = np.asarray(origin, dtype=np.int64)
    if destination is None:
        if lag < 1 or lag >= len(origin):
            raise ValueError("lag must be in [1, n_labels)")
        destination = origin[lag:]
        origin = origin[:-lag]
    else:
        destination = np.asarray(destination, dtype=np.int64)
        if origin.shape != destination.shape:
            raise ValueError("origin and destination labels must have the same shape")
    if n_states < 1 or pseudocount < 0:
        raise ValueError("n_states must be positive and pseudocount non-negative")
    if len(origin) == 0:
        raise ValueError("at least one transition is required")
    if (
        (origin < 0).any() or (destination < 0).any()
        or (origin >= n_states).any() or (destination >= n_states).any()
    ):
        raise ValueError("transition labels are outside [0, n_states)")
    counts = np.full((n_states, n_states), float(pseudocount), dtype=np.float64)
    np.add.at(counts, (origin, destination), 1.0)
    origin_counts = np.bincount(origin, minlength=n_states).astype(np.float64)
    row_sum = counts.sum(axis=1, keepdims=True)
    zero_rows = row_sum[:, 0] == 0
    if zero_rows.any():
        counts[zero_rows] = np.eye(n_states, dtype=np.float64)[zero_rows]
        row_sum = counts.sum(axis=1, keepdims=True)
    return counts / row_sum, origin_counts


def jsd_bits(left: np.ndarray, right: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits for two categorical distributions."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("JSD inputs must be 1D arrays with matching shapes")
    if (left < 0).any() or (right < 0).any():
        raise ValueError("JSD inputs must be non-negative")
    left = left / left.sum()
    right = right / right.sum()
    middle = 0.5 * (left + right)

    def kl_bits(values: np.ndarray) -> float:
        nonzero = values > 0
        return float(np.sum(values[nonzero] * np.log2(values[nonzero] / middle[nonzero])))

    return 0.5 * (kl_bits(left) + kl_bits(right))


def weighted_row_jsd_bits(
    reference: np.ndarray, sample: np.ndarray, row_weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """Compare transition rows using a shared observed-origin weighting."""
    reference = np.asarray(reference, dtype=np.float64)
    sample = np.asarray(sample, dtype=np.float64)
    row_weights = np.asarray(row_weights, dtype=np.float64)
    if reference.shape != sample.shape or reference.ndim != 2:
        raise ValueError("transition matrices must be matching 2D arrays")
    if reference.shape[0] != reference.shape[1] or row_weights.shape != (reference.shape[0],):
        raise ValueError("transition matrices must be square and weights must match rows")
    if (row_weights < 0).any() or row_weights.sum() <= 0:
        raise ValueError("row weights must be non-negative with positive total")
    rows = np.asarray(
        [jsd_bits(reference[index], sample[index]) for index in range(len(reference))]
    )
    return float(np.average(rows, weights=row_weights)), rows


def paired_domain_bootstrap_gain(
    model: np.ndarray,
    baseline: np.ndarray,
    *,
    draws: int = 10000,
    seed: int = 0,
) -> dict[str, object]:
    """Bootstrap domain-balanced baseline-minus-model gains; positive is better."""
    model = np.asarray(model, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if model.shape != baseline.shape or model.ndim != 1 or len(model) == 0:
        raise ValueError("model and baseline must be matching non-empty 1D arrays")
    if not np.isfinite(model).all() or not np.isfinite(baseline).all() or draws < 2:
        raise ValueError("bootstrap inputs must be finite and draws at least two")
    gains = baseline - model
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(gains), size=(draws, len(gains)))
    samples = gains[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "mean_baseline_minus_model": float(gains.mean()),
        "ci95": [float(low), float(high)],
        "domains": len(gains),
        "passes": bool(low > 0),
    }


def geometry_frame_statistics(
    positions: np.ndarray,
    bond_mask: np.ndarray,
    *,
    collision_distance: float = 2.5,
) -> dict[str, np.ndarray]:
    """Compute topology-aware, geometry-only statistics for each CA frame."""
    positions = np.asarray(positions, dtype=np.float64)
    bond_mask = np.asarray(bond_mask, dtype=bool)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [frames, residues, 3]")
    if bond_mask.shape != (positions.shape[1] - 1,):
        raise ValueError("bond_mask must have one entry per adjacent residue pair")
    if not np.isfinite(positions).all() or collision_distance <= 0:
        raise ValueError("positions must be finite and collision_distance positive")
    if not bond_mask.any():
        raise ValueError("at least one topology-valid bond is required")

    bonds = positions[:, 1:] - positions[:, :-1]
    lengths = np.linalg.norm(bonds, axis=-1)
    valid_lengths = lengths[:, bond_mask]
    valid_angles = bond_mask[:-1] & bond_mask[1:]
    if not valid_angles.any():
        raise ValueError("at least one topology-valid bond angle is required")
    unit = bonds / np.clip(lengths[..., None], 1e-12, None)
    angle_cos = (unit[:, :-1] * unit[:, 1:]).sum(axis=-1)[:, valid_angles]

    i, j = np.triu_indices(positions.shape[1], k=3)
    nonbonded = np.ones(len(i), dtype=bool)
    if not nonbonded.any():
        raise ValueError("at least one non-bonded CA pair is required")
    distances = np.linalg.norm(positions[:, i] - positions[:, j], axis=-1)[:, nonbonded]
    return {
        "bond_mean": valid_lengths.mean(axis=1),
        "bond_p99": np.quantile(valid_lengths, 0.99, axis=1),
        "bond_max": valid_lengths.max(axis=1),
        "angle_cos_mean": angle_cos.mean(axis=1),
        "angle_cos_p01": np.quantile(angle_cos, 0.01, axis=1),
        "angle_cos_p99": np.quantile(angle_cos, 0.99, axis=1),
        "collision_fraction": (distances < collision_distance).mean(axis=1),
    }


def aggregate_geometry_panel(statistics: dict[str, np.ndarray]) -> dict[str, float]:
    """Aggregate one panel while preserving the single-frame bond-max tail."""
    if not statistics:
        raise ValueError("geometry statistics cannot be empty")
    result = {}
    for name, values in statistics.items():
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError(f"geometry statistic {name} must be a non-empty finite vector")
        result[name] = float(values.max() if name == "bond_max" else values.mean())
    return result


def calibrate_geometry_envelope(
    statistics: dict[str, np.ndarray],
    panel_size: int,
    *,
    draws: int = 2000,
    alpha: float = 0.01,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Bootstrap a real-vs-real panel envelope for geometry-only rollouts."""
    if panel_size < 1 or draws < 2 or not 0 < alpha < 1:
        raise ValueError("panel_size/draws must be positive and alpha must be in (0, 1)")
    lengths = {len(np.asarray(values)) for values in statistics.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
        raise ValueError("all geometry statistics need the same length of at least two")
    count = next(iter(lengths))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, count, size=(draws, panel_size))
    envelope = {}
    for name, values in statistics.items():
        values = np.asarray(values, dtype=np.float64)
        panels = values[indices]
        aggregate = panels.max(axis=1) if name == "bond_max" else panels.mean(axis=1)
        low, high = np.quantile(aggregate, [alpha / 2, 1 - alpha / 2])
        envelope[name] = {
            "low": float(low),
            "high": float(high),
            "reference": float(values.max() if name == "bond_max" else values.mean()),
        }
    return envelope


def geometry_panel_passes(
    panel: dict[str, float], envelope: dict[str, dict[str, float]]
) -> tuple[bool, dict[str, bool]]:
    """Check one model panel against the frozen real-vs-real envelope."""
    if panel.keys() != envelope.keys():
        raise ValueError("panel and envelope metrics must match exactly")
    checks = {
        name: bool(bounds["low"] <= panel[name] <= bounds["high"])
        for name, bounds in envelope.items()
    }
    return all(checks.values()), checks
