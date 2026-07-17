from scripts.select_subset import LENGTH_BANDS, length_band_quotas, pick_length_proportional


def _fake_residues(per_band=300):
    residues = {}
    for band_index, (_, lo, hi, _) in enumerate(LENGTH_BANDS):
        length = max(lo, 1) if hi >= 10**9 else (lo + hi) // 2
        for index in range(per_band):
            residues[f"b{band_index}_{index:03d}"] = length
    return residues


def test_length_band_quotas_scale_to_confirmation_100():
    assert length_band_quotas(100) == [11, 26, 21, 14, 9, 6, 4, 9]
    assert sum(length_band_quotas(37)) == 37


def test_length_proportional_exclusion_is_deterministic_and_disjoint():
    residues = _fake_residues()
    excluded = {"b0_000", "b1_001", "b7_002"}
    first = pick_length_proportional(20260717, 100, excluded, residues)
    second = pick_length_proportional(20260717, 100, excluded, residues)
    assert first == second
    assert len(first) == len(set(first)) == 100
    assert not (set(first) & excluded)
