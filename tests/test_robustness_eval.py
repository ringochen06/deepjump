from scripts.robustness_eval import _paired_domain_bootstrap, _summarize


def test_summarize_reports_sample_metrics_and_domain_wins():
    rows = [
        {"domain": "a", "rmsd": 1.0, "pdmae": 0.4, "fnc": 0.8},
        {"domain": "a", "rmsd": 3.0, "pdmae": 0.6, "fnc": 0.6},
        {"domain": "b", "rmsd": 4.0, "pdmae": 1.0, "fnc": 0.5},
    ]
    summary = _summarize(rows, {"a": 2.5, "b": 3.5})

    assert summary["rmsd"] == 8 / 3
    assert summary["pdmae"] == 2 / 3
    assert summary["fnc"] == 1.9 / 3
    assert summary["domains_better_than_noop"] == 1
    assert summary["domain_count"] == 2


def test_paired_bootstrap_reports_positive_gain_for_better_model():
    rows = [
        {"domain": "a", "rmsd": 1.0},
        {"domain": "a", "rmsd": 2.0},
        {"domain": "b", "rmsd": 3.0},
    ]
    out = _paired_domain_bootstrap(rows, {"a": 2.0, "b": 4.0}, draws=1000)
    assert out["mean_noop_minus_model"] == 0.75
    assert out["ci95"][0] > 0
