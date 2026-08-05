from scripts.paper_sampler_mechanism_diagnostic import (
    EXPECTED_CELLS_PER_DOMAIN,
    EXPECTED_DOMAIN_IDS,
    EXPECTED_STARTS,
    diagnostic_decision,
)


def _summary(*, fallback_starts, nonfinite_starts, raw_delta):
    return {
        "domains": len(EXPECTED_DOMAIN_IDS),
        "cells": len(EXPECTED_DOMAIN_IDS) * EXPECTED_CELLS_PER_DOMAIN,
        "starts": (
            len(EXPECTED_DOMAIN_IDS)
            * EXPECTED_CELLS_PER_DOMAIN
            * EXPECTED_STARTS
        ),
        "fallback_cells": int(fallback_starts > 0),
        "fallback_starts": fallback_starts,
        "nonfinite_starts": nonfinite_starts,
        "mean_raw_minus_noop": raw_delta,
        "mean_guarded_minus_noop": raw_delta,
    }


def test_diagnostic_support_requires_clean_ode20_and_mean_reproduction():
    decision = diagnostic_decision({
        "mean": _summary(
            fallback_starts=75, nonfinite_starts=0, raw_delta=-0.02
        ),
        "ode_20": _summary(
            fallback_starts=0, nonfinite_starts=0, raw_delta=-0.03
        ),
    })
    assert decision["status"] == "SUPPORT_FIXED_ODE20_MECHANISM"
    assert decision["formal_training_authorized"] is False
    assert decision["external_authorized"] is False
    assert decision["untouched_authorized"] is False
    assert decision["ode_drift_anchor"] == "state"


def test_diagnostic_stops_on_any_ode20_fallback():
    decision = diagnostic_decision({
        "mean": _summary(
            fallback_starts=75, nonfinite_starts=0, raw_delta=-0.02
        ),
        "ode_20": _summary(
            fallback_starts=1, nonfinite_starts=0, raw_delta=-0.03
        ),
    })
    assert decision["status"] == "STOP_FIXED_ODE20_MECHANISM"


def test_diagnostic_stops_when_grid_is_incomplete():
    mean = _summary(fallback_starts=75, nonfinite_starts=0, raw_delta=-0.02)
    mean["cells"] -= 1
    decision = diagnostic_decision({
        "mean": mean,
        "ode_20": _summary(
            fallback_starts=0, nonfinite_starts=0, raw_delta=-0.03
        ),
    })
    assert decision["status"] == "STOP_FIXED_ODE20_MECHANISM"


def test_literal_paper_conditioner_anchor_support_is_separate_and_non_authorizing():
    decision = diagnostic_decision(
        {
            "mean": _summary(
                fallback_starts=75, nonfinite_starts=0, raw_delta=-0.02
            ),
            "ode_20": _summary(
                fallback_starts=0, nonfinite_starts=0, raw_delta=-0.03
            ),
        },
        ode_drift_anchor="conditioner",
    )
    assert (
        decision["status"]
        == "SUPPORT_LITERAL_PAPER_CONDITIONER_ODE20_MECHANISM"
    )
    assert decision["ode_drift_anchor"] == "conditioner"
    assert decision["formal_training_authorized"] is False
    assert decision["external_authorized"] is False
    assert decision["untouched_authorized"] is False


def test_literal_paper_conditioner_anchor_stops_on_worse_raw_mean():
    decision = diagnostic_decision(
        {
            "mean": _summary(
                fallback_starts=75, nonfinite_starts=0, raw_delta=-0.02
            ),
            "ode_20": _summary(
                fallback_starts=0, nonfinite_starts=0, raw_delta=0.01
            ),
        },
        ode_drift_anchor="conditioner",
    )
    assert (
        decision["status"]
        == "STOP_LITERAL_PAPER_CONDITIONER_ODE20_MECHANISM"
    )
