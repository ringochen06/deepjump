import ast
from pathlib import Path

import pytest


EVALUATORS_WITH_MODEL_VECTOR_INPUT = (
    "geometry_robustness_eval.py",
    "rollout_eval.py",
    "rollout_robustness_eval.py",
    "tica_eval.py",
    "tica_panel.py",
    "tica_robustness_eval.py",
    "transition_robustness_eval.py",
)


@pytest.mark.parametrize("script_name", EVALUATORS_WITH_MODEL_VECTOR_INPUT)
def test_model_vector_inputs_use_declared_symmetric_canonicalization(script_name):
    source = (Path(__file__).parents[1] / "scripts" / script_name).read_text()
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_model_layout"
    ]

    assert calls, f"{script_name} must route model-fed vectors through apply_model_layout"
    assert all(
        any(keyword.arg == "canon_symmetric" for keyword in call.keywords)
        for call in calls
    )
