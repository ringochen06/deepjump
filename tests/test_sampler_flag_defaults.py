"""No CLI may quietly disable the V re-masking invariant.

`sample()` defaults `project_v_atom_mask=True` because the zero-padding invariant
is a correctness property: without it the endpoint ODE writes into V's structural
padding, the values re-enter the conditioner, and geometry compounds apart
(measured 3.8 -> 40.8 A CA-CA bond by 150 Euler steps on the formal 500k
checkpoint).

An `action="store_true"` flag defaults to False and is passed positively into the
sampler, so it does not merely fail to enable re-masking -- it actively overrides
the fixed default on every invocation that omits it. Three evaluators shipped
exactly that, and their results fed conclusions in STATUS.md. These tests make the
pattern fail loudly rather than silently.
"""

import ast
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FLAG = "--project-v-atom-mask"
# Adjudicators assert what a *past* run recorded, so they must keep the historical
# value and are not sampler entry points. Matching by prefix rather than listing
# names keeps a newly added adjudicator from being treated as a bypass.
ADJUDICATOR_PREFIX = "adjudicate_"


def _add_argument_calls(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "add_argument":
            yield node


def _flag_definitions():
    """Every add_argument() that defines the re-masking flag, with its source."""
    for path in sorted(SCRIPTS.glob("*.py")):
        if path.name.startswith(ADJUDICATOR_PREFIX):
            continue
        source = path.read_text()
        if FLAG not in source:
            continue
        tree = ast.parse(source)
        for call in _add_argument_calls(tree):
            literals = [a.value for a in call.args if isinstance(a, ast.Constant)]
            if FLAG in literals:
                yield path.name, call


def test_at_least_one_entry_point_defines_the_flag():
    """Guard the guard: if nothing matches, the tests below are vacuous."""
    assert list(_flag_definitions()), "no sampler entry point defines the flag"


@pytest.mark.parametrize("name,call", list(_flag_definitions()), ids=lambda v: getattr(v, "__str__", lambda: v)())
def test_flag_is_not_store_true(name, call):
    """store_true silently passes False and overrides the fixed default."""
    for keyword in call.keywords:
        if keyword.arg == "action":
            value = keyword.value
            is_store_true = isinstance(value, ast.Constant) and value.value == "store_true"
            assert not is_store_true, (
                f"{name}: {FLAG} uses store_true, which passes False into the sampler "
                "on every invocation that omits it"
            )


@pytest.mark.parametrize("name,call", list(_flag_definitions()), ids=lambda v: getattr(v, "__str__", lambda: v)())
def test_flag_defaults_to_enabled(name, call):
    defaults = [k.value for k in call.keywords if k.arg == "default"]
    assert defaults, f"{name}: {FLAG} must state an explicit default"
    default = defaults[0]
    assert isinstance(default, ast.Constant) and default.value is True, (
        f"{name}: {FLAG} must default to True so the invariant holds unless opted out"
    )


def test_adjudicators_still_pin_the_historical_value():
    """Past runs recorded project_v_atom_mask=False; that record must not be rewritten.

    Those runs really did execute with re-masking off, which is why their geometry
    counts are not evidence about the model. Rewriting the expectation to True
    would make an adjudicator pass against a run that never happened.
    """
    # adjudicate_v_mask_projection is the paired discriminator for this very flag:
    # it runs projected=False and projected=True against each other, so its
    # expectation is a parameter by design rather than a pinned historical value.
    parameterised = {"adjudicate_v_mask_projection.py"}
    pinned = [
        path.name for path in SCRIPTS.glob(f"{ADJUDICATOR_PREFIX}*.py")
        if '"project_v_atom_mask"' in path.read_text() and path.name not in parameterised
    ]
    assert pinned, "no adjudicator pins the sampler setting any more"
    for name in pinned:
        assert '"project_v_atom_mask": False' in (SCRIPTS / name).read_text(), name
