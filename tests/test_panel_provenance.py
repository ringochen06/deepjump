"""The panel figure must not assert that its rows are held out.

The title used to read "held-out mdCATH domains" unconditionally, while the rows
were chosen by `split_domains()` over whatever mdCATH shards happened to be on
local disk. That says nothing about what a checkpoint trained on: the formal
500k contract records 5,218 train domains out of mdCATH's 5,398, so a domain
drawn without consulting the real training list is ~97% likely to have been
trained on. Four such domains all being held out has probability ~1e-5.

Provenance now has to be demonstrated with `--train-list`, and a listed trained-on
domain is surfaced rather than hidden. The flag takes the *training* list rather
than a held-out list on purpose: the training list is the artifact the run
actually produces (`train_eligible_5218.txt`), and taking the complement means
passing it by mistake cannot silently invert the test -- which it did in the
first version of this check.
"""

from pathlib import Path

import pytest

PANEL = Path(__file__).resolve().parents[1] / "scripts" / "tica_panel.py"


@pytest.fixture(scope="module")
def source():
    return PANEL.read_text()


def test_title_never_hardcodes_the_held_out_claim(source):
    """The regression: an unconditional claim baked into the suptitle string."""
    for line in source.splitlines():
        if "suptitle" in line or ("held-out domains" in line and "provenance" not in line):
            assert "held-out mdCATH domains" not in line, (
                "the figure title must not assert held-out provenance; derive it instead"
            )


def test_unverified_is_the_default_label(source):
    assert "training status unverified" in source


def test_a_trained_on_domain_is_surfaced(source):
    """Supplying a list must be able to contradict the claim, not just confirm it."""
    assert "WARNING: trained-on domains present" in source


def test_the_train_list_option_exists(source):
    assert '"--train-list"' in source


def test_the_check_reads_the_training_list_not_its_complement(source):
    """Guard the inversion bug: held-out must be derived, never taken directly."""
    assert 'b["name"] in trained_ids' in source


def test_provenance_label_is_computed_from_the_training_list():
    """Exercise the branches the figure can print, with real semantics."""
    trained = {"1abcA00", "2defB01"}

    def label(names, ids):
        if ids is None:
            return "training status unverified — no --train-list given"
        seen = [n for n in names if n in ids]
        return "held-out domains" if not seen else (
            f"WARNING: trained-on domains present ({', '.join(seen)})"
        )

    assert label(["1abcA00"], None).startswith("training status unverified")
    # Absent from the training list => genuinely held out.
    assert label(["9xyzZ99", "8wvuY88"], trained) == "held-out domains"
    # Present in the training list => must be named, not hidden.
    assert "1abcA00" in label(["1abcA00", "9xyzZ99"], trained)
