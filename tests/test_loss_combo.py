"""Loss-weight composition: confirm w_ca / w_offset / w_allatom combine additively
and that no branch silently re-adds an unweighted Ca term (regression for the bug
where the w_offset branch reset `loss = ca + ...`, dropping the w_ca weight)."""

import torch

from deepjump.config import Config
from deepjump.losses import (
    allatom_pairwise_huber_loss,
    ca_bond_length_huber_loss,
    heavy_atom_offset_loss,
    pairwise_vector_huber_loss,
)
from deepjump.training import _step_loss
from deepjump.training import total_loss


def _toy(B=2, N=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    P_hat = torch.randn(B, N, 3, generator=g)
    P_gt = torch.randn(B, N, 3, generator=g)
    V_hat = torch.randn(B, N, 13, 3, generator=g) * 0.3
    V_gt = torch.randn(B, N, 13, 3, generator=g) * 0.3
    batch = {
        "residue_mask": torch.ones(B, N, dtype=torch.bool),
        "res_index": torch.arange(N).repeat(B, 1),
        "bond_mask": torch.ones(B, N - 1, dtype=torch.bool),
        "atom_mask": torch.ones(B, N, 13, dtype=torch.bool),
    }
    return P_hat, V_hat, P_gt, V_gt, batch


def _cfg(w_ca, w_offset, w_allatom, w_bond=0.0):
    c = Config()
    c.model.dist_cutoff = 25.0
    c.train.huber_delta = 1.0
    c.train.w_ca = w_ca
    c.train.w_bond = w_bond
    c.train.w_offset = w_offset
    c.train.w_allatom = w_allatom
    return c


def test_loss_weight_composition():
    P_hat, V_hat, P_gt, V_gt, batch = _toy()
    ca = pairwise_vector_huber_loss(P_hat, P_gt, batch["residue_mask"], 1.0)
    off = heavy_atom_offset_loss(V_hat, V_gt, batch["atom_mask"], 1.0)
    aa = allatom_pairwise_huber_loss(
        P_hat, V_hat, P_gt, V_gt, batch["atom_mask"], batch["residue_mask"], 25.0, 1.0
    )
    bond = ca_bond_length_huber_loss(
        P_hat, P_gt, batch["residue_mask"], batch["bond_mask"], 1.0
    )
    # the three components must be distinct so the assertions below are meaningful
    assert not torch.allclose(ca, off) and not torch.allclose(ca, aa)

    def step(w_ca, w_offset, w_allatom, w_bond=0.0):
        return _step_loss(
            P_hat, V_hat, P_gt, V_gt, batch, _cfg(w_ca, w_offset, w_allatom, w_bond)
        )[0]

    # 1) default w_ca=1 keeps the legacy Ca-only behaviour
    assert torch.allclose(step(1.0, 0.0, 0.0), ca)

    # 2) w_ca=0, w_offset=0, w_allatom=1 -> all-atom loss ONLY (no Ca term)
    assert torch.allclose(step(0.0, 0.0, 1.0), aa)

    # 3) w_ca=0, w_offset=1 -> offset ONLY; must NOT sneak in an unweighted Ca term
    l = step(0.0, 1.0, 0.0)
    assert torch.allclose(l, off)
    assert not torch.allclose(l, ca + off)  # the exact bug this guards against

    # 4) weights are additive across all three branches
    assert torch.allclose(step(1.0, 1.0, 0.0), ca + off)
    assert torch.allclose(step(0.5, 0.0, 2.0), 0.5 * ca + 2.0 * aa)
    assert torch.allclose(step(0.0, 0.0, 1.0, 3.0), aa + 3.0 * bond)

    # An explicit unroll-only override must not depend on the step-1 config weight.
    override = _step_loss(
        P_hat, V_hat, P_gt, V_gt, batch, _cfg(0.0, 0.0, 1.0, 0.0),
        bond_weight=0.25,
    )[0]
    assert torch.allclose(override, aa + 0.25 * bond)


def test_paper_config_loads():
    from pathlib import Path

    from deepjump.config import load_config

    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "v100_paper_d1.yaml")
    assert cfg.data.crop_length == 256
    assert cfg.train.batch_size == 16 and cfg.train.grad_accum == 1
    assert cfg.train.amp_dtype == "fp16"
    assert cfg.train.w_ca == 0.0 and cfg.train.w_offset == 0.0 and cfg.train.w_allatom == 1.0


def test_total_loss_reports_allatom_component():
    P_hat, V_hat, P_gt, V_gt, batch = _toy()
    cfg = _cfg(0.0, 0.0, 1.0)
    out = {"P_hat_1": P_hat, "V_hat_1": V_hat}

    loss, components = total_loss(out, {**batch, "P_1": P_gt, "V_1": V_gt}, cfg)

    assert torch.isfinite(loss)
    assert components["allatom"] > 0
    assert components["ca"] > 0
