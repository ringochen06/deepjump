import pytest
import torch

from deepjump.config import ModelConfig
from deepjump.model import DeepJumpLite


class PerfectEndpointModel(DeepJumpLite):
    def encode(self, batch):
        return None

    def predict_x1(self, P_tau, V_tau, tau, ctx, P_t, V_t, residue_mask):
        P_1 = torch.full_like(P_tau, 2.0)
        V_1 = torch.full_like(V_tau, 3.0) if self.predict_heavy else None
        return P_1, V_1


class IdentityEndpointModel(DeepJumpLite):
    def encode(self, batch):
        return None

    def predict_x1(self, P_tau, V_tau, tau, ctx, P_t, V_t, residue_mask):
        return P_tau, V_tau


def _batch():
    return {
        "P_t": torch.zeros(2, 4, 3),
        "V_t": torch.zeros(2, 4, 13, 3),
        "residue_mask": torch.ones(2, 4, dtype=torch.bool),
        "atom_mask": torch.ones(2, 4, 13, dtype=torch.bool),
    }


@pytest.mark.parametrize("integrator", ["euler", "heun"])
def test_truncated_integrators_recover_perfect_endpoint(integrator):
    model = PerfectEndpointModel(
        ModelConfig(), noise_sigma=0.0, predict_heavy=True
    ).eval()

    P, V = model.sample(
        _batch(), steps=8, integrator=integrator, tau_max=0.95,
        terminal_denoise=True,
    )

    assert torch.equal(P, torch.full_like(P, 2.0))
    assert torch.equal(V, torch.full_like(V, 3.0))


def test_legacy_euler_defaults_remain_available():
    model = PerfectEndpointModel(ModelConfig(), noise_sigma=0.0).eval()
    P, _ = model.sample(_batch(), steps=4)
    assert torch.allclose(P, torch.full_like(P, 2.0))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"steps": 0}, "steps"),
        ({"integrator": "rk4"}, "integrator"),
        ({"tau_max": 0.0}, "tau_max"),
        ({"integrator": "heun", "tau_max": 1.0}, "tau_max"),
        ({"drift_anchor": "bad"}, "drift_anchor"),
    ],
)
def test_invalid_integrator_options_fail_loudly(kwargs, message):
    model = PerfectEndpointModel(ModelConfig(), noise_sigma=0.0).eval()
    with pytest.raises(ValueError, match=message):
        model.sample(_batch(), **kwargs)


def test_conditioner_anchor_is_finite_away_from_endpoint():
    model = PerfectEndpointModel(ModelConfig(), noise_sigma=0.1, predict_heavy=True).eval()
    P, V = model.sample(
        _batch(), steps=2, tau_max=0.95, terminal_denoise=True,
        drift_anchor="conditioner", generator=torch.Generator().manual_seed(7),
    )
    assert torch.isfinite(P).all() and torch.isfinite(V).all()


def test_source_noise_v_is_configurable_and_backward_compatible():
    batch = _batch()
    tau0 = torch.zeros(2)
    legacy = DeepJumpLite(ModelConfig(source_noise_v=False), noise_sigma=0.1)
    _, legacy_v = legacy.interpolate(
        batch["P_t"], batch["V_t"], batch["P_t"], batch["V_t"], tau0,
        torch.Generator().manual_seed(3),
    )
    joint = DeepJumpLite(ModelConfig(source_noise_v=True), noise_sigma=0.1)
    _, joint_v = joint.interpolate(
        batch["P_t"], batch["V_t"], batch["P_t"], batch["V_t"], tau0,
        torch.Generator().manual_seed(3), atom_mask=batch["atom_mask"],
    )

    assert torch.equal(legacy_v, batch["V_t"])
    assert not torch.equal(joint_v, batch["V_t"])
    assert torch.isfinite(joint_v).all()


def test_source_noise_v_supports_an_independent_first_party_scale():
    batch = _batch()
    tau0 = torch.zeros(2)
    shared = DeepJumpLite(
        ModelConfig(source_noise_v=True), noise_sigma=1.5
    )
    separate = DeepJumpLite(
        ModelConfig(source_noise_v=True, source_noise_sigma_v=1.0), noise_sigma=1.5
    )
    _, shared_v = shared.interpolate(
        batch["P_t"], batch["V_t"], batch["P_t"], batch["V_t"], tau0,
        torch.Generator().manual_seed(13), atom_mask=batch["atom_mask"],
    )
    _, separate_v = separate.interpolate(
        batch["P_t"], batch["V_t"], batch["P_t"], batch["V_t"], tau0,
        torch.Generator().manual_seed(13), atom_mask=batch["atom_mask"],
    )

    shared_delta = shared_v - batch["V_t"]
    separate_delta = separate_v - batch["V_t"]
    assert torch.allclose(shared_delta, 1.5 * separate_delta, atol=1e-6, rtol=1e-6)


def test_legacy_vector_source_sigma_tracks_runtime_coordinate_override():
    model = DeepJumpLite(ModelConfig(source_noise_v=True), noise_sigma=0.1)
    assert model._vector_source_noise_sigma() == 0.1
    model.noise_sigma = 0.0
    assert model._vector_source_noise_sigma() == 0.0


def test_sampling_uses_the_independent_vector_source_scale():
    batch = _batch()
    unit = IdentityEndpointModel(
        ModelConfig(source_noise_v=True, source_noise_sigma_v=1.0),
        noise_sigma=1.5, predict_heavy=True,
    ).eval()
    half = IdentityEndpointModel(
        ModelConfig(source_noise_v=True, source_noise_sigma_v=0.5),
        noise_sigma=1.5, predict_heavy=True,
    ).eval()
    P_unit, V_unit = unit.sample(
        batch, steps=1, generator=torch.Generator().manual_seed(17)
    )
    P_half, V_half = half.sample(
        batch, steps=1, generator=torch.Generator().manual_seed(17)
    )

    assert torch.equal(P_unit, P_half)
    assert torch.allclose(V_unit, 2.0 * V_half, atol=1e-6, rtol=1e-6)


def test_source_noise_v_does_not_create_missing_atoms():
    batch = _batch()
    batch["atom_mask"][:, :, 5:] = False
    model = DeepJumpLite(ModelConfig(source_noise_v=True), noise_sigma=0.1)
    _, noisy_v = model.interpolate(
        batch["P_t"], batch["V_t"], batch["P_t"], batch["V_t"], torch.zeros(2),
        torch.Generator().manual_seed(5), atom_mask=batch["atom_mask"],
    )
    assert torch.count_nonzero(noisy_v[:, :, 5:]) == 0
    assert torch.count_nonzero(noisy_v[:, :, :5]) > 0
