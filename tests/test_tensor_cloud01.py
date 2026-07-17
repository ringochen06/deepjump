import pytest
import torch

from deepjump.config import ModelConfig
from deepjump.model import DeepJumpLite
from deepjump.model.deepjump import count_parameters
from deepjump.model.tensor_cloud01 import (
    TensorCloud01Attention,
    TensorCloud01Block,
    TensorCloud01FeedForward,
)


def _rotation():
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(rotation) < 0:
        rotation[:, 0] *= -1
    return rotation


def test_attention_literal_y0_y1_and_pi_minus_pj_sign():
    attention = TensorCloud01Attention(hidden=2, num_heads=1, num_dist_basis=2)
    with torch.no_grad():
        attention.to_qkv_scalar.weight.zero_()
        attention.to_qkv_scalar.bias.zero_()
        attention.to_qkv_vector.weight.zero_()
        attention.sequence_bias.embed.weight.zero_()
        attention.distance_bias.proj.weight.zero_()
        attention.distance_bias.proj.bias.zero_()
        attention.out_scalar.weight.zero_()
        attention.out_scalar.bias.zero_()
        attention.out_scalar.weight[0, -1] = 1.0
        attention.out_vector.weight.zero_()
        attention.out_vector.weight[0, -1] = 1.0

    scalar = torch.zeros(1, 2, 2)
    vector = torch.zeros(1, 2, 2, 3)
    positions = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    scalar_out, vector_out = attention(scalar, vector, positions, mask)
    assert torch.allclose(scalar_out[..., 0], torch.ones(1, 2))
    assert torch.allclose(scalar_out[..., 1], torch.zeros(1, 2))
    assert torch.allclose(vector_out[0, 0, 0], torch.tensor([-0.5, 0.0, 0.0]))
    assert torch.allclose(vector_out[0, 1, 0], torch.tensor([0.5, 0.0, 0.0]))


def test_attention_uses_literal_k_i_dot_q_j_without_scaling():
    attention = TensorCloud01Attention(hidden=2, num_heads=1)
    with torch.no_grad():
        attention.to_qkv_scalar.weight.zero_()
        attention.to_qkv_scalar.bias.zero_()
        attention.to_qkv_vector.weight.zero_()
        attention.to_qkv_scalar.weight[0:2] = torch.eye(2)  # q
        attention.to_qkv_scalar.weight[2:4] = torch.tensor(
            [[2.0, 0.0], [0.0, -1.0]]
        )  # k
    scalar = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    vector = torch.zeros(1, 2, 2, 3)
    expected_q = scalar
    expected_k = scalar @ torch.tensor([[2.0, 0.0], [0.0, -1.0]]).T
    expected = torch.einsum("bid,bjd->bij", expected_k, expected_q)[:, None]
    assert torch.equal(attention._content_logits(scalar, vector), expected)


def test_tensor_cloud01_rotation_translation_mask_and_gradients():
    torch.manual_seed(5)
    block = TensorCloud01Block(8, 2, 4, 4, 10.0)
    scalar = torch.randn(2, 5, 8, requires_grad=True)
    vector = torch.randn(2, 5, 8, 3, requires_grad=True)
    positions = torch.randn(2, 5, 3)
    positions[:, 1] = positions[:, 0]
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    rotation = _rotation()
    translation = torch.tensor([10.0, -3.0, 2.0])

    out_scalar, out_vector = block(scalar, vector, positions, mask)
    rotated_scalar, rotated_vector = block(
        scalar, vector @ rotation.T, positions @ rotation.T + translation, mask
    )
    assert torch.isfinite(out_scalar).all() and torch.isfinite(out_vector).all()
    assert torch.allclose(rotated_scalar, out_scalar, atol=3e-5, rtol=3e-5)
    assert torch.allclose(rotated_vector, out_vector @ rotation.T, atol=3e-5, rtol=3e-5)
    assert torch.equal(out_scalar[0, 3:], torch.zeros_like(out_scalar[0, 3:]))
    assert torch.equal(out_vector[0, 3:], torch.zeros_like(out_vector[0, 3:]))

    (out_scalar.square().mean() + out_vector.square().mean()).backward()
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_attention_all_mask_is_finite_and_zero():
    attention = TensorCloud01Attention(hidden=8, num_heads=2)
    scalar_out, vector_out = attention(
        torch.randn(1, 3, 8),
        torch.randn(1, 3, 8, 3),
        torch.randn(1, 3, 3),
        torch.zeros(1, 3, dtype=torch.bool),
    )
    assert torch.equal(scalar_out, torch.zeros_like(scalar_out))
    assert torch.equal(vector_out, torch.zeros_like(vector_out))


def test_feedforward_equal_multiplicity_and_model_config_fail_closed():
    feedforward = TensorCloud01FeedForward(8)
    scalar, vector = feedforward(torch.randn(1, 3, 8), torch.randn(1, 3, 8, 3))
    assert scalar.shape == (1, 3, 8)
    assert vector.shape == (1, 3, 8, 3)

    with pytest.raises(ValueError, match="vector_channels == hidden"):
        DeepJumpLite(ModelConfig(hidden=8, vector_channels=4, tensor_cloud01=True))
    with pytest.raises(ValueError, match="dedicated path"):
        DeepJumpLite(ModelConfig(
            hidden=8, vector_channels=8, tensor_cloud01=True, tensor_qkv=True
        ))

    model = DeepJumpLite(ModelConfig(
        hidden=8, vector_channels=8, num_heads=2,
        cond_layers=1, transport_layers=1, tensor_cloud01=True,
    ))
    assert isinstance(model.conditioner.blocks[0], TensorCloud01Block)
    assert isinstance(model.transport.blocks[0], TensorCloud01Block)


def test_tensor_cloud01_full_model_forward_backward_and_parameter_scaling():
    torch.manual_seed(9)
    cfg = ModelConfig(
        hidden=8, vector_channels=8, num_heads=2,
        cond_layers=1, transport_layers=1,
        predict_heavy=True, tensor_cloud01=True,
    )
    model = DeepJumpLite(cfg, noise_sigma=0.1, predict_heavy=True)
    batch, residues = 2, 4
    data = {
        "P_t": torch.randn(batch, residues, 3),
        "V_t": torch.randn(batch, residues, 13, 3),
        "P_1": torch.randn(batch, residues, 3),
        "V_1": torch.randn(batch, residues, 13, 3),
        "res_index": torch.randint(0, 20, (batch, residues)),
        "delta_ns": torch.ones(batch),
        "residue_mask": torch.ones(batch, residues, dtype=torch.bool),
        "atom_mask": torch.ones(batch, residues, 13, dtype=torch.bool),
    }
    output = model(data, tau=torch.full((batch,), 0.25))
    assert output["P_hat_1"].shape == data["P_t"].shape
    assert output["V_hat_1"].shape == data["V_t"].shape
    loss = output["P_hat_1"].square().mean() + output["V_hat_1"].square().mean()
    loss.backward()
    expected_unused = {
        "transport.blocks.0.feedforward.scalar_out.weight",
        "transport.blocks.0.feedforward.scalar_out.bias",
    }
    actual_unused = {
        name for name, parameter in model.named_parameters() if parameter.grad is None
    }
    assert actual_unused == expected_unused
    for name, parameter in model.named_parameters():
        if name not in expected_unused:
            assert torch.isfinite(parameter.grad).all(), name

    counts = []
    for hidden in (32, 64, 128):
        scaled = DeepJumpLite(ModelConfig(
            hidden=hidden, vector_channels=hidden, num_heads=4,
            cond_layers=6, transport_layers=6,
            predict_heavy=True, tensor_cloud01=True,
        ), predict_heavy=True)
        counts.append(count_parameters(scaled))
    assert counts == [315_936, 1_225_952, 4_840_032]


def test_tensor_cloud01_checkpoint_roundtrip_and_mismatch_fail_closed():
    cfg = ModelConfig(
        hidden=8, vector_channels=8, num_heads=2,
        cond_layers=1, transport_layers=1,
        predict_heavy=True, tensor_cloud01=True,
    )
    source = DeepJumpLite(cfg, predict_heavy=True)
    restored = DeepJumpLite(cfg, predict_heavy=True)
    restored.load_state_dict(source.state_dict(), strict=True)
    for source_parameter, restored_parameter in zip(
        source.parameters(), restored.parameters(), strict=True
    ):
        assert torch.equal(source_parameter, restored_parameter)

    wrong_width = DeepJumpLite(ModelConfig(
        hidden=16, vector_channels=16, num_heads=2,
        cond_layers=1, transport_layers=1,
        predict_heavy=True, tensor_cloud01=True,
    ), predict_heavy=True)
    with pytest.raises(RuntimeError, match="size mismatch"):
        wrong_width.load_state_dict(source.state_dict(), strict=True)

    legacy = DeepJumpLite(ModelConfig(
        hidden=8, vector_channels=8, num_heads=2,
        cond_layers=1, transport_layers=1,
        predict_heavy=True,
    ), predict_heavy=True)
    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        legacy.load_state_dict(source.state_dict(), strict=True)
