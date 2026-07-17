import torch

from deepjump.config import ModelConfig
from deepjump.model import DeepJumpLite
from deepjump.model.deepjump import count_parameters
from deepjump.model.layers import PaperFeedForward


def test_paper_feedforward_shapes_finite_and_rotation_equivariant():
    torch.manual_seed(2)
    layer = PaperFeedForward(hidden=32, vec_channels=16)
    scalar = torch.randn(2, 5, 32)
    vector = torch.randn(2, 5, 16, 3)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    out_s, out_v = layer(scalar, vector)
    rotated_s, rotated_v = layer(scalar, vector @ q.T)
    assert out_s.shape == scalar.shape and out_v.shape == vector.shape
    assert torch.isfinite(out_s).all() and torch.isfinite(out_v).all()
    assert torch.allclose(rotated_s, out_s, atol=2e-5, rtol=2e-5)
    assert torch.allclose(rotated_v, out_v @ q.T, atol=2e-5, rtol=2e-5)


def test_paper_feedforward_all_branches_receive_finite_gradients():
    torch.manual_seed(7)
    layer = PaperFeedForward(hidden=32, vec_channels=16)
    scalar = torch.randn(2, 5, 32, requires_grad=True)
    vector = torch.randn(2, 5, 16, 3, requires_grad=True)

    out_s, out_v = layer(scalar, vector)
    (out_s.square().mean() + out_v.square().mean()).backward()

    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_paper_architecture_is_near_reported_four_million_scale():
    cfg = ModelConfig(
        hidden=128, vector_channels=64, num_heads=4,
        cond_layers=6, transport_layers=6, vector_qk=True, paper_ff=True,
    )
    parameters = count_parameters(DeepJumpLite(cfg, predict_heavy=True))
    assert 3_500_000 <= parameters <= 4_200_000


def test_tensor_cloud_architecture_is_near_reported_four_million_scale():
    cfg = ModelConfig(
        hidden=128, vector_channels=64, num_heads=4,
        cond_layers=6, transport_layers=6, tensor_qkv=True, paper_ff=True,
    )
    parameters = count_parameters(DeepJumpLite(cfg, predict_heavy=True))
    assert 3_500_000 <= parameters <= 4_200_000
