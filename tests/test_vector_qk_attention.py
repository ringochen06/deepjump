import torch
import pytest

from deepjump.model.layers import EquivAttention


def _inputs():
    torch.manual_seed(4)
    return (
        torch.randn(2, 5, 32),
        torch.randn(2, 5, 16, 3),
        torch.randn(2, 5, 3),
        torch.ones(2, 5, dtype=torch.bool),
    )


def test_zero_gated_vector_qk_exactly_matches_legacy_attention():
    legacy = EquivAttention(32, 16, 4)
    vector = EquivAttention(32, 16, 4, vector_qk=True)
    compatible = vector.load_state_dict(legacy.state_dict(), strict=False)
    assert set(compatible.missing_keys) == {
        "vector_qk_gate", "to_qv.weight", "to_kv.weight",
    }
    assert not compatible.unexpected_keys

    legacy_out = legacy(*_inputs())
    vector_out = vector(*_inputs())
    assert torch.equal(legacy_out[0], vector_out[0])
    assert torch.equal(legacy_out[1], vector_out[1])


def test_vector_qk_gate_and_projections_receive_finite_gradients():
    attention = EquivAttention(32, 16, 4, vector_qk=True)
    with torch.no_grad():
        attention.vector_qk_gate.fill_(0.1)
    scalar, vector = attention(*_inputs())
    (scalar.square().mean() + vector.square().mean()).backward()
    for parameter in (
        attention.vector_qk_gate, attention.to_qv.weight, attention.to_kv.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_vector_qk_attention_is_rotation_equivariant_and_zero_distance_finite():
    attention = EquivAttention(32, 16, 4, vector_qk=True).eval()
    with torch.no_grad():
        attention.vector_qk_gate.fill_(0.2)
    s, v, p, mask = _inputs()
    p[:, 1] = p[:, 0]  # explicit zero-distance pair
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    out_s, out_v = attention(s, v, p, mask)
    rotated_s, rotated_v = attention(s, v @ q.T, p @ q.T, mask)
    assert torch.isfinite(out_s).all() and torch.isfinite(out_v).all()
    assert torch.allclose(rotated_s, out_s, atol=2e-5, rtol=2e-5)
    assert torch.allclose(rotated_v, out_v @ q.T, atol=2e-5, rtol=2e-5)


def test_tensor_qkv_has_joint_projections_without_legacy_gate():
    attention = EquivAttention(32, 16, 4, tensor_qkv=True)
    assert hasattr(attention, "to_qv") and hasattr(attention, "to_kv")
    assert not hasattr(attention, "vector_qk_gate")

    scalar, vector = attention(*_inputs())
    (scalar.square().mean() + vector.square().mean()).backward()
    for parameter in (attention.to_qv.weight, attention.to_kv.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_tensor_qkv_uses_joint_scalar_vector_inner_product_normalization():
    attention = EquivAttention(32, 16, 4, tensor_qkv=True)
    s, v, _, _ = _inputs()
    b, n, _ = s.shape
    q = attention.to_q(s).view(b, n, attention.h, attention.dh)
    k = attention.to_k(s).view(b, n, attention.h, attention.dh)
    qv = attention.to_qv(v).view(b, n, attention.h, attention.cvh, 3)
    kv = attention.to_kv(v).view(b, n, attention.h, attention.cvh, 3)
    expected = (
        torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
        + torch.einsum("bihcx,bjhcx->bhij", qv.float(), kv.float())
    ) / (attention.dh + 3 * attention.cvh) ** 0.5
    actual = attention._content_logits(s, v)
    assert torch.allclose(actual, expected.to(actual.dtype), atol=1e-6, rtol=1e-6)


def test_tensor_qkv_is_rotation_equivariant_and_zero_distance_finite():
    attention = EquivAttention(32, 16, 4, tensor_qkv=True).eval()
    s, v, p, mask = _inputs()
    p[:, 1] = p[:, 0]
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    out_s, out_v = attention(s, v, p, mask)
    rotated_s, rotated_v = attention(s, v @ q.T, p @ q.T, mask)
    assert torch.isfinite(out_s).all() and torch.isfinite(out_v).all()
    assert torch.allclose(rotated_s, out_s, atol=2e-5, rtol=2e-5)
    assert torch.allclose(rotated_v, out_v @ q.T, atol=2e-5, rtol=2e-5)


def test_tensor_qkv_rejects_legacy_vector_gate_combination():
    with pytest.raises(ValueError, match="mutually exclusive"):
        EquivAttention(32, 16, 4, vector_qk=True, tensor_qkv=True)
