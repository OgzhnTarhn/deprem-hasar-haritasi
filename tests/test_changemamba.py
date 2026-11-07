"""ChangeMamba ana modeli ve vendored selective scan testleri (torch gerektirir, CPU)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("einops")

from hasar.models import changemamba  # noqa: E402
from hasar.models.base import validate_output  # noqa: E402
from hasar.models.registry import available, build  # noqa: E402

BATCH = 1
SIZE = 32
SWIN_SIZE = 64


@pytest.fixture(scope="module")
def tiny() -> torch.nn.Module:
    """ChangeMambaMMBDA sarmalayıcısını bir kez kur (CPU, ön eğitim ağırlığı yok)."""
    torch.manual_seed(0)
    return build("changemamba_mmbda_tiny")


@pytest.fixture(scope="module")
def swin() -> torch.nn.Module:
    """DamageFormer Swin-T sarmalayıcısını bir kez kur (CPU, ön eğitim ağırlığı yok)."""
    torch.manual_seed(0)
    return build("damageformer_swin_t")


def _inputs(size: int = SIZE, post_channels: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    pre = torch.randn(BATCH, 3, size, size, generator=generator)
    post = torch.randn(BATCH, post_channels, size, size, generator=generator)
    return pre, post


# --- yapılandırma ---------------------------------------------------------


def test_official_vssm_kwargs() -> None:
    kwargs = changemamba.OFFICIAL_VSSM_KWARGS
    assert kwargs["depths"] == [2, 2, 4, 2]
    assert kwargs["dims"] == 96
    assert kwargs["ssm_d_state"] == 1
    assert kwargs["forward_type"] == "v3noz"
    assert kwargs["downsample_version"] == "v3"
    assert kwargs["patchembed_version"] == "v2"
    assert kwargs["norm_layer"] == "ln"
    assert kwargs["drop_path_rate"] == 0.2


# --- kayıt defteri --------------------------------------------------------


def test_import_registers_models() -> None:
    names = available()
    assert "changemamba_mmbda_tiny" in names
    assert "damageformer_swin_t" in names


def test_build_returns_module_with_net(tiny: torch.nn.Module) -> None:
    assert isinstance(tiny, torch.nn.Module)
    assert isinstance(tiny.net, torch.nn.Module)


# --- ileri geçiş ----------------------------------------------------------


def test_forward_matches_contract(tiny: torch.nn.Module) -> None:
    pre, post = _inputs()
    model = tiny.eval()
    with torch.no_grad():
        out = model(pre, post)
    validate_output(out, BATCH, SIZE, SIZE)


def test_single_channel_post_matches_replicated(tiny: torch.nn.Module) -> None:
    pre, post = _inputs(post_channels=1)
    model = tiny.eval()
    with torch.no_grad():
        single = model(pre, post)
        replicated = model(pre, post.repeat(1, 3, 1, 1))
    for key in ("loc", "clf"):
        assert torch.equal(single[key], replicated[key]), key


def test_two_channel_post_raises(tiny: torch.nn.Module) -> None:
    pre, post = _inputs(post_channels=2)
    model = tiny.eval()
    with pytest.raises(ValueError):
        with torch.no_grad():
            model(pre, post)


def test_gradients_reach_sar_encoder(tiny: torch.nn.Module) -> None:
    # Eğitim kipinde BatchNorm tek örnekle çalışmaz; bu yüzden parti 2.
    generator = torch.Generator().manual_seed(0)
    pre = torch.randn(2, 3, SIZE, SIZE, generator=generator)
    post = torch.randn(2, 3, SIZE, SIZE, generator=generator)
    model = tiny.train()
    out = model(pre, post)
    (out["loc"].sum() + out["clf"].sum()).backward()
    grads = [p.grad for p in model.net.encoder_2.parameters() if p.grad is not None]
    assert grads
    assert any(torch.any(grad != 0) for grad in grads)


# --- tarama arka ucu ------------------------------------------------------


def test_scan_backend_value() -> None:
    assert changemamba.scan_backend() in ("cuda", "torch")


def _naive_selective_scan(u, delta, A, B, C, D, delta_bias, delta_softplus):
    """Testte yazılmış çift döngülü referans (Mamba selective_scan_ref matematiği)."""
    batch, channels, length = u.shape
    groups = B.shape[1]
    state = A.shape[1]
    delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)
    out = torch.zeros(batch, channels, length, dtype=torch.float64)
    for b in range(batch):
        for d in range(channels):
            group = d // (channels // groups)
            x = torch.zeros(state, dtype=torch.float64)
            for t in range(length):
                dA = torch.exp(delta[b, d, t].double() * A[d].double())
                x = dA * x + delta[b, d, t].double() * B[b, group, :, t].double() * u[b, d, t].double()
                out[b, d, t] = torch.sum(x * C[b, group, :, t].double()) + D[d].double() * u[b, d, t].double()
    return out


def test_selective_scan_torch_matches_reference() -> None:
    torch.manual_seed(0)
    batch, channels, length, groups, state = 2, 4, 5, 2, 3
    u = torch.randn(batch, channels, length)
    delta = torch.rand(batch, channels, length)
    A = -torch.rand(channels, state) - 0.1
    B = torch.randn(batch, groups, state, length)
    C = torch.randn(batch, groups, state, length)
    D = torch.randn(channels)
    delta_bias = torch.randn(channels)

    expected = _naive_selective_scan(u, delta, A, B, C, D, delta_bias, True)
    actual = changemamba.selective_scan_torch(u, delta, A, B, C, D, delta_bias, delta_softplus=True)
    assert actual.shape == (batch, channels, length)
    assert torch.allclose(actual.double(), expected, atol=1e-5)


# --- DamageFormer Swin-T --------------------------------------------------


def test_damageformer_swin_forward(swin: torch.nn.Module) -> None:
    pre, post = _inputs(size=SWIN_SIZE)
    model = swin.eval()
    with torch.no_grad():
        out = model(pre, post)
    validate_output(out, BATCH, SWIN_SIZE, SWIN_SIZE)
