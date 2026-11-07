"""BRIGHT Zenodo taban çizgisi modellerinin testleri (torch gerektirir)."""

import hashlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from hasar.models import bright_baselines  # noqa: E402
from hasar.models.base import validate_output  # noqa: E402
from hasar.models.registry import available, build  # noqa: E402

BATCH = 1
SIZE = 64

EXPECTED = {
    "unet": ("bright_unet", "ckpt_UNet_cross_event_zeroshot_turkey-earthquake.pth", 37.82),
    "siamcrnn": (
        "bright_siamcrnn",
        "ckpt_SiamCRNN_cross_event_zeroshot_turkey-earthquake.pth",
        38.32,
    ),
    "damageformer": (
        "bright_damageformer",
        "ckpt_DamageFormer_cross_event_zeroshot_turkey-earthquake.pth",
        37.20,
    ),
    "deeplabv3plus": (
        "bright_deeplabv3plus",
        "ckpt_DeepLabV3Plus_cross_event_zeroshot_turkey-earthquake.pth",
        37.28,
    ),
}

SINGLE_STREAM = ("unet", "deeplabv3plus")


@pytest.fixture(scope="module")
def built() -> dict[str, torch.nn.Module]:
    """Her taban çizgi modelini bir kez kur (CPU)."""
    return {name: build(spec.registry_name) for name, spec in bright_baselines.BASELINES.items()}


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    pre = torch.randn(BATCH, 3, SIZE, SIZE, generator=generator)
    post = torch.randn(BATCH, 3, SIZE, SIZE, generator=generator)
    return pre, post


# --- sabitler -------------------------------------------------------------


def test_zenodo_record() -> None:
    assert bright_baselines.ZENODO_RECORD == "15349462"


def test_checkpoint_url() -> None:
    name = "ckpt_UNet_cross_event_zeroshot_turkey-earthquake.pth"
    assert bright_baselines.checkpoint_url(name) == (f"https://zenodo.org/records/15349462/files/{name}?download=1")


def test_baselines_keys() -> None:
    assert set(bright_baselines.BASELINES) == set(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_baseline_metadata(name: str) -> None:
    registry_name, checkpoint, published = EXPECTED[name]
    spec = bright_baselines.BASELINES[name]
    assert spec.registry_name == registry_name
    assert spec.checkpoint == checkpoint
    assert spec.published_turkey == pytest.approx(published)


# --- kayıt defteri --------------------------------------------------------


def test_import_registers_models() -> None:
    names = available()
    for registry_name, _, _ in EXPECTED.values():
        assert registry_name in names


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_build_returns_module(name: str, built: dict[str, torch.nn.Module]) -> None:
    assert isinstance(built[name], torch.nn.Module)


# --- ileri geçiş ----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_forward_matches_contract(name: str, built: dict[str, torch.nn.Module]) -> None:
    pre, post = _inputs()
    model = built[name].eval()
    with torch.no_grad():
        out = model(pre, post)
    validate_output(out, BATCH, SIZE, SIZE)


@pytest.mark.parametrize("name", SINGLE_STREAM)
def test_single_stream_loc_derived_from_clf(name: str, built: dict[str, torch.nn.Module]) -> None:
    pre, post = _inputs()
    model = built[name].eval()
    with torch.no_grad():
        out = model(pre, post)
    loc, clf = out["loc"], out["clf"]
    assert torch.allclose(loc[:, 0], clf[:, 0])
    assert torch.allclose(loc[:, 1], torch.logsumexp(clf[:, 1:], dim=1))


# --- checkpoint yükleme ---------------------------------------------------


def _prefixed_state_dict(model: torch.nn.Module, prefix: str = "module.") -> dict:
    return {prefix + key: value for key, value in model.net.state_dict().items()}


def _assert_same_state(model: torch.nn.Module, reference: dict) -> None:
    state = model.net.state_dict()
    assert set(state) == set(reference)
    for key, value in reference.items():
        assert torch.equal(state[key], value), key


def test_load_checkpoint_with_module_prefix(tmp_path) -> None:
    source = build("bright_unet")
    reference = source.net.state_dict()
    path = tmp_path / "prefixed.pth"
    torch.save(_prefixed_state_dict(source), path)

    target = build("bright_unet")
    bright_baselines.load_checkpoint(target, path)
    _assert_same_state(target, reference)


def test_load_checkpoint_plain_state_dict(tmp_path) -> None:
    source = build("bright_unet")
    reference = source.net.state_dict()
    path = tmp_path / "plain.pth"
    torch.save(_prefixed_state_dict(source, prefix=""), path)

    target = build("bright_unet")
    bright_baselines.load_checkpoint(target, path)
    _assert_same_state(target, reference)


def test_load_checkpoint_nested_state_dict(tmp_path) -> None:
    source = build("bright_unet")
    reference = source.net.state_dict()
    path = tmp_path / "nested.pth"
    torch.save({"state_dict": _prefixed_state_dict(source)}, path)

    target = build("bright_unet")
    bright_baselines.load_checkpoint(target, path)
    _assert_same_state(target, reference)


def test_load_checkpoint_missing_key_raises(tmp_path) -> None:
    source = build("bright_unet")
    reference = _prefixed_state_dict(source)
    reference.pop(next(iter(reference)))
    path = tmp_path / "missing.pth"
    torch.save(reference, path)

    target = build("bright_unet")
    with pytest.raises(RuntimeError):
        bright_baselines.load_checkpoint(target, path)


# --- yardımcılar ----------------------------------------------------------


def test_file_sha256(tmp_path) -> None:
    payload = b"deprem hasar haritasi\n"
    path = tmp_path / "small.bin"
    path.write_bytes(payload)
    assert bright_baselines.file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_file_md5(tmp_path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"hasar")
    assert bright_baselines.file_md5(path) == hashlib.md5(b"hasar").hexdigest()


def test_expected_md5_known_and_unknown() -> None:
    assert bright_baselines.expected_md5(EXPECTED["siamcrnn"][1]) == "80705c64573c2684805a181b2fe31cab"
    assert bright_baselines.expected_md5("baska.pth") is None


def test_download_rejects_corrupt_cached_file(tmp_path, monkeypatch) -> None:
    name = EXPECTED["unet"][1]
    (tmp_path / name).write_bytes(b"truncated")
    monkeypatch.setattr(bright_baselines.urllib.request, "urlopen", _no_network)
    with pytest.raises(ValueError, match="md5"):
        bright_baselines.download_checkpoint(name, tmp_path)


def test_download_deletes_bad_fresh_file(tmp_path, monkeypatch) -> None:
    import io

    name = EXPECTED["unet"][1]
    monkeypatch.setattr(bright_baselines.urllib.request, "urlopen", lambda url: io.BytesIO(b"half"))
    with pytest.raises(ValueError, match="md5"):
        bright_baselines.download_checkpoint(name, tmp_path)
    assert not (tmp_path / name).exists()
    assert not (tmp_path / f"{name}.part").exists()


def test_download_reuses_unknown_cached_file(tmp_path, monkeypatch) -> None:
    (tmp_path / "baska.pth").write_bytes(b"x")
    monkeypatch.setattr(bright_baselines.urllib.request, "urlopen", _no_network)
    assert bright_baselines.download_checkpoint("baska.pth", tmp_path) == tmp_path / "baska.pth"


def _no_network(url):  # pragma: no cover - called only if the test is wrong
    raise AssertionError(f"ağa çıkılmamalı: {url}")
