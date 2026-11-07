"""Wrappers around the published BRIGHT zero-shot baseline models.

These are the published BRIGHT zero-shot checkpoints (Zenodo record 15349462, CC BY 4.0). They are
used only for the Phase 3 reproduction (project rule 1, first exception: re-running the ready-made
zero-shot checkpoints) and as the B2 ensemble candidate. The ``*_standard_ML_split.pth`` files saw
Turkey during training and are never loaded (project rule 2).

The vendored model code lives in ``third_party/bright/models`` and is loaded lazily by file path,
so ``third_party`` never enters ``sys.path``. Inputs follow the official BRIGHT format: ``pre`` is
3-channel ImageNet-normalized optical and ``post`` is SAR replicated to 3 channels and
ImageNet-normalized, exactly what :func:`hasar.data.normalize.sar_to_input` with
``mode="official"`` produces. Single-stream models receive ``torch.cat([pre, post], dim=1)`` and
return classification logits; two-stream models receive ``(pre, post)`` and return
``(loc, clf)`` logits.

torch is imported at module top: this module is only imported where torch exists.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch

from hasar.models.registry import register

#: Zenodo record holding the published BRIGHT zero-shot checkpoints.
ZENODO_RECORD = "15349462"

_VENDOR_DIR = Path(__file__).resolve().parents[3] / "third_party" / "bright" / "models"
_CHUNK_SIZE = 1 << 20
_MODULES: dict[str, ModuleType] = {}


def _vendor(name: str) -> ModuleType:
    """Load ``third_party/bright/models/<name>.py`` once and return the module."""
    module = _MODULES.get(name)
    if module is None:
        path = _VENDOR_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"bright_vendor_{name}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"BRIGHT model dosyası yüklenemedi: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES[name] = module
    return module


@dataclass(frozen=True)
class Baseline:
    """Metadata of one published BRIGHT baseline."""

    key: str
    registry_name: str
    checkpoint: str
    published_turkey: float
    two_stream: bool
    md5: str


BASELINES: dict[str, Baseline] = {
    "unet": Baseline(
        key="unet",
        registry_name="bright_unet",
        checkpoint="ckpt_UNet_cross_event_zeroshot_turkey-earthquake.pth",
        published_turkey=37.82,
        two_stream=False,
        md5="9dc559a5f815536f735687aef9da0c35",
    ),
    "siamcrnn": Baseline(
        key="siamcrnn",
        registry_name="bright_siamcrnn",
        checkpoint="ckpt_SiamCRNN_cross_event_zeroshot_turkey-earthquake.pth",
        published_turkey=38.32,
        two_stream=True,
        md5="80705c64573c2684805a181b2fe31cab",
    ),
    "damageformer": Baseline(
        key="damageformer",
        registry_name="bright_damageformer",
        checkpoint="ckpt_DamageFormer_cross_event_zeroshot_turkey-earthquake.pth",
        published_turkey=37.20,
        two_stream=True,
        md5="9276c98af54cd5e570418cdbdfbe0638",
    ),
    "deeplabv3plus": Baseline(
        key="deeplabv3plus",
        registry_name="bright_deeplabv3plus",
        checkpoint="ckpt_DeepLabV3Plus_cross_event_zeroshot_turkey-earthquake.pth",
        published_turkey=37.28,
        two_stream=False,
        md5="dca6ca50a89d465733aa9e7475a181bb",
    ),
}


class BrightBaseline(torch.nn.Module):
    """Adapt a vendored BRIGHT model to the project's ``{"loc", "clf"}`` contract."""

    def __init__(self, net: torch.nn.Module, two_stream: bool) -> None:
        super().__init__()
        self.net = net
        self.two_stream = two_stream

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ``{"loc": ..., "clf": ...}`` logits for a batch of pre/post tiles."""
        if self.two_stream:
            loc, clf = self.net(pre, post)
        else:
            clf = self.net(torch.cat([pre, post], dim=1))
            loc = torch.stack([clf[:, 0], torch.logsumexp(clf[:, 1:], dim=1)], dim=1)
        return {"loc": loc, "clf": clf}


@register("bright_unet")
def _build_unet() -> BrightBaseline:
    return BrightBaseline(_vendor("UNet").UNet(in_channels=6, num_classes=4), two_stream=False)


@register("bright_siamcrnn")
def _build_siamcrnn() -> BrightBaseline:
    return BrightBaseline(_vendor("SiamCRNN").SiamCRNN(), two_stream=True)


@register("bright_damageformer")
def _build_damageformer() -> BrightBaseline:
    return BrightBaseline(_vendor("DamageFormer").DamageFormer(), two_stream=True)


@register("bright_deeplabv3plus")
def _build_deeplabv3plus() -> BrightBaseline:
    return BrightBaseline(_vendor("DeepLabV3Plus").DeepLabV3Plus(in_channels=6, num_classes=4), two_stream=False)


def checkpoint_url(name: str) -> str:
    """Return the Zenodo download URL of the checkpoint file ``name``."""
    return f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1"


def expected_md5(name: str) -> str | None:
    """Return the Zenodo md5 of checkpoint ``name``, or ``None`` for an unknown file."""
    for baseline in BASELINES.values():
        if baseline.checkpoint == name:
            return baseline.md5
    return None


def file_md5(path: str | Path) -> str:
    """Return the hex md5 digest of the file at ``path`` (Zenodo publishes md5 checksums)."""
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(name: str, cache_dir: str | Path) -> Path:
    """Download checkpoint ``name`` into ``cache_dir`` and return its path.

    An existing final file is reused. The download streams into ``<name>.part`` and is moved into
    place with :func:`os.replace` only once complete, so an interrupted download never leaves a
    truncated checkpoint behind. For the known baselines the file's md5 is compared with the Zenodo
    checksum (also for a reused file); a mismatch raises :class:`ValueError` and a freshly
    downloaded bad file is deleted.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    final = cache / name
    expected = expected_md5(name)
    if not final.exists():
        part = cache / f"{name}.part"
        with urllib.request.urlopen(checkpoint_url(name)) as response, part.open("wb") as handle:
            while chunk := response.read(_CHUNK_SIZE):
                handle.write(chunk)
        if expected is not None and file_md5(part) != expected:
            part.unlink()
            raise ValueError(f"{name}: indirilen dosyanın md5'i Zenodo ile uyuşmuyor (eksik veya bozuk indirme).")
        os.replace(part, final)
    elif expected is not None and file_md5(final) != expected:
        raise ValueError(f"{final}: md5 Zenodo ile uyuşmuyor; dosyayı silip yeniden indirin.")
    return final


def file_sha256(path: str | Path) -> str:
    """Return the hex SHA-256 digest of the file at ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(model: BrightBaseline, path: str | Path) -> None:
    """Load a published checkpoint into ``model.net``.

    A top-level ``state_dict`` or ``model`` wrapper is unwrapped and a leading ``module.`` is
    stripped from every key (the official checkpoints were saved from ``DataParallel``). Loading is
    strict, so a missing or unexpected key raises :class:`RuntimeError`.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        for wrapper in ("state_dict", "model"):
            inner = payload.get(wrapper)
            if isinstance(inner, dict):
                payload = inner
                break
    state = {key.removeprefix("module."): value for key, value in payload.items()}
    model.net.load_state_dict(state, strict=True)
