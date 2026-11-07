"""Tests for the ``hasar yeniden-uret`` subcommand of :func:`hasar.cli.main`.

The subcommand re-runs a published BRIGHT zero-shot baseline through
:func:`hasar.eval.reproduce`. Everything here is data-free and torch-free: the CLI is expected to
reach the runner through the lazy loader :func:`hasar.cli._load_reproduce`, which the tests replace
with a fake namespace holding a recording ``reproduce`` and a fake ``download_checkpoint``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hasar.cli import main

#: Published checkpoint file of each baseline (pinned here so the test never imports torch).
CHECKPOINTS = {
    "unet": "ckpt_UNet_cross_event_zeroshot_turkey-earthquake.pth",
    "siamcrnn": "ckpt_SiamCRNN_cross_event_zeroshot_turkey-earthquake.pth",
    "damageformer": "ckpt_DamageFormer_cross_event_zeroshot_turkey-earthquake.pth",
    "deeplabv3plus": "ckpt_DeepLabV3Plus_cross_event_zeroshot_turkey-earthquake.pth",
}

#: Positional parameter names of :func:`hasar.eval.reproduce`, in order.
POSITIONAL = ("key", "data_root", "out_dir")


class FakeReproducer:
    """Record every ``reproduce`` call and return a record-like object."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def reproduce(self, *args, **kwargs):
        call = dict(zip(POSITIONAL, args, strict=False))
        call.update(kwargs)
        self.calls.append(call)
        return SimpleNamespace(values={"miou_pct": 38.3}, metric_tag="[ZS-EVENT mIoU, repro]")


class FakeDownloader:
    """Record every ``download_checkpoint`` call and return a path inside the cache."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def download_checkpoint(self, name: str, cache_dir: str | Path) -> Path:
        cache = Path(cache_dir)
        self.calls.append((name, cache))
        return cache / name


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch ``hasar.cli._load_reproduce`` with a fake namespace and return its parts."""
    reproducer = FakeReproducer()
    downloader = FakeDownloader()
    namespace = SimpleNamespace(
        reproduce=reproducer.reproduce,
        download_checkpoint=downloader.download_checkpoint,
    )
    monkeypatch.setattr("hasar.cli._load_reproduce", lambda: namespace, raising=False)
    return SimpleNamespace(reproduce=reproducer, download=downloader, namespace=namespace)


def test_yeniden_uret_passes_arguments_through(tmp_path: Path, fake: SimpleNamespace) -> None:
    data = tmp_path / "data"
    out = tmp_path / "out"
    checkpoint = tmp_path / "ckpt.pth"

    code = main(
        [
            "yeniden-uret",
            "--model",
            "unet",
            "--data",
            str(data),
            "--out",
            str(out),
            "--split",
            "val",
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cuda",
            "--tta",
        ]
    )

    assert code == 0
    assert len(fake.reproduce.calls) == 1
    call = fake.reproduce.calls[0]
    assert call["key"] == "unet"
    assert Path(call["data_root"]) == data
    assert Path(call["out_dir"]) == out
    assert call["split"] == "val"
    assert call["device"] == "cuda"
    assert call["tta"] is True
    assert Path(call["checkpoint_path"]) == checkpoint
    # An explicit checkpoint is never downloaded.
    assert fake.download.calls == []


def test_yeniden_uret_defaults(tmp_path: Path, fake: SimpleNamespace) -> None:
    data = tmp_path / "data"
    out = tmp_path / "out"

    code = main(["yeniden-uret", "--model", "siamcrnn", "--data", str(data), "--out", str(out)])

    assert code == 0
    call = fake.reproduce.calls[0]
    assert call["key"] == "siamcrnn"
    assert call["split"] == "test"
    assert call["device"] == "cpu"
    assert call["tta"] is False


def test_yeniden_uret_downloads_missing_checkpoint(tmp_path: Path, fake: SimpleNamespace) -> None:
    data = tmp_path / "data"
    out = tmp_path / "out"
    cache = tmp_path / "cache"

    code = main(
        [
            "yeniden-uret",
            "--model",
            "damageformer",
            "--data",
            str(data),
            "--out",
            str(out),
            "--cache",
            str(cache),
        ]
    )

    assert code == 0
    assert len(fake.download.calls) == 1
    name, cache_dir = fake.download.calls[0]
    assert name.endswith("_cross_event_zeroshot_turkey-earthquake.pth")
    assert name == CHECKPOINTS["damageformer"]
    assert Path(cache_dir) == cache
    # The downloaded path is handed to the runner.
    assert Path(fake.reproduce.calls[0]["checkpoint_path"]) == cache / name


def test_yeniden_uret_default_cache_is_data_checkpoints(tmp_path: Path, fake: SimpleNamespace) -> None:
    data = tmp_path / "data"
    out = tmp_path / "out"

    code = main(["yeniden-uret", "--model", "unet", "--data", str(data), "--out", str(out)])

    assert code == 0
    assert len(fake.download.calls) == 1
    _, cache_dir = fake.download.calls[0]
    cache = Path(cache_dir)
    assert cache.name == "checkpoints"
    assert cache.parent.name == "data"


@pytest.mark.parametrize("model", sorted(CHECKPOINTS))
def test_yeniden_uret_accepts_every_baseline(tmp_path: Path, fake: SimpleNamespace, model: str) -> None:
    data = tmp_path / "data"
    out = tmp_path / "out"

    code = main(["yeniden-uret", "--model", model, "--data", str(data), "--out", str(out)])

    assert code == 0
    assert fake.reproduce.calls[0]["key"] == model
    assert fake.download.calls[0][0] == CHECKPOINTS[model]


def test_yeniden_uret_unknown_model_exits(tmp_path: Path, fake: SimpleNamespace) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "yeniden-uret",
                "--model",
                "resnet",
                "--data",
                str(tmp_path / "data"),
                "--out",
                str(tmp_path / "out"),
            ]
        )

    assert excinfo.value.code != 0
    assert fake.reproduce.calls == []
    assert fake.download.calls == []
