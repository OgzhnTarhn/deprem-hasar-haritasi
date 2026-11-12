"""Tests for the pre-training gates (``hasar.preflight.gates``, design section 8).

The eleven gates must pass before any GPU time is spent, so each one is pinned here on the CPU
with tiny fakes: a two-head convolutional model, in-memory datasets and result records written to
``tmp_path``. No data files and no network are needed.

The module does not exist yet; the ``gates`` fixture imports it lazily so every test fails with a
clear message instead of a collection error.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hasar.config import Config  # noqa: E402
from hasar.data.events import Split  # noqa: E402
from hasar.results import ResultRecord, write_result  # noqa: E402

CROP = 16
TRAIN_NAMES = [f"noto-earthquake_0000000{i}" for i in range(4)]
VAL_NAMES = ["la_palma-volcano_00000001", "beirut-explosion_00000002"]
TURKEY_NAME = "turkey-earthquake_00000001"

REPRO_TAG = "[ZS-EVENT mIoU, repro]"
VAL_REPRO_TAG = "[VAL13 mIoU, repro]"


@pytest.fixture(scope="module")
def gates():
    """Import ``hasar.preflight.gates`` lazily so a missing module fails the test, not collection."""
    try:
        from hasar.preflight import gates as module
    except ImportError as exc:  # pragma: no cover - only while the module does not exist yet
        pytest.fail(f"hasar.preflight.gates bulunamadı: {exc}")
    return module


# --- fakes (copied from tests/test_train_loop.py) ---------------------------


class TinyModel(torch.nn.Module):
    """Two-head model with a dropout before the convolutions, so RNG state matters."""

    def __init__(self) -> None:
        super().__init__()
        self.drop = torch.nn.Dropout(0.1)
        self.loc = torch.nn.Conv2d(6, 2, 3, padding=1)
        self.clf = torch.nn.Conv2d(6, 4, 3, padding=1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.drop(torch.cat([pre, post], dim=1))
        return {"loc": self.loc(features), "clf": self.clf(features)}


class ZeroHeadModel(TinyModel):
    """Both heads are zero-initialised, so the initial losses are exactly ln 4 and ln 2."""

    def __init__(self) -> None:
        super().__init__()
        for module in (self.loc, self.clf):
            torch.nn.init.zeros_(module.weight)
            torch.nn.init.zeros_(module.bias)


class BiasedModel(TinyModel):
    """Both heads carry a huge bias, so the initial losses are far from ln 4 and ln 2."""

    def __init__(self) -> None:
        super().__init__()
        with torch.no_grad():
            self.loc.bias.copy_(torch.tensor([50.0, 0.0]))
            self.clf.bias.copy_(torch.tensor([50.0, 0.0, 0.0, 0.0]))


class PreOnlyModel(torch.nn.Module):
    """Ignores the post-disaster image, so the SAR branch is not connected."""

    def __init__(self) -> None:
        super().__init__()
        self.loc = torch.nn.Conv2d(3, 2, 3, padding=1)
        self.clf = torch.nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loc": self.loc(pre), "clf": self.clf(pre)}


def make_model(seed: int) -> TinyModel:
    """Return a freshly initialised :class:`TinyModel` from ``seed``."""
    torch.manual_seed(seed)
    return TinyModel()


def _sample(rng: np.random.Generator, name: str) -> dict[str, object]:
    """Return one deterministic sample drawn from ``rng``."""
    pre = rng.random((3, CROP, CROP), dtype=np.float32)
    post = rng.random((3, CROP, CROP), dtype=np.float32)
    clf = rng.integers(0, 4, size=(CROP, CROP), dtype=np.int64)
    loc = np.where((clf > 0) & (clf < 4), 1, clf).astype(np.int64)
    valid = np.ones((CROP, CROP), dtype=np.bool_)
    return {"name": name, "pre": pre, "post": post, "clf": clf, "loc": loc, "valid": valid}


class FakeTrainSet:
    """Map-style train set keyed by ``(index, sample_no)``, deterministic per key."""

    def __init__(self, names: list[str] | None = None) -> None:
        self.names = list(TRAIN_NAMES if names is None else names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, key: tuple[int, int]) -> dict[str, object]:
        index, sample_no = key
        rng = np.random.default_rng([index, sample_no])
        return _sample(rng, self.names[index])


class FakeValSet:
    """Map-style eval set keyed by an integer index, deterministic per index."""

    def __init__(self, names: list[str] | None = None) -> None:
        self.names = list(VAL_NAMES if names is None else names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, object]:
        rng = np.random.default_rng([1000 + index])
        return _sample(rng, self.names[index])


def make_cfg() -> Config:
    """Return the tiny six-step test configuration."""
    cfg = Config()
    return dataclasses.replace(
        cfg,
        train=dataclasses.replace(cfg.train, batch=2, steps=6, amp="fp32", ema=0.9, seed=0),
        sched=dataclasses.replace(cfg.sched, warmup_steps=2),
        eval=dataclasses.replace(cfg.eval, every=3),
        ckpt=dataclasses.replace(cfg.ckpt, every=3, keep_last=2),
        data=dataclasses.replace(cfg.data, workers=0, crop=CROP),
    )


def make_batch(n: int = 8, labels: str = "constant") -> dict[str, torch.Tensor]:
    """Return a fixed batch of ``n`` 16x16 samples.

    ``labels="constant"`` gives every pixel class 1 (a fixed batch the model can memorise);
    ``labels="random"`` gives random four-class labels.
    """
    rng = np.random.default_rng(0)
    pre = torch.from_numpy(rng.random((n, 3, CROP, CROP), dtype=np.float32))
    post = torch.from_numpy(rng.random((n, 3, CROP, CROP), dtype=np.float32))
    if labels == "constant":
        clf = torch.ones((n, CROP, CROP), dtype=torch.long)
    else:
        clf = torch.from_numpy(rng.integers(0, 4, size=(n, CROP, CROP), dtype=np.int64))
    loc = torch.where((clf > 0) & (clf < 4), torch.ones_like(clf), clf)
    return {"pre": pre, "post": post, "loc": loc, "clf": clf}


def mentions(result, text: str) -> bool:
    """Return ``True`` if ``text`` appears in the gate's detail or in its value."""
    return text in result.detail or text in json.dumps(result.value, default=str)


def write_repro(
    directory: Path,
    model: str,
    within_tolerance: bool,
    tag: str = REPRO_TAG,
    name: str | None = None,
) -> Path:
    """Write one reproduction result record and return its path."""
    record = ResultRecord(
        metric_tag=tag,
        values={"miou_pct": 38.0, "published_pct": 38.32, "diff_pct": -0.32},
        model=model,
        split="zero_shot/test" if tag == REPRO_TAG else "zero_shot/val",
        commit="abc123",
        config_hash="none",
        extra={"within_tolerance": within_tolerance},
    )
    return write_result(directory / (name or f"{model}_test.json"), record)


# --- gate 1: reproduction ---------------------------------------------------


def test_gate_reproduction_passes_with_three_required_and_a_failing_extra(gates, tmp_path: Path) -> None:
    for model in gates.REQUIRED_REPRO:
        write_repro(tmp_path, model, within_tolerance=True)
    # A documented negative finding: this checkpoint appears to have seen Turkey.
    write_repro(tmp_path, "bright_deeplabv3plus", within_tolerance=False)
    # A validation record must be ignored (its tag is [VAL13 mIoU, repro]).
    write_repro(tmp_path, "bright_unet", within_tolerance=False, tag=VAL_REPRO_TAG, name="bright_unet_val.json")

    result = gates.gate_reproduction(tmp_path)

    assert result.gate == 1
    assert result.passed is True
    assert mentions(result, "bright_deeplabv3plus")


def test_gate_reproduction_fails_when_a_required_model_is_out_of_tolerance(gates, tmp_path: Path) -> None:
    for model in gates.REQUIRED_REPRO:
        write_repro(tmp_path, model, within_tolerance=model != "bright_unet")

    result = gates.gate_reproduction(tmp_path)

    assert result.passed is False
    assert mentions(result, "bright_unet")


def test_gate_reproduction_fails_when_a_required_model_is_missing(gates, tmp_path: Path) -> None:
    for model in gates.REQUIRED_REPRO[:2]:
        write_repro(tmp_path, model, within_tolerance=True)

    result = gates.gate_reproduction(tmp_path)

    assert result.passed is False
    assert mentions(result, "bright_damageformer")


def test_gate_reproduction_fails_without_any_record(gates, tmp_path: Path) -> None:
    result = gates.gate_reproduction(tmp_path)

    assert result.passed is False


# --- gate 2: split ----------------------------------------------------------


def _split(train: int, val: int, test: int, turkey_in_val: bool = False) -> Split:
    """Return a synthetic split with the given part sizes."""
    train_names = tuple(f"noto-earthquake_{index:08d}" for index in range(train))
    val_names = [f"beirut-explosion_{index:08d}" for index in range(val)]
    if turkey_in_val and val_names:
        val_names[0] = TURKEY_NAME
    test_names = tuple(f"{TURKEY_NAME.rsplit('_', 1)[0]}_{index:08d}" for index in range(test))
    return Split(train=train_names, val=tuple(val_names), test=test_names)


def test_gate_split_passes_with_expected_counts(gates) -> None:
    result = gates.gate_split(_split(2, 1, 3), expected=(2, 1, 3))

    assert result.gate == 2
    assert result.passed is True


def test_gate_split_fails_with_wrong_counts(gates) -> None:
    result = gates.gate_split(_split(2, 1, 3), expected=(2, 2, 3))

    assert result.passed is False


def test_gate_split_fails_when_turkey_is_in_val(gates) -> None:
    result = gates.gate_split(_split(2, 1, 3, turkey_in_val=True), expected=(2, 1, 3))

    assert result.passed is False


# --- gate 3: sampler length -------------------------------------------------


def test_gate_sampler_length_matches_steps_times_batch(gates) -> None:
    result = gates.gate_sampler_length(100, 4, 10, seed=0)

    assert result.gate == 3
    assert result.passed is True


# --- gate 4: initial loss ---------------------------------------------------


def test_gate_initial_loss_passes_for_zero_heads(gates) -> None:
    result = gates.gate_initial_loss(ZeroHeadModel(), make_batch())

    assert result.gate == 4
    assert result.passed is True


def test_gate_initial_loss_fails_for_huge_biases(gates) -> None:
    result = gates.gate_initial_loss(BiasedModel(), make_batch())

    assert result.passed is False


# --- gate 5: overfit --------------------------------------------------------


def test_gate_overfit_memorises_a_fixed_batch(gates) -> None:
    # A constant label map is the only fixed batch this tiny model memorises below ratio 0.2 in
    # 150 steps; random four-class labels stay near 0.94 because the Lovász terms do not move.
    result = gates.gate_overfit(make_model(0), make_batch(labels="constant"), steps=150, lr=1e-2)

    assert result.gate == 5
    assert result.passed is True


# --- gate 6: SAR branch connected -------------------------------------------


def test_gate_sar_connected_passes_for_tiny_model(gates) -> None:
    result = gates.gate_sar_connected(make_model(0), make_batch())

    assert result.gate == 6
    assert result.passed is True


def test_gate_sar_connected_fails_when_post_is_ignored(gates) -> None:
    result = gates.gate_sar_connected(PreOnlyModel(), make_batch())

    assert result.passed is False


# --- gate 7: metric edge cases ----------------------------------------------


def test_gate_metric_edges_checks_perfect_and_always_background(gates) -> None:
    result = gates.gate_metric_edges(np.array([700, 200, 60, 40]))

    assert result.gate == 7
    assert result.passed is True
    assert result.value["always_background"] == pytest.approx(0.7 / 4)


# --- gate 8: speed projection -----------------------------------------------


def test_gate_speed_projects_cu_within_budget(gates) -> None:
    result = gates.gate_speed([0.5, 0.5, 0.5], steps=1000, cu_per_hour=5.4, budget_cu=10.0)

    assert result.gate == 8
    assert result.passed is True
    assert result.value["projected_cu"] == pytest.approx(0.5 * 1000 / 3600 * 5.4)


def test_gate_speed_fails_when_projection_exceeds_budget(gates) -> None:
    result = gates.gate_speed([2.0, 2.0], steps=40000, cu_per_hour=5.4, budget_cu=10.0)

    assert result.passed is False


# --- gate 9: gradient clipping threshold ------------------------------------


def test_gate_grad_clip_uses_the_quantile(gates) -> None:
    norms = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    result = gates.gate_grad_clip(norms, quantile=0.9)

    assert result.gate == 9
    assert result.passed is True
    assert result.value["clip"] == pytest.approx(float(np.quantile(norms, 0.9)))


def test_gate_grad_clip_fails_for_non_finite_norms(gates) -> None:
    result = gates.gate_grad_clip([0.1, float("nan")], quantile=0.9)

    assert result.passed is False


# --- gate 10: resume --------------------------------------------------------


def test_gate_resume_reproduces_the_uninterrupted_run(gates, tmp_path: Path) -> None:
    result = gates.gate_resume(
        make_cfg(),
        lambda: make_model(0),
        FakeTrainSet(),
        FakeValSet(),
        tmp_path,
        stop=4,
    )

    assert result.gate == 10
    assert result.passed is True


# --- gate 11: kernel --------------------------------------------------------


def test_gate_kernel_fails_without_cuda(gates) -> None:
    result = gates.gate_kernel(lambda: TinyModel(), "cpu")

    assert result.gate == 11
    assert result.passed is False
    assert "cuda" in result.detail


# --- preflight.json ---------------------------------------------------------


def _results(gates, passed: bool = True) -> list:
    """Return two gate results, the second optionally failing."""
    return [
        gates.GateResult(gate=1, name="reproduction", passed=True, value={"models": 3}, detail="ok"),
        gates.GateResult(gate=2, name="split", passed=passed, value={"train": 2795}, detail="ok"),
    ]


def test_write_and_read_preflight_round_trip(gates, tmp_path: Path) -> None:
    path = tmp_path / "preflight.json"

    gates.write_preflight(path, _results(gates), config_hash="abc123", commit="deadbeef")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["config_hash"] == "abc123"
    assert payload["commit"] == "deadbeef"
    assert payload["all_passed"] is True
    assert [entry["gate"] for entry in payload["gates"]] == [1, 2]
    assert payload["gates"][0]["name"] == "reproduction"
    assert payload["gates"][0]["value"] == {"models": 3}

    read_back = gates.read_preflight(path)
    assert read_back["config_hash"] == "abc123"
    assert read_back["all_passed"] is True
    assert [entry["gate"] for entry in read_back["gates"]] == [1, 2]


def test_preflight_is_current_requires_the_file(gates, tmp_path: Path) -> None:
    assert gates.preflight_is_current(tmp_path / "preflight.json", "abc123") is False


def test_preflight_is_current_requires_the_same_hash(gates, tmp_path: Path) -> None:
    path = tmp_path / "preflight.json"
    gates.write_preflight(path, _results(gates), config_hash="abc123", commit="deadbeef")

    assert gates.preflight_is_current(path, "other") is False


def test_preflight_is_current_requires_all_gates_passed(gates, tmp_path: Path) -> None:
    path = tmp_path / "preflight.json"
    gates.write_preflight(path, _results(gates, passed=False), config_hash="abc123", commit="deadbeef")

    assert gates.preflight_is_current(path, "abc123") is False


def test_preflight_is_current_when_hash_matches_and_all_passed(gates, tmp_path: Path) -> None:
    path = tmp_path / "preflight.json"
    gates.write_preflight(path, _results(gates), config_hash="abc123", commit="deadbeef")

    assert gates.preflight_is_current(path, "abc123") is True


def test_gate_resume_with_tolerance_reports_atol(gates, tmp_path: Path) -> None:
    result = gates.gate_resume(
        make_cfg(), lambda: make_model(0), FakeTrainSet(), FakeValSet(), tmp_path, stop=4, atol=1e-6
    )
    assert result.passed
    assert result.value["atol"] == 1e-6


def test_gate_kernel_on_cpu_model_reports_no_scan_backend(gates) -> None:
    assert gates._scan_backend_of(TinyModel()) == "none"


# --- failed_gate ------------------------------------------------------------


def test_failed_gate_records_the_exception(gates) -> None:
    result = gates.failed_gate(5, RuntimeError("boom"))

    assert result.passed is False
    assert result.name == gates.GATE_NAMES[4]
    assert result.value["error"] == "RuntimeError"
    assert "boom" in result.value["message"]
    assert result.detail.startswith("Kapı çöktü: ")
    assert "RuntimeError" in result.detail


def test_failed_gate_truncates_a_long_message(gates) -> None:
    result = gates.failed_gate(5, RuntimeError("x" * 5000))

    assert len(result.value["message"]) <= 2000


# --- write_preflight: complete flag -----------------------------------------


def test_write_preflight_incomplete_run_is_not_all_passed(gates, tmp_path: Path) -> None:
    path = tmp_path / "preflight.json"

    gates.write_preflight(path, _results(gates), config_hash="abc123", commit="deadbeef", complete=False)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["complete"] is False
    assert payload["all_passed"] is False
    assert gates.preflight_is_current(path, "abc123") is False


def test_write_preflight_defaults_to_complete(gates, tmp_path: Path) -> None:
    path = tmp_path / "preflight.json"

    gates.write_preflight(path, _results(gates), config_hash="abc123", commit="deadbeef")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["complete"] is True


# --- gate_overfit: amp keyword ----------------------------------------------


def test_gate_overfit_accepts_amp_on_cpu(gates) -> None:
    result = gates.gate_overfit(make_model(0), make_batch(labels="constant"), steps=150, lr=1e-2, amp=True)

    assert result.gate == 5
    assert result.passed is True
