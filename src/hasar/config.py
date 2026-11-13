"""Typed, frozen configuration for the damage-mapping pipeline.

The single YAML file (``configs/ana.yaml``) is parsed into frozen dataclasses, one per section,
so that a run's configuration cannot change after it is loaded and its hash can identify the run.

The training length field is deliberately named ``train.steps`` and means **optimizer steps**,
not samples. BRIGHT's ``max_iters`` counts samples (800.000 / 16 = 50.000 steps) while
ChangeMamba's ``max_iters`` counts steps; naming the field ``steps`` avoids that ambiguity.

Loading is generic: every dataclass field is walked, unknown keys are rejected with their dotted
path, ``int`` values are accepted for ``float`` fields, ``bool`` is rejected where a number is
expected, lists become tuples for tuple fields and ``None`` is allowed for optional fields.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

TEST_EVENT = "turkey-earthquake"
PROTOCOL = "zero_shot"


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or holds an invalid value."""


# --- sections ---------------------------------------------------------------


@dataclass(frozen=True)
class Data:
    """Dataset split and loading."""

    protocol: str = PROTOCOL
    test_event: str = TEST_EVENT
    crop: int = 640
    workers: int = 8
    nodata_ignore: bool = True

    def __post_init__(self) -> None:
        if self.protocol != PROTOCOL:
            raise ConfigError(f"data.protocol: yalnız '{PROTOCOL}' destekleniyor, '{self.protocol}' verildi.")
        if self.test_event != TEST_EVENT:
            raise ConfigError(f"data.test_event: yalnız '{TEST_EVENT}' olabilir, '{self.test_event}' verildi.")
        if self.crop <= 0:
            raise ConfigError(f"data.crop: pozitif olmalı, {self.crop} verildi.")
        if self.workers < 0:
            raise ConfigError(f"data.workers: negatif olamaz, {self.workers} verildi.")


@dataclass(frozen=True)
class Aug:
    """Training-time augmentation."""

    flips: bool = True
    rot90: bool = True
    photometric: bool = False


@dataclass(frozen=True)
class Norm:
    """Input normalisation statistics."""

    optical: str = "imagenet"
    sar: str = "official"

    def __post_init__(self) -> None:
        if self.sar not in ("official", "train_stats"):
            raise ConfigError(f"norm.sar: 'official' veya 'train_stats' olmalı, '{self.sar}' verildi.")


@dataclass(frozen=True)
class Model:
    """Model selection and initial weights."""

    name: str = "changemamba_mmbda_tiny"
    init: str = "vssm_tiny_imagenet"
    fallback: str = "damageformer_swin_t"


@dataclass(frozen=True)
class Optim:
    """Optimizer settings."""

    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 5e-3

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ConfigError(f"optim.lr: pozitif olmalı, {self.lr} verildi.")
        if self.weight_decay < 0:
            raise ConfigError(f"optim.weight_decay: negatif olamaz, {self.weight_decay} verildi.")


@dataclass(frozen=True)
class Sched:
    """Learning-rate schedule."""

    type: str = "cosine"
    warmup_steps: int = 1500
    min_lr: float = 1e-6

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ConfigError(f"sched.warmup_steps: negatif olamaz, {self.warmup_steps} verildi.")
        if self.min_lr < 0:
            raise ConfigError(f"sched.min_lr: negatif olamaz, {self.min_lr} verildi.")


@dataclass(frozen=True)
class Train:
    """Training loop: ``steps`` counts optimizer steps, not samples.

    ``micro_batch`` is the largest chunk one forward/backward pass handles; the gradients of the
    chunks are accumulated into one optimizer step of ``batch`` samples. ``None`` means the whole
    batch; on CUDA out-of-memory the loop halves it automatically.
    """

    batch: int = 16
    micro_batch: int | None = None
    steps: int = 40000
    amp: str = "bf16"
    ema: float = 0.999
    clip: float | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch <= 0:
            raise ConfigError(f"train.batch: pozitif olmalı, {self.batch} verildi.")
        if self.micro_batch is not None and not 1 <= self.micro_batch <= self.batch:
            raise ConfigError(f"train.micro_batch: None veya 1..batch aralığında olmalı, {self.micro_batch} verildi.")
        if self.steps <= 0:
            raise ConfigError(f"train.steps: pozitif olmalı, {self.steps} verildi.")
        if self.amp not in ("bf16", "fp32"):
            raise ConfigError(f"train.amp: 'bf16' veya 'fp32' olmalı, '{self.amp}' verildi.")
        if not 0.0 <= self.ema <= 1.0:
            raise ConfigError(f"train.ema: [0, 1] aralığında olmalı, {self.ema} verildi.")
        if self.clip is not None and self.clip <= 0:
            raise ConfigError(f"train.clip: None veya pozitif olmalı, {self.clip} verildi.")


@dataclass(frozen=True)
class Loss:
    """Loss weights and class handling."""

    ce_loc: float = 1.0
    ce_clf: float = 1.0
    lovasz_loc: float = 0.5
    lovasz_clf: float = 0.75
    class_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    ignore_index: int = 255
    lovasz_fp32: bool = True

    def __post_init__(self) -> None:
        if len(self.class_weights) != 4:
            raise ConfigError(f"loss.class_weights: 4 sınıf ağırlığı bekleniyor, {len(self.class_weights)} verildi.")


@dataclass(frozen=True)
class Eval:
    """Validation cadence and checkpoint selection."""

    every: int = 2000
    select: str = "ema_val_miou"

    def __post_init__(self) -> None:
        if self.every <= 0:
            raise ConfigError(f"eval.every: pozitif olmalı, {self.every} verildi.")


@dataclass(frozen=True)
class Ckpt:
    """Checkpoint cadence and retention."""

    every: int = 1000
    keep_last: int = 2
    keep_best: int = 1
    hub_every: int = 5000

    def __post_init__(self) -> None:
        if self.every <= 0:
            raise ConfigError(f"ckpt.every: pozitif olmalı, {self.every} verildi.")
        if self.keep_last < 0:
            raise ConfigError(f"ckpt.keep_last: negatif olamaz, {self.keep_last} verildi.")
        if self.keep_best < 0:
            raise ConfigError(f"ckpt.keep_best: negatif olamaz, {self.keep_best} verildi.")
        if self.hub_every < 0:
            raise ConfigError(f"ckpt.hub_every: negatif olamaz, {self.hub_every} verildi.")


@dataclass(frozen=True)
class Config:
    """Whole run configuration; one frozen section per YAML block."""

    experiment: str = "ana"
    data: Data = field(default_factory=Data)
    aug: Aug = field(default_factory=Aug)
    norm: Norm = field(default_factory=Norm)
    model: Model = field(default_factory=Model)
    optim: Optim = field(default_factory=Optim)
    sched: Sched = field(default_factory=Sched)
    train: Train = field(default_factory=Train)
    loss: Loss = field(default_factory=Loss)
    eval: Eval = field(default_factory=Eval)
    ckpt: Ckpt = field(default_factory=Ckpt)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serialisable nested dict (tuples become lists)."""
        return _plain(dataclasses.asdict(self))


# --- generic loading --------------------------------------------------------


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _dotted(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _coerce(hint: Any, value: Any, path: str) -> Any:
    """Check ``value`` against ``hint`` and return the coerced value."""
    origin = get_origin(hint)

    if origin in (Union, types.UnionType):
        args = get_args(hint)
        if value is None and type(None) in args:
            return None
        inner = [arg for arg in args if arg is not type(None)]
        if len(inner) == 1:
            return _coerce(inner[0], value, path)
        raise ConfigError(f"{path}: desteklenmeyen tip birleşimi {hint!r}.")

    if dataclasses.is_dataclass(hint):
        return _build(hint, value, path)

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: liste bekleniyor, {type(value).__name__} verildi.")
        args = get_args(hint)
        element = args[0] if args else Any
        return tuple(_coerce(element, item, f"{path}[{index}]") for index, item in enumerate(value))

    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: bool bekleniyor, {type(value).__name__} verildi.")
        return value

    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: int bekleniyor, {type(value).__name__} verildi.")
        return value

    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: sayı bekleniyor, {type(value).__name__} verildi.")
        return float(value)

    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: str bekleniyor, {type(value).__name__} verildi.")
        return value

    return value


def _build(cls: type, payload: Any, path: str) -> Any:
    """Build a frozen dataclass from ``payload``, filling missing fields with defaults."""
    if not isinstance(payload, dict):
        raise ConfigError(f"{path or cls.__name__}: eşleme (mapping) bekleniyor, {type(payload).__name__} verildi.")

    hints = get_type_hints(cls)
    fields = {item.name: item for item in dataclasses.fields(cls)}

    for key in payload:
        if key not in fields:
            raise ConfigError(f"Bilinmeyen anahtar: {_dotted(path, str(key))}")

    kwargs = {}
    for name, item in fields.items():
        if name in payload:
            kwargs[name] = _coerce(hints[name], payload[name], _dotted(path, name))
        elif item.default is dataclasses.MISSING and item.default_factory is dataclasses.MISSING:
            raise ConfigError(f"Eksik anahtar: {_dotted(path, name)}")

    return cls(**kwargs)


def load_config(path: Path | str) -> Config:
    """Load a YAML file into a :class:`Config`; missing keys take their defaults."""
    text = Path(path).read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ConfigError(f"{path}: üst düzey eşleme (mapping) bekleniyor, {type(payload).__name__} verildi.")
    return _build(Config, payload, "")


# --- hashing ----------------------------------------------------------------


def config_hash(cfg: Config) -> str:
    """Return the first 12 hex chars of the canonical JSON sha256 of ``cfg``."""
    canonical = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def run_id(cfg: Config) -> str:
    """Return the run identifier: experiment name plus config hash."""
    return f"{cfg.experiment}-{config_hash(cfg)}"
