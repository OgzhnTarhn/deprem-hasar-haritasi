"""Tests for hasar.config: frozen dataclasses, YAML loading, validation and hashing."""

import dataclasses
import json
import re
from pathlib import Path

import pytest
import yaml
from hasar.config import Config, ConfigError, config_hash, load_config, run_id

REPO_ROOT = Path(__file__).parents[1]


def write_yaml(tmp_path: Path, payload: dict, name: str = "cfg.yaml") -> Path:
    """Write a YAML document to tmp_path and return its path."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def load_payload(tmp_path: Path, payload: dict) -> Config:
    return load_config(write_yaml(tmp_path, payload))


# --- defaults ---------------------------------------------------------------


def test_defaults() -> None:
    cfg = Config()
    assert cfg.experiment == "ana"

    assert cfg.data.protocol == "zero_shot"
    assert cfg.data.test_event == "turkey-earthquake"
    assert cfg.data.crop == 640
    assert cfg.data.workers == 8
    assert cfg.data.nodata_ignore is True

    assert cfg.aug.flips is True
    assert cfg.aug.rot90 is True
    assert cfg.aug.photometric is False

    assert cfg.norm.optical == "imagenet"
    assert cfg.norm.sar == "official"

    assert cfg.model.name == "changemamba_mmbda_tiny"
    assert cfg.model.init == "vssm_tiny_imagenet"
    assert cfg.model.fallback == "damageformer_swin_t"

    assert cfg.optim.name == "adamw"
    assert cfg.optim.lr == 1e-4
    assert cfg.optim.weight_decay == 5e-3

    assert cfg.sched.type == "cosine"
    assert cfg.sched.warmup_steps == 1500
    assert cfg.sched.min_lr == 1e-6

    assert cfg.train.batch == 16
    assert cfg.train.steps == 40000
    assert cfg.train.amp == "bf16"
    assert cfg.train.ema == 0.999
    assert cfg.train.clip is None
    assert cfg.train.seed == 42

    assert cfg.loss.ce_loc == 1.0
    assert cfg.loss.ce_clf == 1.0
    assert cfg.loss.lovasz_loc == 0.5
    assert cfg.loss.lovasz_clf == 0.75
    assert tuple(cfg.loss.class_weights) == (1.0, 1.0, 1.0, 1.0)
    assert cfg.loss.ignore_index == 255
    assert cfg.loss.lovasz_fp32 is True

    assert cfg.eval.every == 2000
    assert cfg.eval.select == "ema_val_miou"

    assert cfg.ckpt.every == 1000
    assert cfg.ckpt.keep_last == 2
    assert cfg.ckpt.keep_best == 1
    assert cfg.ckpt.hub_every == 5000


def test_config_and_sections_are_frozen() -> None:
    cfg = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.experiment = "baska"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.train.batch = 8  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.optim.lr = 1e-3  # type: ignore[misc]


# --- loading ----------------------------------------------------------------


def test_empty_yaml_gives_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == Config()


def test_missing_keys_take_defaults(tmp_path: Path) -> None:
    cfg = load_payload(tmp_path, {"train": {"batch": 8}})
    assert cfg.train.batch == 8
    assert cfg.train.steps == 40000
    assert cfg.optim.lr == 1e-4
    assert cfg.data.crop == 640


def test_partial_sections_merge_with_defaults(tmp_path: Path) -> None:
    cfg = load_payload(
        tmp_path,
        {
            "experiment": "varyant",
            "data": {"crop": 512, "workers": 4},
            "optim": {"lr": 2e-4},
            "loss": {"class_weights": [1.0, 2.0, 3.0, 4.0]},
        },
    )
    assert cfg.experiment == "varyant"
    assert cfg.data.crop == 512
    assert cfg.data.workers == 4
    assert cfg.data.test_event == "turkey-earthquake"
    assert cfg.optim.lr == 2e-4
    assert cfg.optim.weight_decay == 5e-3
    assert tuple(cfg.loss.class_weights) == (1.0, 2.0, 3.0, 4.0)


def test_valid_non_default_values_load(tmp_path: Path) -> None:
    cfg = load_payload(
        tmp_path,
        {
            "norm": {"sar": "train_stats"},
            "train": {"amp": "fp32", "clip": 1.0},
            "sched": {"type": "linear"},
        },
    )
    assert cfg.norm.sar == "train_stats"
    assert cfg.train.amp == "fp32"
    assert cfg.train.clip == 1.0
    assert cfg.sched.type == "linear"


# --- unknown keys -----------------------------------------------------------


def test_unknown_nested_key_reports_dotted_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_payload(tmp_path, {"train": {"batchsize": 16}})
    assert "train.batchsize" in str(excinfo.value)


def test_unknown_key_in_other_section_reports_dotted_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_payload(tmp_path, {"data": {"crop_size": 512}})
    assert "data.crop_size" in str(excinfo.value)


def test_unknown_top_level_key_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_payload(tmp_path, {"bilinmeyen": 1})
    assert "bilinmeyen" in str(excinfo.value)


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"data": {"test_event": "la_palma-volcano"}}, id="test_event"),
        pytest.param({"data": {"protocol": "standard_ML"}}, id="protocol"),
        pytest.param({"norm": {"sar": "minmax"}}, id="norm_sar"),
        pytest.param({"train": {"amp": "fp16"}}, id="amp"),
        pytest.param({"train": {"clip": 0.0}}, id="clip_zero"),
        pytest.param({"train": {"clip": -1.0}}, id="clip_negative"),
        pytest.param({"train": {"steps": 0}}, id="steps_zero"),
        pytest.param({"train": {"steps": -10}}, id="steps_negative"),
        pytest.param({"train": {"batch": 0}}, id="batch_zero"),
        pytest.param({"train": {"batch": -4}}, id="batch_negative"),
        pytest.param({"eval": {"every": 0}}, id="eval_every_zero"),
        pytest.param({"eval": {"every": -2000}}, id="eval_every_negative"),
        pytest.param({"ckpt": {"every": 0}}, id="ckpt_every_zero"),
        pytest.param({"ckpt": {"every": -1000}}, id="ckpt_every_negative"),
    ],
)
def test_invalid_values_raise_config_error(tmp_path: Path, payload: dict) -> None:
    with pytest.raises(ConfigError):
        load_payload(tmp_path, payload)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"train": {"batch": "16"}}, id="string_for_int"),
        pytest.param({"train": {"batch": True}}, id="bool_for_int"),
        pytest.param({"train": {"steps": False}}, id="bool_for_steps"),
        pytest.param({"data": {"crop": "640"}}, id="string_for_crop"),
        pytest.param({"optim": {"lr": "1e-4"}}, id="string_for_float"),
        pytest.param({"train": {"ema": "0.999"}}, id="string_for_ema"),
        pytest.param({"train": {"clip": "1.0"}}, id="string_for_clip"),
        pytest.param({"data": {"nodata_ignore": "true"}}, id="string_for_bool"),
        pytest.param({"aug": {"flips": 1}}, id="int_for_bool"),
        pytest.param({"loss": {"class_weights": 1.0}}, id="scalar_for_list"),
    ],
)
def test_wrong_types_raise_config_error(tmp_path: Path, payload: dict) -> None:
    with pytest.raises(ConfigError):
        load_payload(tmp_path, payload)


def test_int_is_accepted_for_float_field(tmp_path: Path) -> None:
    cfg = load_payload(tmp_path, {"optim": {"lr": 1}, "train": {"ema": 1}})
    assert cfg.optim.lr == 1.0
    assert isinstance(cfg.optim.lr, float)
    assert cfg.train.ema == 1.0
    assert isinstance(cfg.train.ema, float)


# --- hashing ----------------------------------------------------------------

HEX12 = re.compile(r"^[0-9a-f]{12}$")


def test_config_hash_format() -> None:
    digest = config_hash(Config())
    assert isinstance(digest, str)
    assert HEX12.match(digest), digest


def test_config_hash_is_deterministic() -> None:
    assert config_hash(Config()) == config_hash(Config())


def test_config_hash_independent_of_yaml_key_order(tmp_path: Path) -> None:
    first = write_yaml(
        tmp_path,
        {"train": {"batch": 8, "steps": 1000}, "optim": {"lr": 2e-4, "name": "adamw"}},
        name="first.yaml",
    )
    second = write_yaml(
        tmp_path,
        {"optim": {"name": "adamw", "lr": 2e-4}, "train": {"steps": 1000, "batch": 8}},
        name="second.yaml",
    )
    assert config_hash(load_config(first)) == config_hash(load_config(second))


def test_config_hash_matches_explicit_defaults(tmp_path: Path) -> None:
    explicit = {
        "experiment": "ana",
        "data": {
            "protocol": "zero_shot",
            "test_event": "turkey-earthquake",
            "crop": 640,
            "workers": 8,
            "nodata_ignore": True,
        },
        "aug": {"flips": True, "rot90": True, "photometric": False},
        "norm": {"optical": "imagenet", "sar": "official"},
        "model": {
            "name": "changemamba_mmbda_tiny",
            "init": "vssm_tiny_imagenet",
            "fallback": "damageformer_swin_t",
        },
        "optim": {"name": "adamw", "lr": 1e-4, "weight_decay": 5e-3},
        "sched": {"type": "cosine", "warmup_steps": 1500, "min_lr": 1e-6},
        "train": {
            "batch": 16,
            "steps": 40000,
            "amp": "bf16",
            "ema": 0.999,
            "clip": None,
            "seed": 42,
        },
        "loss": {
            "ce_loc": 1.0,
            "ce_clf": 1.0,
            "lovasz_loc": 0.5,
            "lovasz_clf": 0.75,
            "class_weights": [1.0, 1.0, 1.0, 1.0],
            "ignore_index": 255,
            "lovasz_fp32": True,
        },
        "eval": {"every": 2000, "select": "ema_val_miou"},
        "ckpt": {"every": 1000, "keep_last": 2, "keep_best": 1, "hub_every": 5000},
    }
    cfg = load_payload(tmp_path, explicit)
    assert cfg == Config()
    assert config_hash(cfg) == config_hash(Config())


def test_config_hash_changes_with_value(tmp_path: Path) -> None:
    base = config_hash(Config())
    changed = config_hash(load_payload(tmp_path, {"optim": {"lr": 2e-4}}))
    assert changed != base


def test_config_hash_changes_for_each_section(tmp_path: Path) -> None:
    base = config_hash(Config())
    overrides = [
        {"experiment": "varyant"},
        {"data": {"crop": 512}},
        {"aug": {"photometric": True}},
        {"norm": {"sar": "train_stats"}},
        {"model": {"name": "damageformer_swin_t"}},
        {"optim": {"weight_decay": 1e-2}},
        {"sched": {"warmup_steps": 100}},
        {"train": {"seed": 7}},
        {"loss": {"lovasz_clf": 1.0}},
        {"eval": {"every": 1000}},
        {"ckpt": {"keep_last": 3}},
    ]
    hashes = {config_hash(load_payload(tmp_path, override)) for override in overrides}
    assert len(hashes) == len(overrides)
    assert base not in hashes


# --- run_id -----------------------------------------------------------------


def test_run_id() -> None:
    cfg = Config()
    assert run_id(cfg) == f"ana-{config_hash(cfg)}"


def test_run_id_uses_experiment(tmp_path: Path) -> None:
    cfg = load_payload(tmp_path, {"experiment": "varyant"})
    assert run_id(cfg) == f"varyant-{config_hash(cfg)}"
    assert run_id(cfg) != run_id(Config())


# --- to_dict ----------------------------------------------------------------


def test_to_dict_is_json_serialisable() -> None:
    cfg = Config()
    payload = cfg.to_dict()
    text = json.dumps(payload)
    assert json.loads(text) == payload


def test_to_dict_is_plain_nested() -> None:
    payload = Config().to_dict()
    assert isinstance(payload, dict)
    assert payload["experiment"] == "ana"
    assert isinstance(payload["train"], dict)
    assert payload["train"]["batch"] == 16
    assert payload["train"]["clip"] is None
    assert isinstance(payload["loss"]["class_weights"], list)
    assert payload["loss"]["class_weights"] == [1.0, 1.0, 1.0, 1.0]
    assert not any(isinstance(value, tuple) for value in payload["loss"].values())


def test_to_dict_reflects_values(tmp_path: Path) -> None:
    cfg = load_payload(tmp_path, {"optim": {"lr": 2e-4}, "train": {"batch": 8}})
    payload = cfg.to_dict()
    assert payload["optim"]["lr"] == 2e-4
    assert payload["train"]["batch"] == 8


# --- repository config ------------------------------------------------------


def test_repo_ana_yaml_matches_defaults() -> None:
    path = REPO_ROOT / "configs" / "ana.yaml"
    cfg = load_config(path)
    assert cfg == Config()
    assert config_hash(cfg) == config_hash(Config())
