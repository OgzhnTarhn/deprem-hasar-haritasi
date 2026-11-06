"""Result schema and JSON persistence for every reported number.

Project rules 4 and 5: every number in the README and in the reports must come from a
``ResultRecord`` JSON file under ``results/``, and every number carries its metric type tag
(``[ZS-EVENT mIoU]``, ``[STD-EVENT mIoU]``, ...). ``metric_tag`` is therefore mandatory: a record
without a valid tag cannot be built, written or read.

Writes are atomic: the payload goes to a temporary file in the target directory, is flushed and
``fsync``-ed, and only then replaces the destination with :func:`os.replace`. A failed write never
leaves a half-written result file behind.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

#: Base metric tags, without the surrounding brackets. Rule 4: the tag names the metric type.
BASE_TAGS: tuple[str, ...] = (
    "ZS-EVENT mIoU",
    "ZS-EVENT bina-makro-F1",
    "STD-EVENT mIoU",
    "STD-SET mIoU",
    "DFC-TEST mIoU",
    "CVPR-TEST segm_AP",
    "VAL13 mIoU",
)

_QUALIFIER_RE = re.compile(r"[a-z0-9][a-z0-9-]*")
_REQUIRED_STRINGS = ("model", "split", "commit", "config_hash")


class ResultSchemaError(ValueError):
    """Raised when a result record or its JSON payload violates the schema."""


def validate_metric_tag(tag: str) -> None:
    """Validate a metric tag such as ``[ZS-EVENT mIoU]`` or ``[VAL13 mIoU, seed-0]``.

    The tag must be bracketed, its base must be one of :data:`BASE_TAGS` and the optional
    qualifier after the comma must match ``[a-z0-9][a-z0-9-]*``. Raises
    :class:`ResultSchemaError` otherwise.
    """
    if not isinstance(tag, str):
        raise ResultSchemaError(f"metric_tag metin olmalı, alınan: {type(tag).__name__}")
    if not (tag.startswith("[") and tag.endswith("]")):
        raise ResultSchemaError(f"metric_tag köşeli parantez içinde olmalı: {tag!r}")
    inner = tag[1:-1]
    base, sep, qualifier = inner.partition(",")
    if base not in BASE_TAGS:
        raise ResultSchemaError(f"bilinmeyen metric_tag tabanı: {base!r} (izinli: {BASE_TAGS})")
    if sep:
        # Exactly one space after the comma, then a lowercase qualifier such as ``flip-tta``.
        if not qualifier.startswith(" ") or not _QUALIFIER_RE.fullmatch(qualifier[1:]):
            raise ResultSchemaError(f"geçersiz metric_tag niteleyicisi: {qualifier!r}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_values(values: object) -> None:
    if not isinstance(values, dict) or not values:
        raise ResultSchemaError("values boş olmayan bir sözlük olmalı")
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ResultSchemaError(f"values anahtarı boş olmayan metin olmalı: {key!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ResultSchemaError(f"values[{key!r}] sayı olmalı, alınan: {value!r}")
        if not math.isfinite(value):
            raise ResultSchemaError(f"values[{key!r}] sonlu olmalı, alınan: {value!r}")


@dataclass(frozen=True)
class ResultRecord:
    """One reported number (or a small group of numbers) with its provenance.

    ``metric_tag`` is mandatory and validated on construction; ``created_at`` defaults to the
    current UTC time in ISO-8601 form.
    """

    metric_tag: str
    values: dict[str, float]
    model: str
    split: str
    commit: str
    config_hash: str
    seed: int | None = None
    extra: dict[str, object] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        validate_metric_tag(self.metric_tag)
        _validate_values(self.values)
        for name in _REQUIRED_STRINGS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ResultSchemaError(f"{name} boş olmayan metin olmalı, alınan: {value!r}")
        if not isinstance(self.created_at, str) or not self.created_at:
            raise ResultSchemaError("created_at boş olmayan ISO-8601 metni olmalı")
        if not isinstance(self.extra, dict):
            raise ResultSchemaError("extra sözlük olmalı")


def write_result(path: Path | str, record: ResultRecord) -> Path:
    """Write ``record`` to ``path`` as UTF-8, indented, key-sorted JSON, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(asdict(record), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
    return path


def read_result(path: Path | str) -> ResultRecord:
    """Read a result JSON file and return the validated :class:`ResultRecord`."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResultSchemaError(f"geçersiz JSON ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise ResultSchemaError(f"sonuç dosyası bir JSON nesnesi olmalı ({path})")
    if "metric_tag" not in payload:
        raise ResultSchemaError(f"metric_tag zorunlu, dosyada yok ({path})")
    try:
        return ResultRecord(**payload)
    except TypeError as exc:
        raise ResultSchemaError(f"sonuç şeması uyuşmuyor ({path}): {exc}") from exc
