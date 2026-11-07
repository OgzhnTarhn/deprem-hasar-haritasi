"""README result table generated from the JSON files under ``results/``.

Project rules 4 and 5: every number in the README comes from a ``results/**/*.json`` file
written by :mod:`hasar.results`, and every number carries its metric type tag. This module
collects those records, renders them as a Markdown table and replaces the block between the
``<!-- sonuclar:basla -->`` and ``<!-- sonuclar:bitir -->`` markers of the README. Nothing is
typed by hand: the table is a pure function of the result files.
"""

from __future__ import annotations

from pathlib import Path

from hasar.results import ResultRecord, ResultSchemaError, read_result

#: Markers that delimit the generated block inside the README.
START = "<!-- sonuclar:basla -->"
END = "<!-- sonuclar:bitir -->"

#: Line placed above the table so readers know the block is generated.
NOTE = "Bu tablo `hasar readme` ile results/ dosyalarından üretildi; elle düzenlemeyin."

#: Value keys in display priority order; the first one present in a record is shown.
_VALUE_KEYS: tuple[str, ...] = ("miou_pct", "macro_f1_pct", "best_miou_pct", "diff_pct")

_HEADER = "| Model | Bölme | Metrik | Değer | Kaynak |"
_SEPARATOR = "| --- | --- | --- | --- | --- |"


def format_number(value: float) -> str:
    """Format ``value`` with two decimals and a Turkish decimal comma (``41.284`` -> ``41,28``)."""
    return f"{value:.2f}".replace(".", ",")


def collect(results: Path | str) -> list[tuple[Path, ResultRecord]]:
    """Return every valid result under ``results`` as ``(relative path, record)``, sorted by path.

    Data summaries (no ``metric_tag``), broken JSON and records with an invalid tag are skipped
    silently; only files that :func:`hasar.results.read_result` accepts are reported.
    """
    root = Path(results)
    collected: list[tuple[Path, ResultRecord]] = []
    for path in root.rglob("*.json"):
        try:
            record = read_result(path)
        except (ResultSchemaError, OSError):
            continue
        collected.append((path.relative_to(root), record))
    collected.sort(key=lambda item: item[0].as_posix())
    return collected


def _value_cell(values: dict[str, float]) -> str:
    """Render the value cell, preferring the headline metric and appending a CI when present."""
    key = next((name for name in _VALUE_KEYS if name in values), None)
    if key is None:
        key = next(iter(values))
    text = format_number(values[key])
    if "ci_low_pct" in values and "ci_high_pct" in values:
        text += f" [{format_number(values['ci_low_pct'])}; {format_number(values['ci_high_pct'])}]"
    return text


def render_table(records: list[tuple[Path, ResultRecord]]) -> str:
    """Render the Markdown table; the plain ``[ZS-EVENT mIoU]`` row comes first."""
    ordered = sorted(records, key=lambda item: 0 if item[1].metric_tag == "[ZS-EVENT mIoU]" else 1)
    lines = [_HEADER, _SEPARATOR]
    for path, record in ordered:
        lines.append(
            f"| {record.model} | {record.split} | {record.metric_tag} | "
            f"{_value_cell(record.values)} | `{path.as_posix()}` |"
        )
    return "\n".join(lines)


def _render_block(results: Path | str) -> str:
    """Return the full marker-delimited block for the current result files."""
    return f"{START}\n{NOTE}\n\n{render_table(collect(results))}\n{END}"


def _replace_block(text: str, block: str) -> str:
    """Replace the text between the markers with ``block``; raise ``ValueError`` if they are missing."""
    start = text.find(START)
    end = text.find(END)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"README içinde {START} ve {END} işaretleri bulunamadı")
    return text[:start] + block + text[end + len(END) :]


def update_readme(readme: Path | str, results: Path | str) -> bool:
    """Rewrite the README result block from ``results``; return whether the file changed."""
    readme = Path(readme)
    text = readme.read_text(encoding="utf-8")
    new_text = _replace_block(text, _render_block(results))
    if new_text == text:
        return False
    readme.write_text(new_text, encoding="utf-8")
    return True


def check_readme(readme: Path | str, results: Path | str) -> bool:
    """Return ``True`` when the README block is already current, without writing anything."""
    text = Path(readme).read_text(encoding="utf-8")
    return _replace_block(text, _render_block(results)) == text
