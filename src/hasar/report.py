"""README result tables generated from the JSON files under ``results/``.

Project rules 4 and 5: every number in the README comes from a ``results/**/*.json`` file
written by :mod:`hasar.results`, and every number carries its metric type tag. This module
collects those records and renders separate Markdown tables in the README's result blocks.
Each table is a pure function of the result files.
"""

from __future__ import annotations

from pathlib import Path

from hasar.results import ResultRecord, ResultSchemaError, read_result

#: Markers that delimit the generated blocks inside the README.
START = "<!-- sonuclar:basla -->"
END = "<!-- sonuclar:bitir -->"
FEWSHOT_START = "<!-- fewshot:basla -->"
FEWSHOT_END = "<!-- fewshot:bitir -->"
_FEWSHOT_HEADING = (
    "### Few-shot ek deneyi (Türkiye uyarlama etiketleri kullanılır; zero-shot ile ve 41,28 ile kıyaslanmaz)"
)

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
    if "damaged_diff_pct" in values and "destroyed_diff_pct" in values:
        parts = []
        for label, prefix in (("hasarlı", "damaged"), ("yıkık", "destroyed")):
            cell = f"{label} {format_number(values[f'{prefix}_diff_pct'])}"
            if f"{prefix}_low_pct" in values and f"{prefix}_high_pct" in values:
                low, high = values[f"{prefix}_low_pct"], values[f"{prefix}_high_pct"]
                cell += f" [{format_number(low)}; {format_number(high)}]"
            parts.append(cell)
        return " · ".join(parts)
    key = next((name for name in _VALUE_KEYS if name in values), None)
    if key is None:
        key = next(iter(values))
    text = format_number(values[key])
    if "ci_low_pct" in values and "ci_high_pct" in values:
        text += f" [{format_number(values['ci_low_pct'])}; {format_number(values['ci_high_pct'])}]"
    return text


def render_table(records: list[tuple[Path, ResultRecord]], *, headline: str = "[ZS-EVENT mIoU]") -> str:
    """Render the Markdown table with headline rows first."""
    ordered = sorted(records, key=lambda item: item[1].metric_tag != headline)
    lines = [_HEADER, _SEPARATOR]
    for path, record in ordered:
        lines.append(
            f"| {record.model} | {record.split} | {record.metric_tag} | "
            f"{_value_cell(record.values)} | `{path.as_posix()}` |"
        )
    return "\n".join(lines)


def _replace_block(text: str, start_marker: str, end_marker: str, body: str) -> str:
    """Replace the body between the markers, raising if the markers are invalid."""
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"README içinde {start_marker} ve {end_marker} işaretleri bulunamadı")
    return text[: start + len(start_marker)] + "\n" + body + text[end:]


def _render_readme(text: str, results: Path | str) -> str:
    zero_shot: list[tuple[Path, ResultRecord]] = []
    fewshot: list[tuple[Path, ResultRecord]] = []
    for path, record in collect(results):
        base_tag = record.metric_tag[1:-1].split(",", 1)[0].strip()
        target = fewshot if base_tag.startswith("FS-EVENT") else zero_shot
        target.append((path, record))

    text = _replace_block(text, START, END, f"{NOTE}\n\n{render_table(zero_shot)}\n")
    if not fewshot and (FEWSHOT_START not in text or FEWSHOT_END not in text):
        return text

    body = ""
    if fewshot:
        table = render_table(fewshot, headline="[FS-EVENT mIoU]")
        body = f"{_FEWSHOT_HEADING}\n\n{NOTE}\n\n{table}\n"
    return _replace_block(text, FEWSHOT_START, FEWSHOT_END, body)


def update_readme(readme: Path | str, results: Path | str) -> bool:
    """Rewrite the README result blocks from ``results``; return whether the file changed."""
    readme = Path(readme)
    text = readme.read_text(encoding="utf-8")
    new_text = _render_readme(text, results)
    if new_text == text:
        return False
    readme.write_text(new_text, encoding="utf-8")
    return True


def check_readme(readme: Path | str, results: Path | str) -> bool:
    """Return ``True`` when the README result blocks are current, without writing anything."""
    text = Path(readme).read_text(encoding="utf-8")
    return _render_readme(text, results) == text
