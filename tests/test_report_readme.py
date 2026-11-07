"""Tests for the README result table generator (hasar.report) and the ``readme`` CLI.

Rule 5: every number in the README comes from ``results/**/*.json``. These tests build result
files with :func:`hasar.results.write_result` in ``tmp_path`` and check that the rendered table,
the README block update and the ``hasar readme`` subcommand all read from those files only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hasar.cli import main
from hasar.report import END, START, collect, format_number, render_table, update_readme
from hasar.results import ResultRecord, write_result

NOTE = "Bu tablo `hasar readme` ile results/ dosyalarından üretildi; elle düzenlemeyin."


def make_record(**overrides: object) -> ResultRecord:
    """Build a valid record, letting each test override single fields."""
    fields: dict[str, object] = {
        "metric_tag": "[ZS-EVENT mIoU]",
        "values": {"miou_pct": 41.28},
        "model": "changemamba",
        "split": "turkey-earthquake",
        "commit": "0123456789abcdef",
        "config_hash": "deadbeef",
    }
    fields.update(overrides)
    return ResultRecord(**fields)  # type: ignore[arg-type]


def write(tmp_path: Path, rel: str, record: ResultRecord) -> Path:
    """Write ``record`` under ``tmp_path / rel`` and return the path."""
    return write_result(tmp_path / rel, record)


def readme_with_block(body: str = "eski tablo") -> str:
    """Return a README text with the result block between the markers."""
    return f"# Başlık\n\nGiriş metni.\n\n{START}\n{body}\n{END}\n\nKapanış metni.\n"


# --- markers ---------------------------------------------------------------


def test_markers_are_exact() -> None:
    assert START == "<!-- sonuclar:basla -->"
    assert END == "<!-- sonuclar:bitir -->"


# --- format_number ---------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (41.284, "41,28"),
        (41.28, "41,28"),
        (0.0, "0,00"),
        (100, "100,00"),
        (-1.5, "-1,50"),
        (-1.234, "-1,23"),
        (7.0, "7,00"),
    ],
)
def test_format_number_uses_turkish_comma(value: float, expected: str) -> None:
    assert format_number(value) == expected


# --- collect ---------------------------------------------------------------


def test_collect_returns_records_sorted_by_path(tmp_path: Path) -> None:
    write(tmp_path, "arastirma/b.json", make_record(metric_tag="[VAL13 mIoU]", values={"miou_pct": 50.0}))
    write(tmp_path, "ana/a.json", make_record(values={"miou_pct": 41.28}))

    collected = collect(tmp_path)

    assert [Path(path).name for path, _ in collected] == ["a.json", "b.json"]
    assert [record.values["miou_pct"] for _, record in collected] == [41.28, 50.0]


def test_collect_skips_data_summaries_and_invalid_files(tmp_path: Path) -> None:
    write(tmp_path, "ana/ok.json", make_record())
    # A data summary carries ``kind`` instead of ``metric_tag``.
    (tmp_path / "veri-ozet.json").write_text(
        json.dumps({"kind": "data-summary", "counts": {"train": 777}}), encoding="utf-8"
    )
    # Broken JSON and a record with an invalid tag are skipped silently.
    (tmp_path / "bozuk.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "gecersiz.json").write_text(
        json.dumps(
            {
                "metric_tag": "[mIoU]",
                "values": {"miou_pct": 1.0},
                "model": "m",
                "split": "s",
                "commit": "c",
                "config_hash": "h",
            }
        ),
        encoding="utf-8",
    )

    collected = collect(tmp_path)

    assert [Path(path).name for path, _ in collected] == ["ok.json"]


def test_collect_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    assert collect(tmp_path) == []


def test_collect_ignores_non_json_files(tmp_path: Path) -> None:
    write(tmp_path, "ana/ok.json", make_record())
    (tmp_path / "notlar.txt").write_text("metin", encoding="utf-8")

    assert [Path(path).as_posix() for path, _ in collect(tmp_path)] == ["ana/ok.json"]


# --- render_table ----------------------------------------------------------


def test_render_table_header_and_row(tmp_path: Path) -> None:
    write(tmp_path, "ana/final.json", make_record(values={"miou_pct": 41.284}))

    table = render_table(collect(tmp_path))
    lines = table.splitlines()

    assert lines[0] == "| Model | Bölme | Metrik | Değer | Kaynak |"
    assert "---" in lines[1]
    assert "changemamba" in lines[2]
    assert "turkey-earthquake" in lines[2]
    assert "[ZS-EVENT mIoU]" in lines[2]
    assert "41,28" in lines[2]
    assert "`ana/final.json`" in lines[2]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"miou_pct": 41.284, "macro_f1_pct": 60.0}, "41,28"),
        ({"macro_f1_pct": 60.123, "best_miou_pct": 50.0}, "60,12"),
        ({"best_miou_pct": 50.0, "diff_pct": 1.0}, "50,00"),
        ({"diff_pct": -1.234}, "-1,23"),
    ],
)
def test_render_table_value_key_priority(values: dict[str, float], expected: str) -> None:
    record = make_record(values=values)

    table = render_table([(Path("ana/x.json"), record)])
    cells = [cell.strip() for cell in table.splitlines()[2].strip("|").split("|")]

    assert cells[3] == expected


def test_render_table_shows_confidence_interval() -> None:
    record = make_record(values={"miou_pct": 41.284, "ci_low_pct": 40.1, "ci_high_pct": 42.5})

    table = render_table([(Path("ana/x.json"), record)])
    cells = [cell.strip() for cell in table.splitlines()[2].strip("|").split("|")]

    assert cells[3] == "41,28 [40,10; 42,50]"


def test_render_table_lists_plain_zs_event_first() -> None:
    other = make_record(metric_tag="[VAL13 mIoU]", values={"miou_pct": 50.0})
    primary = make_record(metric_tag="[ZS-EVENT mIoU]", values={"miou_pct": 41.28})
    qualified = make_record(metric_tag="[ZS-EVENT mIoU, flip-tta]", values={"miou_pct": 41.5})
    records = [
        (Path("a/other.json"), other),
        (Path("b/primary.json"), primary),
        (Path("c/qualified.json"), qualified),
    ]

    table = render_table(records)
    rows = table.splitlines()[2:]

    assert "`b/primary.json`" in rows[0]
    assert "`a/other.json`" in rows[1]
    assert "`c/qualified.json`" in rows[2]


def test_render_table_empty_records_has_header_only() -> None:
    lines = render_table([]).splitlines()

    assert lines[0] == "| Model | Bölme | Metrik | Değer | Kaynak |"
    assert len(lines) == 2


# --- update_readme ---------------------------------------------------------


def test_update_readme_replaces_block_and_keeps_markers(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write(results, "ana/final.json", make_record(values={"miou_pct": 41.284}))
    readme = tmp_path / "README.md"
    readme.write_text(readme_with_block(), encoding="utf-8")

    changed = update_readme(readme, results)

    assert changed is True
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("# Başlık\n\nGiriş metni.\n\n")
    assert text.endswith("\n\nKapanış metni.\n")
    assert text.count(START) == 1
    assert text.count(END) == 1
    assert "eski tablo" not in text
    assert "| Model | Bölme | Metrik | Değer | Kaynak |" in text
    assert "`ana/final.json`" in text
    assert NOTE in text
    assert text.index(START) < text.index(NOTE) < text.index(END)


def test_update_readme_returns_false_when_already_current(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write(results, "ana/final.json", make_record())
    readme = tmp_path / "README.md"
    readme.write_text(readme_with_block(), encoding="utf-8")

    assert update_readme(readme, results) is True
    assert update_readme(readme, results) is False


def test_update_readme_missing_markers_raises(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write(results, "ana/final.json", make_record())
    readme = tmp_path / "README.md"
    readme.write_text("# Başlık\n\nİşaret yok.\n", encoding="utf-8")

    with pytest.raises(ValueError):
        update_readme(readme, results)


# --- hasar readme CLI ------------------------------------------------------


def test_cli_readme_updates_file(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write(results, "ana/final.json", make_record(values={"miou_pct": 41.284}))
    readme = tmp_path / "README.md"
    readme.write_text(readme_with_block(), encoding="utf-8")

    code = main(["readme", "--readme", str(readme), "--results", str(results)])

    assert code == 0
    text = readme.read_text(encoding="utf-8")
    assert "41,28" in text
    assert "`ana/final.json`" in text
    assert NOTE in text


def test_cli_readme_check_returns_zero_when_current(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write(results, "ana/final.json", make_record())
    readme = tmp_path / "README.md"
    readme.write_text(readme_with_block(), encoding="utf-8")
    assert main(["readme", "--readme", str(readme), "--results", str(results)]) == 0

    code = main(["readme", "--readme", str(readme), "--results", str(results), "--check"])

    assert code == 0


def test_cli_readme_check_returns_one_and_does_not_write_when_stale(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write(results, "ana/final.json", make_record())
    readme = tmp_path / "README.md"
    readme.write_text(readme_with_block(), encoding="utf-8")
    before = readme.read_text(encoding="utf-8")

    code = main(["readme", "--readme", str(readme), "--results", str(results), "--check"])

    assert code == 1
    assert readme.read_text(encoding="utf-8") == before


def test_cli_readme_check_detects_new_result(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write(results, "ana/final.json", make_record())
    readme = tmp_path / "README.md"
    readme.write_text(readme_with_block(), encoding="utf-8")
    assert main(["readme", "--readme", str(readme), "--results", str(results)]) == 0

    write(results, "ana/yeni.json", make_record(metric_tag="[VAL13 mIoU]", values={"miou_pct": 50.0}))

    assert main(["readme", "--readme", str(readme), "--results", str(results), "--check"]) == 1
