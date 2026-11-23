import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("defter_kontrol", ROOT / "scripts" / "defter_kontrol.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def write_notebook(tmp_path, source, **overrides):
    cell = {"cell_type": "code", "source": source, "outputs": [], "execution_count": None}
    cell.update(overrides)
    path = tmp_path / "synthetic.ipynb"
    path.write_text(json.dumps({"nbformat": 4, "cells": [cell]}), encoding="utf-8")
    return path


CLEAN_SOURCE = 'command = "hasar agirlik --config configs/varyant.yaml --preflight"\n'


def test_repo_notebook_passes_check():
    assert checker.check(ROOT / "notebooks" / "ana_egitim.ipynb") == []


def test_repo_fewshot_notebook_passes_check():
    assert checker.check(ROOT / "notebooks" / "fewshot.ipynb") == []


def test_missing_variant_config_is_reported(tmp_path):
    path = write_notebook(tmp_path, 'command = "hasar agirlik --config configs/ana.yaml --preflight"\n')
    assert checker.check(path) == ["eksik: configs/varyant.yaml"]


def test_selective_scan_is_reported(tmp_path):
    path = write_notebook(tmp_path, CLEAN_SOURCE + "import selective_scan\n")
    problems = checker.check(path)
    assert len(problems) == 1
    assert "selective_scan" in problems[0]


@pytest.mark.parametrize(
    ("source", "overrides", "message"),
    [
        (CLEAN_SOURCE, {"outputs": [{"output_type": "stream", "text": "çıktı"}]}, "outputs boş değil"),
        (CLEAN_SOURCE, {"execution_count": 1}, "execution_count None değil"),
        (CLEAN_SOURCE + "if\n", {}, "Python olarak ayrıştırılamadı"),
    ],
)
def test_existing_cell_rules_are_reported(tmp_path, source, overrides, message):
    path = write_notebook(tmp_path, source, **overrides)
    problems = checker.check(path)
    assert len(problems) == 1
    assert message in problems[0]


def _fewshot_cells() -> list[dict]:
    sources = [
        (
            '# Hücre 1: kurulum\nTAG = "fewshot-kilidi"\n'
            'CFG = ("configs/fewshot_a.yaml", "configs/fewshot_b.yaml")\n'
            'SPLIT = "results/fewshot/bolme.json"\n'
        ),
        "# Hücre 2: ortam\nGPU = 'A100'\n",
        (
            '# Hücre 3: kilit ve veri\nc1 = "hasar sha256"\nLOCKED = False\n'
            'if not LOCKED:\n    raise RuntimeError("kilit")\nc2 = "hasar indir"\n'
        ),
        '# Hücre 4: ön kontrol ve eğitim\nc1 = "hasar on-kontrol --preflight"\nc2 = "hasar egit --preflight"\n',
        "# Hücre 5: A\nrun_arm(*RUNS[0])\n",
        "# Hücre 6: B\nrun_arm(*RUNS[1])\n",
        '# Hücre 7: test\nc1 = "hasar ornekler"\nc2 = "hasar fewshot-test"\nDONE = "test_done.json"\n',
        "# Hücre 8: bitir\nruntime.unassign()\n",
    ]
    return [{"cell_type": "code", "source": src, "outputs": [], "execution_count": None} for src in sources]


def _write_fewshot(tmp_path: Path, cells: list[dict]) -> Path:
    path = tmp_path / "fewshot.ipynb"
    path.write_text(json.dumps({"nbformat": 4, "cells": cells}), encoding="utf-8")
    return path


def test_synthetic_fewshot_passes(tmp_path):
    path = _write_fewshot(tmp_path, _fewshot_cells())
    assert checker.check(path) == []


def test_fewshot_missing_cell_header_is_reported(tmp_path):
    cells = _fewshot_cells()
    cells[1]["source"] = "# ortam\nGPU = 'A100'\n"
    problems = checker.check(_write_fewshot(tmp_path, cells))
    assert any("eksik hücre başlığı: # Hücre 2:" in p for p in problems)


def test_fewshot_test_outside_cell_7_is_reported(tmp_path):
    cells = _fewshot_cells()
    cells[4]["source"] += 'extra = "hasar fewshot-test"\n'
    problems = checker.check(_write_fewshot(tmp_path, cells))
    assert any("yalnız Hücre 7 içinde olmalı" in p for p in problems)


def test_fewshot_test_before_run_arm_is_reported(tmp_path):
    cells = _fewshot_cells()
    cells[4], cells[6] = cells[6], cells[4]
    problems = checker.check(_write_fewshot(tmp_path, cells))
    assert any("run_arm(*RUNS[0]) ve run_arm(*RUNS[1])" in p for p in problems)


def test_fewshot_missing_lock_stop_is_reported(tmp_path):
    cells = _fewshot_cells()
    cells[2]["source"] = '# Hücre 3: kilit ve veri\nc1 = "hasar sha256"\nc2 = "hasar indir"\n'
    problems = checker.check(_write_fewshot(tmp_path, cells))
    assert any("kilit durdurma eksik" in p for p in problems)


def test_fewshot_on_kontrol_before_lock_stop_is_reported(tmp_path):
    cells = _fewshot_cells()
    cells[1]["source"] += 'early = "hasar on-kontrol"\n'
    problems = checker.check(_write_fewshot(tmp_path, cells))
    assert any("kilit durdurmadan önce yasak komut: hasar on-kontrol" in p for p in problems)


def test_fewshot_cell_with_outputs_is_reported(tmp_path):
    cells = _fewshot_cells()
    cells[0]["outputs"] = [{"output_type": "stream", "text": "çıktı"}]
    problems = checker.check(_write_fewshot(tmp_path, cells))
    assert any("outputs boş değil" in p for p in problems)
