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
