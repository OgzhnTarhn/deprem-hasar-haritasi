"""Check that a Colab notebook is clean and runs top to bottom without manual edits.

Usage: ``python scripts/defter_kontrol.py <ipynb>``

Every notebook must be nbformat 4, every code cell must be stripped (no outputs, no execution
count), and every code cell must parse as Python after dropping Colab magics (``!``/``%`` lines).
Profile selection is based on the file name:

- ``fewshot.ipynb``: checks the few-shot Turkey adaptation rules (no zero-shot knobs or
  ``selective_scan``, required few-shot commands and files, ordered ``# Hücre 1:``..``# Hücre 8:``
  code cells, ``if not LOCKED:`` + ``raise`` stop before any data/training/test command, and
  ``hasar fewshot-test`` only in Hücre 7 after both ``run_arm`` calls).
- Any other notebook: checks the zero-shot rules (no manual knobs or ``selective_scan``,
  includes ``configs/varyant.yaml``, ``hasar agirlik --config``, and ``--preflight``).

Exits 1 with a message on failure.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

#: Defter kodunda bulunmaması gereken elle doldurulan ayarlar ve eski çekirdek.
FORBIDDEN = ('REF = "protokol-kilidi"', "RUN_FINAL = False", "selective_scan")

#: Strings that must appear in the zero-shot notebook code.
REQUIRED = ("--preflight", "configs/varyant.yaml", "hasar agirlik --config")

#: Few-shot defter kodunda bulunmaması gereken ayarlar ve zero-shot komutları.
FEWSHOT_FORBIDDEN = (
    'REF = "protokol-kilidi"',
    "RUN_FINAL",
    "selective_scan",
    "hasar final ",
    "hasar sicaklik",
    "STEPS_OVERRIDE",
)

#: Few-shot defter kodunda bulunması gereken komutlar ve sabitler.
FEWSHOT_REQUIRED = (
    "--preflight",
    "configs/fewshot_a.yaml",
    "configs/fewshot_b.yaml",
    '"fewshot-kilidi"',
    "hasar sha256",
    "results/fewshot/bolme.json",
    "runtime.unassign()",
    "hasar fewshot-test",
    "test_done.json",
)

#: Kilit durdurmadan önce çalışmaması gereken komutlar.
FEWSHOT_GUARDED_COMMANDS = (
    "hasar on-kontrol",
    "hasar egit",
    "hasar indir",
    "hasar ornekler",
    "hasar fewshot-test",
)

_CELL_HEADER_RE = re.compile(r"^# Hücre (\d+):")


def _code_text(cell: dict) -> str:
    """Join a cell's ``source`` (a list of lines or a single string) into one text."""
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _strip_magics(text: str) -> str:
    """Drop Colab magic lines (``!``/``%``) so the rest parses as plain Python."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("!", "%")))


def _check_zeroshot(code_cells: list[tuple[int, str]]) -> list[str]:
    """Run zero-shot specific checks on code cells."""
    problems: list[str] = []
    code = "\n".join(text for _, text in code_cells)
    for needle in FORBIDDEN:
        if needle in code:
            problems.append(f"elle doldurulan ayar kaldı: {needle}")
    for needle in REQUIRED:
        if needle not in code:
            problems.append(f"eksik: {needle}")
    return problems


def _check_fewshot(code_cells: list[tuple[int, str]]) -> list[str]:
    """Run few-shot specific checks on code cells."""
    problems: list[str] = []
    code = "\n".join(text for _, text in code_cells)
    for needle in FEWSHOT_FORBIDDEN:
        if needle in code:
            problems.append(f"yasak ayar veya komut kaldı: {needle}")
    for needle in FEWSHOT_REQUIRED:
        if needle not in code:
            problems.append(f"eksik: {needle}")

    headers: list[int] = []
    header_by_n: dict[int, list[int]] = {}
    for cell_index, text in code_cells:
        first_line = text.splitlines()[0] if text.splitlines() else ""
        match = _CELL_HEADER_RE.match(first_line)
        if match:
            n = int(match.group(1))
            headers.append(n)
            header_by_n.setdefault(n, []).append(cell_index)

    for n in range(1, 9):
        if n not in header_by_n:
            problems.append(f"eksik hücre başlığı: # Hücre {n}:")
    for n, idxs in sorted(header_by_n.items()):
        if len(idxs) > 1:
            problems.append(f"yinelenen hücre başlığı: # Hücre {n}:")
    if headers != sorted(headers):
        problems.append(f"hücre sırası bozuk: {headers}")

    h7_indices = header_by_n.get(7, [])
    h7_index = h7_indices[0] if len(h7_indices) == 1 else None
    for cell_index, text in code_cells:
        if "hasar fewshot-test" in text and cell_index != h7_index:
            problems.append(f"hücre {cell_index}: hasar fewshot-test yalnız Hücre 7 içinde olmalı")

    arm0_indices = [idx for idx, text in code_cells if "run_arm(*RUNS[0])" in text]
    arm1_indices = [idx for idx, text in code_cells if "run_arm(*RUNS[1])" in text]
    if h7_index is not None and (
        not arm0_indices or not arm1_indices or max(arm0_indices) >= h7_index or max(arm1_indices) >= h7_index
    ):
        problems.append("hasar fewshot-test, run_arm(*RUNS[0]) ve run_arm(*RUNS[1]) hücrelerinden sonra gelmeli")

    lock_cell: tuple[int, str, int] | None = None
    for cell_index, text in code_cells:
        lock_pos = text.find("if not LOCKED:")
        if lock_pos == -1:
            continue
        match = re.search(r"\braise\b", text[lock_pos:])
        if match is not None:
            lock_cell = (cell_index, text, lock_pos + match.start())
            break

    if lock_cell is None:
        problems.append("kilit durdurma eksik: 'if not LOCKED:' sonrası 'raise' bulunamadı")
    else:
        lock_idx, lock_text, raise_pos = lock_cell
        for cell_index, text in code_cells:
            if cell_index < lock_idx:
                for cmd in FEWSHOT_GUARDED_COMMANDS:
                    if cmd in text:
                        problems.append(f"hücre {cell_index}: kilit durdurmadan önce yasak komut: {cmd}")
        before_raise = lock_text[:raise_pos]
        for cmd in FEWSHOT_GUARDED_COMMANDS:
            if cmd in before_raise:
                problems.append(f"hücre {lock_idx}: kilit durdurmadan önce yasak komut: {cmd}")

    return problems


def check(path: Path) -> list[str]:
    """Return the list of problems found in the notebook at ``path`` (empty when it is fine)."""
    problems: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"defter okunamadı: {exc}"]

    if notebook.get("nbformat") != 4:
        problems.append(f"nbformat 4 olmalı, gelen: {notebook.get('nbformat')}")

    code_cells: list[tuple[int, str]] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") != []:
            problems.append(f"hücre {index}: outputs boş değil")
        if cell.get("execution_count") is not None:
            problems.append(f"hücre {index}: execution_count None değil")
        text = _code_text(cell)
        code_cells.append((index, text))
        try:
            ast.parse(_strip_magics(text))
        except SyntaxError as exc:
            problems.append(f"hücre {index}: Python olarak ayrıştırılamadı: {exc}")

    if path.name == "fewshot.ipynb":
        problems.extend(_check_fewshot(code_cells))
    else:
        problems.extend(_check_zeroshot(code_cells))
    return problems


def main(argv: list[str]) -> int:
    """Check the notebook given on the command line and print the verdict."""
    if len(argv) != 2:
        print("kullanım: python scripts/defter_kontrol.py <ipynb>")
        return 1
    problems = check(Path(argv[1]))
    if problems:
        for problem in problems:
            print("HATA:", problem)
        return 1
    print("defter tamam")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
