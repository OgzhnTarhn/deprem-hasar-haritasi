"""Check that a Colab notebook is clean and runs top to bottom without manual edits.

Usage: ``python scripts/defter_kontrol.py <ipynb>``

The notebook must be nbformat 4, every code cell must be stripped (no outputs, no execution
count), every code cell must parse as Python after dropping Colab magics (``!``/``%`` lines),
and the code must not keep manual knobs (``REF = "protokol-kilidi"``, ``RUN_FINAL = False``)
while it must pass ``--preflight`` to the training command. Exits 1 with a message on failure.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

#: Strings that must not appear in the notebook code (manual knobs the user had to fill in).
FORBIDDEN = ('REF = "protokol-kilidi"', "RUN_FINAL = False")

#: Strings that must appear in the notebook code.
REQUIRED = ("--preflight",)


def _code_text(cell: dict) -> str:
    """Join a cell's ``source`` (a list of lines or a single string) into one text."""
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _strip_magics(text: str) -> str:
    """Drop Colab magic lines (``!``/``%``) so the rest parses as plain Python."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("!", "%")))


def check(path: Path) -> list[str]:
    """Return the list of problems found in the notebook at ``path`` (empty when it is fine)."""
    problems: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"defter okunamadı: {exc}"]

    if notebook.get("nbformat") != 4:
        problems.append(f"nbformat 4 olmalı, gelen: {notebook.get('nbformat')}")

    code_texts: list[str] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") != []:
            problems.append(f"hücre {index}: outputs boş değil")
        if cell.get("execution_count") is not None:
            problems.append(f"hücre {index}: execution_count None değil")
        text = _code_text(cell)
        code_texts.append(text)
        try:
            ast.parse(_strip_magics(text))
        except SyntaxError as exc:
            problems.append(f"hücre {index}: Python olarak ayrıştırılamadı: {exc}")

    code = "\n".join(code_texts)
    for needle in FORBIDDEN:
        if needle in code:
            problems.append(f"elle doldurulan ayar kaldı: {needle}")
    for needle in REQUIRED:
        if needle not in code:
            problems.append(f"eksik: {needle}")
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
