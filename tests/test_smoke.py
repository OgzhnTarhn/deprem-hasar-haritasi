"""Smoke test: the package and its subpackages import cleanly."""

import hasar
import hasar.analysis
import hasar.data
import hasar.eval
import hasar.models
import hasar.preflight
import hasar.train


def test_package_imports() -> None:
    assert hasar.__version__ == "0.1.0"
