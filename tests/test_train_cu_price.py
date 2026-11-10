"""Gate 8 CU price per hour follows the GPU the notebook actually got (A100 or L4)."""

import pytest

run = pytest.importorskip("hasar.train.run")


@pytest.mark.parametrize(
    ("name", "price"),
    [
        ("NVIDIA A100-SXM4-40GB", 13.0),
        ("NVIDIA L4", 5.0),
        ("Tesla T4", 13.0),
        ("", 13.0),
    ],
)
def test_cu_per_hour_by_gpu_name(name: str, price: float) -> None:
    assert run.cu_per_hour(name) == price
