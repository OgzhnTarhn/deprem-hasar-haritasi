"""Tests for the deterministic training sampler (``hasar.data.sampler``).

The sampler is the single source of the training order: it must be reproducible from
``(seed, sample_no)`` alone so that a run can be resumed mid-epoch, and it must have a
finite length when ``total_steps`` is given (the ``max_iters`` trap: steps, not samples).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from hasar.data.sampler import StepSampler, rng_for

# --- rng_for ----------------------------------------------------------------


def test_rng_for_is_deterministic() -> None:
    first = rng_for(42, 7).random(8)
    second = rng_for(42, 7).random(8)
    assert np.array_equal(first, second)


def test_rng_for_returns_generator() -> None:
    assert isinstance(rng_for(42, 0), np.random.Generator)


def test_rng_for_default_stream_is_zero() -> None:
    assert np.array_equal(rng_for(42, 7).random(8), rng_for(42, 7, stream=0).random(8))


@pytest.mark.parametrize(
    "other",
    [
        pytest.param((43, 7, 0), id="seed"),
        pytest.param((42, 8, 0), id="sample_no"),
        pytest.param((42, 7, 1), id="stream"),
    ],
)
def test_rng_for_changes_with_each_argument(other: tuple[int, int, int]) -> None:
    base = rng_for(42, 7).random(8)
    changed = rng_for(*other).random(8)
    assert not np.array_equal(base, changed)


# --- StepSampler: epoch permutations ----------------------------------------


def test_first_sample_no_starts_at_start_step_times_batch() -> None:
    sampler = StepSampler(50, 4, seed=42, total_steps=10, start_step=3)
    index, sample_no = next(iter(sampler))
    assert sample_no == 3 * 4
    assert index == sampler.index_at(3 * 4)


def test_each_epoch_is_a_permutation() -> None:
    n_items = 50
    sampler = StepSampler(n_items, 5, seed=42, total_steps=20)
    items = list(itertools.islice(iter(sampler), 2 * n_items))

    first_epoch = [index for index, _ in items[:n_items]]
    second_epoch = [index for index, _ in items[n_items:]]

    assert sorted(first_epoch) == list(range(n_items))
    assert sorted(second_epoch) == list(range(n_items))


def test_epochs_differ() -> None:
    n_items = 50
    sampler = StepSampler(n_items, 5, seed=42, total_steps=20)
    items = list(itertools.islice(iter(sampler), 2 * n_items))

    first_epoch = [index for index, _ in items[:n_items]]
    second_epoch = [index for index, _ in items[n_items:]]

    assert first_epoch != second_epoch


def test_epoch_matches_explicit_permutation() -> None:
    n_items = 50
    sampler = StepSampler(n_items, 5, seed=42, total_steps=20)
    items = list(itertools.islice(iter(sampler), n_items))

    expected = np.random.default_rng([42, 0]).permutation(n_items)
    assert [index for index, _ in items] == list(expected)


def test_sample_no_is_contiguous() -> None:
    sampler = StepSampler(50, 4, seed=42, total_steps=10)
    sample_nos = [sample_no for _, sample_no in sampler]
    assert sample_nos == list(range(0, 10 * 4))


# --- StepSampler: determinism -----------------------------------------------


def test_same_seed_same_sequence() -> None:
    first = list(StepSampler(50, 4, seed=42, total_steps=10))
    second = list(StepSampler(50, 4, seed=42, total_steps=10))
    assert first == second


def test_different_seed_different_sequence() -> None:
    first = list(StepSampler(50, 4, seed=42, total_steps=10))
    second = list(StepSampler(50, 4, seed=43, total_steps=10))
    assert first != second


# --- index_at ---------------------------------------------------------------


def test_index_at_agrees_with_iteration() -> None:
    sampler = StepSampler(50, 4, seed=42, total_steps=10)
    for index, sample_no in sampler:
        assert sampler.index_at(sample_no) == index


def test_index_at_is_repeatable() -> None:
    sampler = StepSampler(50, 4, seed=42, total_steps=10)
    assert sampler.index_at(17) == sampler.index_at(17)


# --- finite length (gate 3, the max_iters trap) -----------------------------


def test_len_with_total_steps() -> None:
    sampler = StepSampler(50, 4, seed=42, total_steps=10)
    assert len(sampler) == 10 * 4


def test_len_with_start_step() -> None:
    sampler = StepSampler(50, 4, seed=42, total_steps=10, start_step=4)
    assert len(sampler) == (10 - 4) * 4


def test_iteration_yields_exactly_len_items() -> None:
    sampler = StepSampler(50, 4, seed=42, total_steps=10, start_step=4)
    assert len(list(sampler)) == len(sampler)


def test_len_without_total_steps_raises_type_error() -> None:
    sampler = StepSampler(50, 4, seed=42)
    with pytest.raises(TypeError):
        len(sampler)


def test_iteration_without_total_steps_is_infinite() -> None:
    sampler = StepSampler(50, 4, seed=42)
    items = list(itertools.islice(iter(sampler), 1000))
    assert len(items) == 1000
    assert [sample_no for _, sample_no in items] == list(range(1000))


# --- resume -----------------------------------------------------------------


def test_resume_matches_tail_of_full_run() -> None:
    n_items, batch, seed = 50, 4, 42
    full = list(StepSampler(n_items, batch, seed, total_steps=10))
    resumed = list(StepSampler(n_items, batch, seed, total_steps=10, start_step=4))
    assert resumed == full[4 * batch :]


def test_resume_at_epoch_boundary() -> None:
    n_items, batch, seed = 50, 5, 42
    full = list(StepSampler(n_items, batch, seed, total_steps=20))
    resumed = list(StepSampler(n_items, batch, seed, total_steps=20, start_step=10))
    assert resumed == full[10 * batch :]


# --- invalid arguments ------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"n_items": 0, "batch": 4}, id="n_items_zero"),
        pytest.param({"n_items": -1, "batch": 4}, id="n_items_negative"),
        pytest.param({"n_items": 50, "batch": 0}, id="batch_zero"),
        pytest.param({"n_items": 50, "batch": -4}, id="batch_negative"),
        pytest.param({"n_items": 50, "batch": 4, "start_step": -1}, id="start_step_negative"),
        pytest.param(
            {"n_items": 50, "batch": 4, "total_steps": 10, "start_step": 11},
            id="start_step_past_total",
        ),
    ],
)
def test_invalid_arguments_raise_value_error(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        StepSampler(seed=42, **kwargs)
