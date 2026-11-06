"""Deterministic training sampler.

The sampler is a pure function of ``(seed, sample counter)``: the item at sample number ``n`` is
``permutation(seed, n // n_items)[n % n_items]``, where each epoch permutation is drawn from
``np.random.default_rng([seed, epoch])``. Nothing depends on the iteration history, so a run can be
resumed mid-epoch by starting the counter at ``start_step * batch`` and the sequence is exactly the
tail of the uninterrupted run.

Because the counter is a *sample* counter, the loader length is ``steps * batch`` when
``total_steps`` is given. This is deliberately not BRIGHT's ``max_iters``, which counts samples
(800.000 / 16 = 50.000 steps) while ChangeMamba's ``max_iters`` counts steps.

The class imports only numpy; it still works as a torch ``sampler=`` because it defines
``__iter__`` and ``__len__``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def rng_for(seed: int, sample_no: int, stream: int = 0) -> np.random.Generator:
    """Return the generator for one draw, keyed by ``(seed, sample_no, stream)``.

    ``stream`` separates independent random decisions that share the same sample counter (for
    example augmentation versus sampling) without changing the sampler's own sequence.
    """
    return np.random.default_rng([seed, sample_no, stream])


class StepSampler:
    """Infinite or step-bounded sampler over ``n_items`` indices, reproducible from ``(seed, n)``.

    Parameters
    ----------
    n_items:
        Number of items in the dataset; each epoch is a permutation of ``range(n_items)``.
    batch:
        Batch size. The sample counter advances by one per item, so a step consumes ``batch``
        consecutive sample numbers.
    seed:
        Run seed; part of every generator key.
    total_steps:
        Number of optimizer steps. When given, iteration stops after ``total_steps * batch``
        samples and ``len()`` is finite. When ``None``, iteration is infinite and ``len()`` raises
        ``TypeError``.
    start_step:
        First step to yield, used to resume a run. The first sample number is
        ``start_step * batch``.
    """

    def __init__(
        self,
        n_items: int,
        batch: int,
        seed: int,
        total_steps: int | None = None,
        start_step: int = 0,
    ) -> None:
        if n_items <= 0:
            raise ValueError(f"n_items must be positive, got {n_items}")
        if batch <= 0:
            raise ValueError(f"batch must be positive, got {batch}")
        if start_step < 0:
            raise ValueError(f"start_step must be non-negative, got {start_step}")
        if total_steps is not None and start_step > total_steps:
            raise ValueError(f"start_step ({start_step}) must not exceed total_steps ({total_steps})")

        self.n_items = n_items
        self.batch = batch
        self.seed = seed
        self.total_steps = total_steps
        self.start_step = start_step

        # Cache of the current epoch permutation, so iteration is O(1) per item.
        self._epoch: int | None = None
        self._permutation: np.ndarray | None = None

    def _permutation_for(self, epoch: int) -> np.ndarray:
        """Return the permutation of ``epoch``, reusing the cached one when possible."""
        if self._epoch != epoch:
            self._permutation = np.random.default_rng([self.seed, epoch]).permutation(self.n_items)
            self._epoch = epoch
        assert self._permutation is not None
        return self._permutation

    def index_at(self, sample_no: int) -> int:
        """Return the dataset index for sample number ``sample_no``."""
        epoch, offset = divmod(sample_no, self.n_items)
        return int(self._permutation_for(epoch)[offset])

    def __iter__(self) -> Iterator[tuple[int, int]]:
        """Yield ``(index, sample_no)`` pairs starting at ``start_step * batch``."""
        sample_no = self.start_step * self.batch
        stop = None if self.total_steps is None else self.total_steps * self.batch
        while stop is None or sample_no < stop:
            yield self.index_at(sample_no), sample_no
            sample_no += 1

    def __len__(self) -> int:
        """Return the number of samples to yield; requires ``total_steps``."""
        if self.total_steps is None:
            raise TypeError("len() is undefined without total_steps; the sampler is infinite")
        return (self.total_steps - self.start_step) * self.batch
