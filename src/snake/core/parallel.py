"""Utilities for running parallel computations with processes and shared memory."""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Any, NamedTuple

import numpy as np
from joblib import Parallel, delayed
from numpy._typing import _ShapeLike
from numpy.typing import DTypeLike, NDArray

log = logging.getLogger(__name__)


class SharedArray(NamedTuple):
    """Properties of an array stored in shared memory."""

    name: str
    shape: _ShapeLike
    dtype: DTypeLike

    @property
    def nbytes(self) -> int:
        """Number of bytes needed to store the array."""
        return int(np.prod(self.shape) * np.dtype(self.dtype).itemsize)

    @contextmanager
    def as_array(self) -> Generator[NDArray, None, None]:
        """Get the array from shared memory."""
        shm = SharedMemory(name=self.name, size=self.nbytes, create=False)
        yield np.ndarray(shape=self.shape, dtype=self.dtype, buffer=shm.buf)
        shm.close()

    def to_array(self) -> NDArray:
        """Copy the array from shared memory, and close the shared memory."""
        with self.as_array() as arr:
            return arr.copy()

    @classmethod
    def from_array(cls, manager: SharedMemoryManager, array: NDArray) -> SharedArray:
        """Copy an array to shared memory."""
        shm = manager.SharedMemory(size=array.nbytes)
        _arr_view = np.ndarray(shape=array.shape, dtype=array.dtype, buffer=shm.buf)
        _arr_view[:] = array  # copy to shared memory
        return SharedArray(shm.name, shape=array.shape, dtype=array.dtype)


class SHM_Wrapper:
    """Wrapper for function to be call with parallel shared memory.

    Parameters
    ----------
    func : Callable
        Function to be called with shared memory arrays.
    """

    # A decorator would not work here because of the way joblib works.
    def __init__(self, func: Callable):
        self.func = func

    def __call__(
        self,
        shared_input: SharedArray,
        shared_output: SharedArray,
        i: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Run in parallel with shared memory."""
        with shared_input.as_array() as input, shared_output.as_array() as output:
            self.func(input, output, i, *args, **kwargs)


def run_parallel(
    func: Callable,
    input_array: NDArray,
    output_array: NDArray,
    # n_jobs: int = -1,
    n_jobs: int = 1,
    parallel_axis: int = 0,
    *args: Any,
    **kwargs: Any,
) -> NDArray:
    """Run a function in parallel with shared memory."""
    with (
        SharedMemoryManager() as smm,
        Parallel(n_jobs=n_jobs, backend="multiprocessing") as parallel,
    ):
        share_input = SharedArray.from_array(smm, input_array)
        share_output = SharedArray.from_array(smm, output_array)
        parallel(
            delayed(SHM_Wrapper(func))(
                share_input,
                share_output,
                i,
                *args,
                **kwargs,
            )
            for i in range(input_array.shape[parallel_axis])
        )
        output_array = share_output.to_array()  # copy back
        smm.shutdown()

    return output_array
