"""Utilities for running parallel computations with processes and shared memory."""

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Any, NamedTuple

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import DTypeLike, NDArray

log = logging.getLogger(__name__)


class ArrayProps(NamedTuple):
    """Properties of an array stored in shared memory."""

    name: str
    shape: tuple[int, ...]
    dtype: DTypeLike


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
        input_props: ArrayProps,
        output_props: ArrayProps,
        i: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Run in parallel with shared memory."""
        with array_from_shm(input_props, output_props) as (input, output):
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
        # input_shm = smm.SharedMemory(size=input_array.nbytes)
        # input_array_sm = np.ndarray(
        #     input_array.shape, dtype=input_array.dtype, buffer=input_shm.buf
        # )
        # input_array_sm[:] = input_array  # move to shared memory
        # output_shm = smm.SharedMemory(size=output_array.nbytes)
        # output_array_sm = np.ndarray(
        #     output_array.shape, dtype=output_array.dtype, buffer=output_shm.buf
        # )
        # input_prop = ArrayProps(input_shm.name, input_array.shape, input_array.dtype)
        # output_prop = ArrayProps(
        #     output_shm.name, output_array.shape, output_array.dtype
        # )
        input_prop, input_array_sm, input_shm = array_to_shm(input_array, smm)
        output_prop, output_array_sm, output_shm = array_to_shm(output_array, smm)
        input_array_sm[:] = input_array  # move to shared memory
        parallel(
            delayed(SHM_Wrapper(func))(
                input_prop,
                output_prop,
                i,
                *args,
                **kwargs,
            )
            for i in range(input_array.shape[parallel_axis])
        )
        output_array[:] = output_array_sm  # copy back
        smm.shutdown()

    return output_array


@contextmanager
def array_from_shm(
    *array_props: ArrayProps | None,
) -> Generator[list[NDArray | None], None, None]:
    """Get arrays from shared memory."""
    shms = []
    arrays = []
    for prop in array_props:
        if prop is None:  # optional arrays are ignored
            arrays.append(None)
            continue
        nbytes = int(np.dtype(prop.dtype).itemsize * np.prod(prop.shape))
        shms.append(SharedMemory(name=prop.name, size=nbytes))
        arrays.append(
            np.ndarray(shape=prop.shape, dtype=prop.dtype, buffer=shms[-1].buf)
        )
    yield arrays
    del arrays
    for s in shms:
        s.close()
    del shms


def array_to_shm(
    array: NDArray, smm: SharedMemoryManager
) -> tuple[ArrayProps, NDArray, SharedMemory]:
    """Move an array to shared memory."""
    shm = smm.SharedMemory(size=array.nbytes)
    array_sm: NDArray = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    array_sm[:] = array  # move to shared memory
    # Returning the shm object is required to avoid garbage collection (and segfault)
    return ArrayProps(shm.name, array.shape, str(array.dtype)), array_sm, shm
