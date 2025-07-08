"""Mathematical transformations of data."""

import base64
import logging
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from snake._meta import ThreeInts

from .parallel import run_parallel
from ._utils import DuplicateFilter

log = logging.getLogger(__name__)

dup_filter = DuplicateFilter(log)


def effective_affine(old: NDArray, new: NDArray) -> NDArray:
    """Compute the effective affine transformation between two affine matrices."""
    new_affine = np.asarray(new, dtype=np.float32)
    old_affine = np.asarray(old, dtype=np.float32)
    return np.linalg.inv(old_affine) @ new_affine


def _validate_gpu_affine(use_gpu: bool = True) -> tuple[bool, Callable, ModuleType]:
    """Check if we can use the affine_transform from cupy."""
    if use_gpu:
        try:
            import cupy as xp
            from cupyx.scipy.ndimage import affine_transform as cu_affine_transform

            try:
                use_gpu = xp.cuda.is_available()
            except Exception as exc:
                use_gpu = False
                raise ImportError from exc

            def affine_transform(
                x: NDArray,
                *args: Any,
                output_shape: ThreeInts,
                output: NDArray[np.float32] = None,
                **kwargs: Any,
            ) -> NDArray:
                output_gpu = xp.zeros(output_shape, dtype=x.dtype)
                cu_affine_transform(
                    x,
                    *args,
                    output_shape=output_shape,
                    output=output_gpu,
                    **kwargs,
                    texture_memory=x.dtype == xp.float32,
                )
                if output is not None:
                    xp.copyto(output, output_gpu)
                    return output
                else:
                    return output_gpu.get()
        except ImportError:
            use_gpu = False
    if not use_gpu:
        import numpy as xp
        with dup_filter:
            log.warning("Cupy not available, using CPU.")
        from scipy.ndimage import affine_transform
    return use_gpu, affine_transform, xp


def apply_affine(
    data: NDArray[np.float32],
    old_affine: NDArray[np.float32],
    new_affine: NDArray[np.float32],
    new_shape: ThreeInts,
    output: NDArray[np.float32] = None,
    transform_affine: NDArray[np.float32] = None,
    use_gpu: bool = True,
    mode="nearest",
    **kwargs: Any,
) -> NDArray[np.float32]:
    """Apply the new affine on the data.

    Parameters
    ----------
    data : NDArray
        Data to be transformed. 3D Array.
    old_affine : NDArray
        Affine of the original data
    new_affine : NDArray
        Affine of the new data
    new_shape : ThreeInts
        Shape of the new data
    transform_affine : NDArray, optional
        Transformation affine, by default None
    use_gpu : bool, optional
        Try to use GPU, by default True
    output: NDArray, optional
        Output array, by default None

    Returns
    -------
    NDArray
    Resampled data with ``new_affine`` orientation and ``new_shape`` shape.
    """
    use_gpu, affine_transform, xp = _validate_gpu_affine(use_gpu)
    if transform_affine is None:
        transform_affine = effective_affine(old=old_affine, new=new_affine)

    if np.allclose(transform_affine, np.eye(4)) and data.shape == new_shape:
        if output is not None:
            np.copyto(output, data)
            return output
        else:
            return data

    data_ = xp.asarray(data)
    transform_affine = xp.asarray(transform_affine, dtype=xp.float32)
    return affine_transform(
        data_,
        matrix=transform_affine,
        output_shape=new_shape,
        output=output,
        mode=mode,
        **kwargs,
    )


def __apply_affine(
    x: NDArray, output: NDArray, i: int, *args: Any, **kwargs: Any
) -> NDArray:
    return apply_affine(x[i], *args, output=output[i], **kwargs)


def apply_affine4d(
    data: NDArray,
    old_affine: NDArray,
    new_affine: NDArray,
    new_shape: ThreeInts,
    use_gpu: bool = False,
    n_jobs: int = -1,
    axis: int = 0,
    **kwargs: Any,
) -> NDArray:
    """
    Apply the new affine on 4D data.

    Parameters
    ----------
    data : NDArray
        Data to be transformed. 3D Array.
    old_affine : NDArray
        Affine of the original data
    new_affine : NDArray
        Affine of the new data
    new_shape : ThreeInts
        Shape of the new data
    transform_affine : NDArray, optional
        Transformation affine, by default None
    use_gpu : bool, optional

    Returns
    -------
    NDArray
        Resampled data with ``new_affine`` orientation and ``new_shape`` shape.

    See Also
    --------
    apply_affine
    """
    n_frames = data.shape[axis]

    new_array = np.zeros((n_frames, *new_shape), dtype=data.dtype)

    transform_affine = effective_affine(old_affine, new_affine)
    use_gpu = _validate_gpu_affine(use_gpu)[0]

    if not use_gpu:
        run_parallel(
            __apply_affine,
            data,
            new_array,
            old_affine=old_affine,
            new_affine=new_affine,
            use_gpu=False,
            transform_affine=transform_affine,
            new_shape=new_shape,
            n_jobs=n_jobs,
            parallel_axis=axis,
            **kwargs,
        )
    else:
        for i in range(n_frames):
            new_array[i] = apply_affine(
                data[i],
                old_affine=old_affine,
                new_affine=new_affine,
                use_gpu=use_gpu,
                transform_affine=transform_affine,
                new_shape=new_shape,
            )

    return new_array


def serialize_array(arr: NDArray) -> str:
    """Serialize the array for mrd compatible format."""
    return "__".join(
        [
            base64.b64encode(arr.tobytes()).decode(),
            str(arr.shape),
            str(arr.dtype),
        ]
    )


def unserialize_array(s: str) -> NDArray:
    """Unserialize the array for mrd compatible format."""
    data, shape, dtype = s.split("__")
    shape = eval(shape)  # FIXME
    return np.frombuffer(base64.b64decode(data.encode()), dtype=dtype).reshape(*shape)
