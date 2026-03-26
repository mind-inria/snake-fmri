"""Utilities for the MRD format."""

from copy import deepcopy
import scipy as sp
import numpy as np
from numpy.typing import NDArray

from ..phantom import Phantom, DynamicData
from ..simulation import SimConfig


def get_phantom_state(
    phantom: Phantom,
    dyn_datas: list[DynamicData],
    i: int,
    sim_conf: SimConfig,
    aggregate: bool = True,
) -> tuple[NDArray, NDArray | None]:
    """Get phantom state after applying all temporal variation."""
    frame_phantom = deepcopy(phantom)
    for dyn_data in dyn_datas:
        frame_phantom = dyn_data.func(frame_phantom, dyn_data.data, i)

    frame_phantom = frame_phantom.resample(
        new_affine=sim_conf.fov.affine,
        new_shape=sim_conf.shape,
        use_gpu=True,
    )
    return (
        frame_phantom.contrast(
            sim_conf=sim_conf,
            resample=False,
            aggregate=aggregate,
        ),
        frame_phantom.smaps,
    )


def fft(image: NDArray, axis: tuple[int, ...] | int = -1) -> NDArray:
    """Apply the FFT operator.

    Parameters
    ----------
    image : array
        Image in space.
    axis : int
        Axis to apply the FFT.

    Returns
    -------
    kspace_data : array
        kspace data.
    """
    return sp.fft.ifftshift(
        sp.fft.fftn(sp.fft.fftshift(image, axes=axis), norm="ortho", axes=axis),
        axes=axis,
    )


def get_noise(chunk_data: NDArray, cov: NDArray, rng: np.random.Generator) -> NDArray:
    """Generate noise for a given chunk of k-space data."""
    n_coils = cov.shape[0]

    chunk_size, n_coils, *xyz = chunk_data.shape

    noise_shape = (2, *xyz[::-1], chunk_size)
    noise = np.ascontiguousarray(
        rng.multivariate_normal(np.zeros(n_coils), cov, size=noise_shape).T,
        dtype=np.float32,
    )
    noise = noise.view(np.complex64)
    noise = noise[..., 0]
    noise = np.moveaxis(noise, 1, 0)
    return noise
