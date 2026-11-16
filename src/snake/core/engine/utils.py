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
) -> [NDArray, NDArray]:
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


def apply_field_drift(
    ksp: NDArray,
    shot_index: NDArray,
    n_shots: int,
    sample_times_ms: NDArray,
    te_ms: float,
    drift_hz: tuple[float, float],
) -> NDArray:
    """Add a spatially uniform off-resonance that drifts linearly over the shots.

    Shot ``s`` (in acquisition order, out of ``n_shots``) sees a uniform field offset
    ``df_s = drift_hz[0] + (drift_hz[1] - drift_hz[0]) * s / (n_shots - 1)`` (Hz), e.g. a
    B0 drift during the acquisition of a volume. A uniform field factors out of the
    signal integral, so each sample is multiplied by
    ``exp(2i pi df_s (te + t))``, with ``t`` the time of the sample from the echo: the
    phase accrues from the excitation, so the drift also changes the phase at the echo
    from shot to shot. The sign follows mri-nufft's off-resonance convention.

    Parameters
    ----------
    ksp: NDArray
        k-space of the shots, shape (n_chunk_shots, n_coils, n_samples).
    shot_index: NDArray
        Acquisition index of each shot of ``ksp`` (0 .. n_shots - 1).
    n_shots: int
        Total number of shots over which the drift goes from start to end.
    sample_times_ms: NDArray
        Time of each sample of a shot relative to the echo (ms), shape (n_samples,).
    te_ms: float
        Echo time (ms).
    drift_hz: tuple[float, float]
        Field offset at the first and at the last shot (Hz).

    Returns
    -------
    NDArray
        The k-space with the drift applied (same shape and dtype).
    """
    shot_index = np.asarray(shot_index, dtype=np.float64)
    frac = shot_index / max(n_shots - 1, 1)
    df = drift_hz[0] + (drift_hz[1] - drift_hz[0]) * frac  # (n_chunk_shots,)
    t_s = 1e-3 * (te_ms + np.asarray(sample_times_ms, dtype=np.float64))  # (n_samples,)
    phase = np.exp(2j * np.pi * df[:, None] * t_s[None, :])  # (n_chunk_shots, n_samples)
    return (ksp * phase[:, None, :]).astype(ksp.dtype)
