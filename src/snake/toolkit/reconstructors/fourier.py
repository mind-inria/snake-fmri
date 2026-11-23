"""FFT operators for MRI reconstruction."""

from numpy.typing import NDArray
import scipy as sp
from snake.mrd_utils.loader import NonCartesianFrameDataLoader
from mrinufft.operators import FourierOperatorBase


def fft(image: NDArray, axis: int | tuple[int] = -1) -> NDArray:
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


def ifft(kspace_data: NDArray, axis: int | tuple[int] = -1) -> NDArray:
    """Apply the inverse FFT operator.

    Parameters
    ----------
    kspace_data : array
        Image in space.
    axis : int
        Axis to apply the FFT.

    Returns
    -------
    image_data : array
        image data.
    """
    return sp.fft.fftshift(
        sp.fft.ifftn(sp.fft.ifftshift(kspace_data, axes=axis), norm="ortho", axes=axis),
        axes=axis,
    )


def init_nufft(
    data_loader: NonCartesianFrameDataLoader,
    nufft_backend: str,
    density_compensation: bool = False,
) -> FourierOperatorBase:
    """Initialize the NUFFT operator from the data_loader."""
    from mrinufft import get_operator

    smaps = data_loader.get_smaps()
    shape = data_loader.shape
    traj, _ = data_loader.get_kspace_frame(0)


    kwargs = dict(
        shape=shape,
        n_coils=data_loader.n_coils,
        smaps=smaps[...,0].copy()  if data_loader.slice_2d else smaps,
    )
    if density_compensation is False:
        kwargs["density"] = None
    else:
        kwargs["density"] = density_compensation
    if "stacked" in nufft_backend:
        kwargs["z_index"] = "auto"

    return get_operator(
        nufft_backend,
        samples=traj,
        **kwargs,
    )
