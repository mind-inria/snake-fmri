"""utilities for phantoms."""

from numpy.typing import NDArray
from enum import IntEnum
from snake._meta import NoCaseEnum
from importlib.resources import files
from scipy.ndimage import zoom


def resize_tissues(
    input: NDArray, output: NDArray, i: int, z: tuple[float], order: int = 3
) -> None:
    """Resize the tissues."""
    output[i] = zoom(input[i], z, order=order)


class PropTissueEnum(IntEnum):
    """Enum for the tissue properties."""

    T1 = 0
    T2 = 1
    T2s = 2
    rho = 3
    chi = 4


class TissueFile(str, NoCaseEnum):
    """Enum for the tissue properties file."""

    tissue_1T5 = str(files("snake.core.phantom.data") / "tissues_properties_1T5.csv")
    tissue_7T = str(files("snake.core.phantom.data") / "tissues_properties_7T.csv")
    tissue_3T = str(files("snake.core.phantom.data") / "tissues_properties_3T.csv")
