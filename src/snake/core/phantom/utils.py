"""utilities for phantoms."""

import logging
import os
from enum import IntEnum
from importlib.resources import files

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom

from snake._meta import NoCaseEnum

log = logging.getLogger(__name__)


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


def parse_tissue_file(
    tissue_file: TissueFile | str,
) -> dict[str, tuple[float, float, float, float, float]]:
    """Parse the tissue properties file.

    And return a dictionary with the tissue name as key and the properties as value.
    properties are T1, T2, T2s, rho and chi.
    """
    tissues = dict()
    try:
        if isinstance(tissue_file, TissueFile):
            tissue_file = tissue_file.value
        else:
            tissue_file = TissueFile[tissue_file].value
    except ValueError as exc:
        if not os.path.exists(tissue_file):
            raise FileNotFoundError(f"File {tissue_file} does not exist.") from exc
    finally:
        tissue_file = str(tissue_file)
    log.info(f"Using tissue file:{tissue_file} ")
    with open(tissue_file) as f:

        lines = f.readlines()
        select = []
        for line in lines[1:]:
            vals = line.split(",")
            t1, t2, t2s, rho, chi = map(np.float32, vals[1:])
            name = vals[0]
            tissues[name] = (t1, t2, t2s, rho, chi)
    return tissues
