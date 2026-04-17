"""Generate off-resonance field_map from the phantom."""

from snake.core import SimConfig

import numpy as np
from numpy.typing import NDArray


def get_field_map(mask: NDArray, prop: NDArray, sim_conf: SimConfig) -> np.ndarray:
    """Generate off-resonance field map from the phantom."""
    raise NotImplementedError(
        "Off-resonance field map generation is not implemented yet."
    )
