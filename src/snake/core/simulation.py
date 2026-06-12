"""SImulation base objects."""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
import numpy as np
from snake._meta import ThreeInts, ThreeFloats
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R
from .._meta import dataclass_repr_html

log = logging.getLogger(__name__)


@dataclass
class GreConfig:
    """Gradient Recall Echo Sequence parameters."""

    """TR: Repetition Time in ms.
    This is the time between two consecutive RF pulses."""
    TR: float
    """TE: Echo Time in ms.
    This is the time between the RF pulse and the echo."""
    TE: float
    """FA: Flip Angle in degrees.
    This is the angle of the RF pulse to the magnetization."""
    FA: float

    _repr_html_ = dataclass_repr_html

    def __post_init__(self) -> None:
        """Validate the parameters. And create a Effective TR."""
        if self.TE >= self.TR:
            raise ValueError("TE must be less than TR.")
        if self.FA < 0 or self.FA > 180:
            raise ValueError("FA must be between 0 and 180 degrees.")
        if self.TR < 0 or self.TE < 0:
            raise ValueError("TR and TE must be positive.")

        self.TR_eff = (
            self.TR
        )  # To be updated if needed. this will be used for the contrast calculation


@dataclass
class HardwareConfig:
    """Scanner Hardware parameters."""

    gmax: float = 40
    smax: float = 200
    n_coils: int = 8
    dwell_time_ms: float = 1e-3
    raster_time_ms: float = 1e-2
    field: float = 3.0

    _repr_html_ = dataclass_repr_html


default_hardware = HardwareConfig()

default_gre = GreConfig(TR=50, TE=30, FA=15)


@dataclass
class FOVConfig:
    """Field of View configuration.

    This class is used to define the FOV of the simulation.
    It uses the RAS convention and mm units.

    Default values are from the BrainWeb dataset.
    """

    size: ThreeFloats = (181, 217, 181)
    """Size of the FOV in millimeter."""
    offset: ThreeFloats = (-90.25, -126.25, -72.25)
    """distance (in mm) of the bottom left left voxel to magnet isocenter."""
    angles: ThreeFloats = (0, 0, 0)
    """Euler Rotation Angles of the FOV in degrees"""
    res_mm: ThreeFloats = (1, 1, 1)
    """Resolution of the FOV in mm."""
    _repr_html_ = dataclass_repr_html

    def __post_init__(self) -> None:
        """Validate the parameters."""
        if any(r <= 0 for r in self.res_mm) or any(s <= 0 for s in self.size):
            raise ValueError("resolution and size must be positive.")
        if any(abs(a) > 180 for a in self.angles):
            raise ValueError("Angles must be between -180 and 180 degrees.")
        if any(r > s for r, s in zip(self.res_mm, self.size, strict=False)):
            log.warning(
                "Resolution is higher than the size of the FOV,"
                " setting to 1voxel thickness."
            )
            self.size = tuple(
                max(r, s) for r, s in zip(self.res_mm, self.size, strict=False)
            )

    @classmethod
    def from_affine(cls, affine: NDArray, size: ThreeFloats) -> FOVConfig:
        """Create a FOVConfig from an affine matrix."""
        res_mm = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
        offset = affine[:3, 3]
        angles = R.from_matrix(affine[:3, :3] / res_mm).as_euler("xyz", degrees=True)
        return cls(res_mm=res_mm, offset=offset, angles=angles, size=size)

    @property
    def affine(self) -> NDArray[np.float32]:
        """Generate an affine matrix from the FOV configuration."""
        affine = np.eye(4, dtype=np.float32)
        affine[:3, :3] = np.diag(self.res_mm)
        affine[:3, 3] = np.array(self.offset)
        rotation_matrix = R.from_euler("xyz", self.angles, degrees=True).as_matrix()
        affine[:3, :3] = affine[:3, :3] @ rotation_matrix
        return affine

    @property
    def shape(self) -> ThreeInts:
        """Shape of the associated array in voxels units."""
        return tuple(round(s / r) for s, r in zip(self.size, self.res_mm, strict=False))


@dataclass
class SimConfig:
    """All base configuration of a simulation."""

    max_sim_time: float = 300
    seq: GreConfig = field(default_factory=lambda: GreConfig(TR=50, TE=30, FA=15))
    hardware: HardwareConfig = field(default_factory=lambda: HardwareConfig())
    fov: FOVConfig = field(default_factory=lambda: FOVConfig())

    # fov_mm: tuple[float, float, float] = (192.0, 192.0, 128.0)
    # shape: tuple[int, int, int] = (192, 192, 128)  # Target reconstruction shape
    rng_seed: int = 19290506

    _repr_html_ = dataclass_repr_html

    def __post_init__(self) -> None:
        # To be compatible with frozen dataclass
        self.rng: np.random.Generator = np.random.default_rng(self.rng_seed)

    @property
    def max_n_shots(self) -> int:
        """Maximum number of frames."""
        return int(self.max_sim_time * 1000 / self.sim_tr_ms)

    @property
    def res_mm(self) -> ThreeFloats:
        """Voxel resolution in mm."""
        return self.fov.res_mm

    @property
    def sim_tr_ms(self) -> float:
        """Simulation resolution in ms."""
        return self.seq.TR

    @property
    def shape(self) -> ThreeInts:
        """Shape of the simulation."""
        return self.fov.shape

    @property
    def fov_mm(self) -> ThreeFloats:
        """Size of the FOV in mm."""
        return self.fov.size
