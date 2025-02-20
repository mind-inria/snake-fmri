"""SImulation base objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
from snake._meta import ThreeInts, ThreeFloats
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R


def _repr_html_(obj: Any, vertical: bool = True) -> str:
    """
    Recursive HTML representation for dataclasses.

    This function generates an HTML table representation of a dataclass,
    including nested dataclasses.

    Parameters
    ----------
    obj: The dataclass instance.

    Returns
    -------
        str: An HTML table string representing the dataclass.
    """
    class_name = obj.__class__.__name__
    table_rows = [
        '<table style="border:1px solid lightgray;">'
        '<caption style="border:1px solid lightgray;">'
        f"<strong>{class_name}</strong></caption>"
    ]
    from dataclasses import fields

    resolved_hints = obj.__annotations__
    field_names = [f.name for f in fields(obj)]
    field_values = {name: getattr(obj, name) for name in field_names}
    resolved_field_types = {name: resolved_hints[name] for name in field_names}

    if vertical:  # switch between vertical and horizontal mode
        for field_name in field_names:
            # Recursively call _repr_html_ for nested dataclasses
            field_value = field_values[field_name]
            field_type = resolved_field_types[field_name]
            try:
                field_value_str = field_value._repr_html_(vertical=not vertical)
            except AttributeError:
                field_value_str = repr(field_value)

            table_rows.append(
                f"<tr><td>{field_name}(<i>{field_type}</i>)</td>"
                f"<td>{field_value_str}</td></tr>"
            )
    else:
        table_rows.append(
            "<tr>"
            + "".join(
                [
                    f"<td>{field_name} (<i>{field_type}</i>)</td>"
                    for field_name, field_type in resolved_field_types.items()
                ]
            )
            + "</tr>"
        )
        values = []
        for field_value in field_values.values():
            # Recursively call _repr_html_ for nested dataclasses
            try:
                field_value_str = field_value._repr_html_(
                    vertical=not vertical
                )  # alternates orientation
            except AttributeError:
                field_value_str = repr(field_value)
            values.append(f"<td>{field_value_str}</td>")
        table_rows.append("<tr>" + "".join(values) + "</tr>")
    table_rows.append("</table>")
    return "\n".join(table_rows)


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

    _repr_html_ = _repr_html_

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
    raster_time_ms: float = 5e-3
    field: float = 3.0

    _repr_html_ = _repr_html_


default_hardware = HardwareConfig()

default_gre = GreConfig(TR=50, TE=30, FA=15)


@dataclass
class FOVConfig:
    """Field of View configuration.

    This class is used to define the FOV of the simulation.
    It uses the RAS convention and mm units.
    """

    size: ThreeFloats = (192, 192, 128)
    """Size of the FOV in millimeter."""
    offset: ThreeFloats = (0, 0, 0)
    """distance (in mm) of the bottom left left voxel to magnet isocenter."""
    angles: ThreeFloats = (0, 0, 0)
    """Euler Rotation Angles of the FOV in degrees"""
    res_mm: ThreeFloats = (1, 1, 1)
    """Resolution of the FOV in mm."""
    _repr_html_ = _repr_html_

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
    def shape(self):
        """Shape of the associated array in voxels units."""
        return tuple(round(s / r) for s, r in zip(self.size, self.res_mm))


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

    _repr_html_ = _repr_html_

    def __post_init__(self) -> None:
        # To be compatible with frozen dataclass
        self.rng: np.random.Generator = np.random.default_rng(self.rng_seed)

    @property
    def max_n_shots(self) -> int:
        """Maximum number of frames."""
        return int(self.max_sim_time * 1000 / self.sim_tr_ms)

    @property
    def res_mm(self) -> tuple[float, ...]:
        """Voxel resolution in mm."""
        return self.fov.res_mm

    @property
    def sim_tr_ms(self) -> float:
        """Simulation resolution in ms."""
        return self.seq.TR
