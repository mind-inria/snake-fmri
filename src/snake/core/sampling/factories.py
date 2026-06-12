"""K-space trajectory factories."""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator, Mapping, Sequence
from typing import Any
import numpy as np
from numpy.typing import NDArray
from mrinufft.trajectories.maths import R2D
from mrinufft.trajectories.tools import rotate, stack
from mrinufft.trajectories.trajectory2D import (
    initialize_2D_radial,
    initialize_2D_spiral,
)
from mrinufft.trajectories.utils import (
    check_hardware_constraints,
    compute_gradients_and_slew_rates,
)
from scipy.stats import norm  # type: ignore

from ..._meta import NoCaseEnum

logger = logging.getLogger(__name__)

SlicerType = list[slice | np.ndarray[Any, np.dtype[np.int64]] | int]


def validate_rng(rng: int | None | np.random.Generator = None) -> np.random.Generator:
    """Validate Random Number Generator."""
    if isinstance(rng, int | list):
        return np.random.default_rng(rng)
    elif rng is None:
        return np.random.default_rng()
    elif isinstance(rng, np.random.Generator):
        return rng
    else:
        raise ValueError("rng should be a numpy Generator, None or an integer seed.")


class VDSorder(NoCaseEnum):
    """Available ordering for variable density sampling."""

    CENTER_OUT = "center-out"
    RANDOM = "random"
    TOP_DOWN = "top-down"


class VDSpdf(NoCaseEnum):
    """Available law for variable density sampling."""

    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    EQUISPACED = "equispaced"


def get_kspace_slice_loc(
    dim_size: int,
    center_prop: int | float,
    accel: int = 4,
    pdf: VDSpdf = VDSpdf.GAUSSIAN,
    rng: int | None | np.random.Generator = None,
    order: VDSorder = VDSorder.CENTER_OUT,
) -> np.ndarray:
    """Get slice index at a random position.

    Parameters
    ----------
    dim_size: int
        Dimension size
    center_prop: float or int
        Proportion of center of kspace to continuouly sample
    accel: float
        Undersampling/Acceleration factor
    pdf: str, optional
        Probability density function for the remaining samples.
        "gaussian" (default) or "uniform".
    rng: random state

    Returns
    -------
    np.ndarray: array of size dim_size/accel.
    """
    order = VDSorder(order)
    pdf = VDSpdf(pdf)
    if accel == 0:
        return np.arange(dim_size)  # type: ignore

    indexes = list(range(dim_size))

    if not isinstance(center_prop, int):
        center_prop = int(center_prop * dim_size)

    center_start = (dim_size - center_prop) // 2
    center_stop = (dim_size + center_prop) // 2
    center_indexes = indexes[center_start:center_stop]
    borders = np.asarray([*indexes[:center_start], *indexes[center_stop:]])

    n_samples_borders = (dim_size - len(center_indexes)) // accel
    if n_samples_borders < 1:
        raise ValueError(
            "acceleration factor, center_prop and dimension not compatible."
            "Edges will not be sampled. "
        )
    rng = validate_rng(rng)

    def _get_samples(p: NDArray) -> list[int]:
        p /= np.sum(p)
        return list(rng.choice(borders, size=n_samples_borders, replace=False, p=p))

    if pdf is VDSpdf.GAUSSIAN:
        p = norm.pdf(np.linspace(norm.ppf(0.001), norm.ppf(0.999), len(borders)))
        sampled_in_border = _get_samples(p)
    elif pdf is VDSpdf.UNIFORM:
        p = np.ones(len(borders))
        sampled_in_border = _get_samples(p)
    elif pdf is VDSpdf.EQUISPACED:
        sampled_in_border = list(borders[::accel])
        pass
    else:
        raise ValueError("Unsupported value for pdf.")
        # TODO: allow custom pdf as argument (vector or function.)

    line_locs = np.array(sorted(center_indexes + sampled_in_border))
    # apply order of lines
    if order == VDSorder.CENTER_OUT:
        line_locs = flip2center(sorted(line_locs), dim_size // 2)
    elif order == VDSorder.RANDOM:
        line_locs = rng.permutation(line_locs)
    elif order == VDSorder.TOP_DOWN:
        line_locs = np.array(sorted(line_locs))
    else:
        raise ValueError(f"Unknown direction '{order}'.")
    return line_locs


def get_cartesian_mask(
    shape: tuple[int, ...],
    n_frames: int,
    rng: int | None | np.random.Generator = None,
    constant: bool = False,
    center_prop: float | int = 0.3,
    accel: int = 4,
    accel_axis: int = 0,
    pdf: VDSpdf = VDSpdf.GAUSSIAN,
) -> np.ndarray:
    """
    Get a cartesian mask for fMRI kspace data.

    Parameters
    ----------
    shape: tuple
        shape of fMRI volume.
    n_frames: int
        number of frames.
    rng: Generator or int or None (default)
        Random number generator or seed.
    constant: bool
        If True, the mask is constant across time.
    center_prop: float
        Proportion of center of kspace to continuouly sample
    accel: float
        Undersampling/Acceleration factor
    pdf: str, optional
        Probability density function for the remaining samples.
        "gaussian" (default) or "uniform".
    rng: random state

    Returns
    -------
    np.ndarray: random mask for an acquisition.
    """
    rng = validate_rng(rng)

    mask = np.zeros((n_frames, *shape))
    slicer: SlicerType = [slice(None, None, None)] * (1 + len(shape))
    if accel_axis < 0:
        accel_axis = len(shape) + accel_axis
    if not (0 < accel_axis < len(shape)):
        raise ValueError(
            "accel_axis should be lower than the number of spatial dimension."
        )
    if constant:
        mask_loc = get_kspace_slice_loc(shape[accel_axis], center_prop, accel, pdf, rng)
        slicer[accel_axis + 1] = mask_loc
        mask[tuple(slicer)] = 1
        return mask

    for i in range(n_frames):
        mask_loc = get_kspace_slice_loc(shape[accel_axis], center_prop, accel, pdf, rng)
        slicer[0] = i
        slicer[accel_axis + 1] = mask_loc
        mask[tuple(slicer)] = 1
    return mask


def flip2center(mask_cols: Sequence[int], center_value: int) -> np.ndarray:
    """
    Reorder a list by starting by a center_position and alternating left/right.

    Parameters
    ----------
    mask_cols: list or np.array
        List of columns to reorder.
    center_pos: int
        Position of the center column.

    Returns
    -------
    np.array: reordered columns.
    """
    center_pos = np.argmin(np.abs(np.array(mask_cols) - center_value))
    mask_cols = list(mask_cols)
    left = mask_cols[center_pos::-1]
    right = mask_cols[center_pos + 1 :]
    new_cols = []
    while left or right:
        if left:
            new_cols.append(left.pop(0))
        if right:
            new_cols.append(right.pop(0))
    return np.array(new_cols)


def check_trajectory(
    trajectory: NDArray, osf: int, gmax: float, smax: float
) -> np.bool_:
    """Check if a trajectory is feasible or not."""
    grads, slew = compute_gradients_and_slew_rates(trajectory[:, ::osf, :])
    is_ok, max_grad, max_slew = check_hardware_constraints(grads, slew, gmax, smax)
    return np.all(is_ok)


def vds_factory(
    shape: tuple[int, ...],
    acs: float | int,
    accel: int,
    accel_axis: int,
    order: VDSorder = VDSorder.CENTER_OUT,
    pdf: VDSpdf = VDSpdf.GAUSSIAN,
    rng: int | None | np.random.Generator = None,
) -> np.ndarray:
    """
    Create a variable density sampling trajectory.

    Parameters
    ----------
    shape
        Shape of the kspace.
    acs
        autocalibration line number (int) or proportion (float)
    direction
        Direction of the sampling.
    TR
        Time to acquire the k-space. Exclusive with base_TR.
    base_TR
        Time to acquire a full volume in the base trajectory. Exclusive with TR.
    pdf
        Probability density function of the sampling. "gaussian" or "uniform"
    rng
        Random number generator or seed.

    Returns
    -------
    KspaceTrajectory
        Variable density sampling trajectory.
    """
    if accel_axis < 0:
        accel_axis = len(shape) + accel_axis
    if not (0 <= accel_axis < len(shape)):
        raise ValueError(
            "accel_axis should be lower than the number of spatial dimension."
        )

    line_locs = get_kspace_slice_loc(shape[accel_axis], acs, accel, pdf, rng, order)
    # initialize the trajetory. -1 is the default value,
    # and we put the line index in the correct axis (0-indexed)
    shots = -np.ones((len(line_locs), 1, len(shape)), dtype=np.int32)
    for shot_idx, line_loc in enumerate(line_locs):
        shots[shot_idx, :, accel_axis] = line_loc
    return shots


def radial_factory(
    shape: tuple[int, ...],
    n_shots: int,
    n_points: int,
    expansion: str | None = None,
    n_repeat: int = 0,
    **kwargs: Mapping[str, Any],
) -> np.ndarray:
    """Create a radial sampling trajectory."""
    traj_points = initialize_2D_radial(n_shots, n_points)

    if len(shape) == 3:
        if expansion is None:
            raise ValueError("Expansion should be provided for 3D radial sampling.")
        if n_repeat is None:
            raise ValueError("n_repeat should be provided for 3D radial sampling.")
        if expansion == "stacked":
            traj_points = stack(
                traj_points,
                nb_stacks=n_repeat,
            )
        elif expansion == "rotated":
            traj_points = rotate(
                traj_points,
                nb_rotations=n_repeat,
            )
    else:
        raise ValueError("Only 2D and 3D trajectories are supported.")

    return traj_points


def stack_spiral_factory(
    shape: tuple[int, ...],
    accelz: int,
    acsz: int | float,
    n_samples: int,
    nb_revolutions: int,
    in_out: bool = True,
    spiral: str = "archimedes",
    n_shot_slices: int = 1,
    orderz: VDSorder = VDSorder.CENTER_OUT,
    pdfz: VDSpdf = VDSpdf.GAUSSIAN,
    rng: int | None | np.random.Generator = None,
    rotate_angle: AngleRotation | float = 0.0,
) -> np.ndarray:
    """Generate a trajectory of stack of spiral."""
    sizeZ = shape[-1]

    z_index = get_kspace_slice_loc(sizeZ, acsz, accelz, pdf=pdfz, rng=rng, order=orderz)

    rotate_angle = float(rotate_angle)

    spiral2D = initialize_2D_spiral(
        Nc=n_shot_slices,
        Ns=n_samples,
        nb_revolutions=nb_revolutions,
        spiral=spiral,
        in_out=in_out,
    ).reshape(-1, 2)
    z_kspace = (z_index - sizeZ // 2) / sizeZ
    # create the equivalent 3d trajectory
    nsamples = len(spiral2D) // n_shot_slices
    spiral2D = spiral2D.reshape(n_shot_slices, nsamples, 2)
    nz = len(z_kspace)
    kspace_locs3d = np.zeros((nz * n_shot_slices, nsamples, 3), dtype=np.float32)
    # TODO use numpy api for this ?
    for i in range(nz):
        if rotate_angle != 0:
            rotated_spiral = spiral2D @ R2D(rotate_angle * i)
        else:
            rotated_spiral = spiral2D
        kspace_locs3d[i * n_shot_slices : (i + 1) * n_shot_slices, :, :2] = (
            rotated_spiral
        )

        kspace_locs3d[i * n_shot_slices : (i + 1) * n_shot_slices, :, 2] = z_kspace[i]

    return kspace_locs3d.astype(np.float32)


#########################################
# Generators                            #
#########################################


class AngleRotation(float, NoCaseEnum):
    """Available rotation angle for density sampling."""

    ZERO = 0
    GOLDEN = 2.39996322972865332  # 2pi(2-phi)
    GOLDEN_MRI = 1.941678793  # 115.15 deg


def rotate_trajectory(
    trajectories: Generator[np.ndarray, None, None], theta: AngleRotation | float = 0
) -> Generator[np.ndarray, None, None]:
    """Incrementally rotate a trajectory.

    Parameters
    ----------
    trajectories:
        Trajectory to rotate.
    """
    if not isinstance(theta, float):
        theta = theta.value

    for traj in trajectories:
        if traj.shape[-1] == 2:
            rot = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
        else:
            rot = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
            )

        theta += theta

        yield np.einsum("ij,klj->kli", rot, traj)


def stacked_epi2d(
    shape: tuple[int, int, int],
    freq_locs: NDArray,
    phase_locs: NDArray,
    slice_locs: NDArray,
) -> NDArray:
    """Generate a list of 2D epi plane, stacked."""
    Ns, Np, Nf = map(len, (slice_locs, phase_locs, freq_locs))

    epi_3d_coord = np.zeros((Nf * Np * Ns, 3), dtype=np.uint32)
    coord = 0
    for s in range(Ns):
        for p in range(0, Np, 2):
            for f in range(Nf):
                epi_3d_coord[coord] = (slice_locs[s], phase_locs[p], freq_locs[f])
                coord += 1
            if p + 1 >= Np:
                continue  # no room for an extra trip
            for f in range(Nf - 1, -1, -1):
                epi_3d_coord[coord] = (slice_locs[s], phase_locs[p + 1], freq_locs[f])
                coord += 1
    return epi_3d_coord


def stacked_epi_factory(
    shape: tuple[int, int, int],
    acsz: int | float,
    accelz: int,
    orderz: VDSorder = VDSorder.CENTER_OUT,
    pdfz: VDSpdf = VDSpdf.GAUSSIAN,
    rng: int | None | np.random.Generator = None,
) -> np.ndarray:
    """Generate a VDS stack of fully sampled EPI trajectory."""
    sizeZ = shape[-1] #shape[-1]
    z_index = get_kspace_slice_loc(sizeZ, acsz, accelz, pdf=pdfz, rng=rng, order=orderz)

    epi_3d_coord = stacked_epi2d(
        shape,
        slice_locs=z_index,
        phase_locs=np.arange(shape[1]), #shape[0]
        freq_locs=np.arange(shape[0]), #shape[1]
    )

    epi3d_stacked = epi_3d_coord.reshape(len(z_index), shape[1] * shape[0], 3)

    return epi3d_stacked


def evi_factory(
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Generate a Echo Volume Imaging trajectory."""
    epi_3d_coord = stacked_epi2d(
        shape,
        slice_locs=np.arange(shape[0]),
        phase_locs=np.arange(shape[1]),
        freq_locs=np.arange(shape[2]),
    )

    evi = epi_3d_coord.reshape(-1, 3)
    return evi


def trajectory_generator(
    traj_factory: Callable[..., np.ndarray],
    shape: tuple[int, ...],
    **kwargs: Any,
) -> Generator[np.ndarray, None, None]:
    """Generate a trajectory.

    Parameters
    ----------
    traj_factory
        Trajectory factory function.
    n_batch
        Number of shot to deliver at once.
    kwargs
        Trajectory factory kwargs.

    Yields
    ------
    np.ndarray
        Kspace trajectory.
    """
    if kwargs.pop("constant", False):
        logger.debug("constant trajectory")
        traj = traj_factory(shape, **kwargs)
        while True:
            yield traj
    while True:
        yield traj_factory(shape, **kwargs)
