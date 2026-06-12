"""Samplers generate kspace trajectories."""

from __future__ import annotations
import ismrmrd as mrd
import numpy as np
from typing import ClassVar
from numpy.typing import NDArray
from tqdm.auto import tqdm
from ..simulation import SimConfig
from .base import BaseSampler
from .factories import (
    AngleRotation,
    VDSorder,
    VDSpdf,
    stack_spiral_factory,
    stacked_epi_factory,
    evi_factory,
    rotate_trajectory,
    get_kspace_slice_loc
)
from snake.mrd_utils.utils import ACQ
from snake._meta import batched, EnvConfig
from mrinufft.io import read_trajectory
from collections.abc import Generator
from pydantic import Field

class NonCartesianAcquisitionSampler(BaseSampler):
    """
    Base class for non-cartesian acquisition samplers.

    Parameters
    ----------
    constant: bool
        If True, the trajectory is constant.
    obs_time_ms: int
        Time spent to acquire a single shot
    in_out: bool
        If true, the trajectory is acquired with a double join pattern
        from/to the periphery
    ndim: int
        Number of dimensions of the trajectory (2 or 3)
    """

    __engine__ = "NUFFT"
    in_out: bool = True
    obs_time_ms: int = 30

    def TR_vol_ms(self, sim_conf: SimConfig) -> float:
        """Return the volumic TR in ms."""
        single_frame = self.get_next_frame(sim_conf)
        return sim_conf.seq.TR * single_frame.shape[0]

    def add_all_acq_mrd(
        self,
        dataset: mrd.Dataset,
        sim_conf: SimConfig,
    ) -> mrd.Dataset:
        """Generate all mrd_acquisitions."""
        single_frame = self.get_next_frame(sim_conf)
        n_shots_frame = single_frame.shape[0]
        n_samples = single_frame.shape[1]
        TR_vol_ms = sim_conf.seq.TR * single_frame.shape[0]
        n_ksp_frames_true = sim_conf.max_sim_time * 1000 / TR_vol_ms
        n_ksp_frames = int(n_ksp_frames_true)

        trajectory_dimension = single_frame.shape[-1]

        self.log.info("Generating %d frames", n_ksp_frames)
        self.log.info("Frame have %d shots", n_shots_frame)
        self.log.info("Shot have %d samples", n_samples)
        self.log.info("Tobs %.3f ms", n_samples * sim_conf.hardware.dwell_time_ms)
        self.log.info("volume TR: %.3f ms", TR_vol_ms)

        if self.constant:
            self.log.info("Constant Trajectory")

        if n_ksp_frames == 0:
            raise ValueError(
                "No frame can be generated with the current configuration"
                " (TR/shot too long or max_sim_time too short)"
            )
        if n_ksp_frames != n_ksp_frames_true:
            self.log.warning(
                "Volumic TR does not align with max simulation time, "
                "last incomplete frame will be discarded."
            )
            self.log.warning("Updating the max_sim_time to match.")
            sim_conf.max_sim_time = TR_vol_ms * n_ksp_frames / 1000
        self.log.info("Start Sampling pattern generation")
        kspace_data_vol = np.zeros(
            (n_shots_frame, sim_conf.hardware.n_coils, n_samples),
            dtype=np.complex64,
        )

        hdr = mrd.xsd.CreateFromDocument(dataset.read_xml_header())
        hdr.encoding[0].encodingLimits = mrd.xsd.encodingLimitsType(
            kspace_encoding_step_0=mrd.xsd.limitType(0, n_samples, n_samples // 2),
            kspace_encoding_step_1=mrd.xsd.limitType(
                0, n_shots_frame, n_shots_frame // 2
            ),
            repetition=mrd.xsd.limitType(0, n_ksp_frames, 0),
        )
        dataset.write_xml_header(mrd.xsd.ToXML(hdr))  # write the updated header back

        # Write the acquisition.
        # We create the dataset manually with custom dtype.
        # Compared to using mrd.Dataset.append_acquisition
        # - this is faster (20-50%)
        # - uses fixed sized array (All shot have the same size !)
        # - allow for smart chunking (useful for reading/writing efficiently)

        acq_dtype = np.dtype(
            [
                ("head", mrd.hdf5.acquisition_header_dtype),
                ("data", np.float32, (sim_conf.hardware.n_coils * n_samples * 2,)),
                ("traj", np.float32, (n_samples * trajectory_dimension,)),
            ]
        )
        acq_size = np.empty((1,), dtype=acq_dtype).nbytes
        chunk = int(
            np.ceil((n_shots_frame * acq_size) / EnvConfig["SNAKE_HDF5_CHUNK_SIZE"])
        )
        chunk = min(chunk, n_shots_frame)
        chunk_write_sizes = [
            len(c)
            for c in batched(
                range(n_shots_frame * n_ksp_frames),
                int(
                    np.ceil(
                        EnvConfig["SNAKE_HDF5_CHUNK_SIZE"]
                        / (acq_size * n_shots_frame * n_ksp_frames)
                    )
                ),
            )
        ]

        self.log.debug("chunk size for hdf5 %s, elem %s Bytes", chunk, acq_size)

        pbar = tqdm(total=n_ksp_frames * n_shots_frame)
        dataset._dataset.create_dataset(
            "data",
            shape=(n_ksp_frames * n_shots_frame,),
            dtype=acq_dtype,
            chunks=(chunk,),
        )
        write_start = 0
        counter = 0
        for i in range(n_ksp_frames):
            kspace_traj_vol = self.get_next_frame(sim_conf)
            for j in range(n_shots_frame):
                flags = 0
                if j == 0:
                    flags |= ACQ.FIRST_IN_ENCODE_STEP1
                    flags |= ACQ.FIRST_IN_REPETITION
                if j == n_shots_frame - 1:
                    flags |= ACQ.LAST_IN_ENCODE_STEP1
                    flags |= ACQ.LAST_IN_REPETITION

                if counter == 0:
                    current_chunk_size = chunk_write_sizes.pop()
                    acq_chunk = np.empty((current_chunk_size,), dtype=acq_dtype)

                acq_chunk[counter]["head"] = np.frombuffer(
                    mrd.AcquisitionHeader(
                        version=1,
                        flags=flags,
                        scan_counter=counter,
                        sample_time_us=self.obs_time_ms * 1000 / n_samples,
                        center_sample=n_samples // 2 if self.in_out else 0,
                        idx=mrd.EncodingCounters(
                            repetition=i,
                            kspace_encode_step_1=j,
                            kspace_encode_step_2=1,
                        ),
                        active_channels=sim_conf.hardware.n_coils,
                        available_channels=sim_conf.hardware.n_coils,
                        number_of_samples=n_samples,
                        trajectory_dimensions=trajectory_dimension,
                    ),
                    dtype=mrd.hdf5.acquisition_header_dtype,
                )
                acq_chunk[counter]["data"] = (
                    kspace_data_vol[j, :, :].view(np.float32).ravel()
                )
                acq_chunk[counter]["traj"] = np.float32(kspace_traj_vol[j, :]).ravel()
                counter += 1
                if counter == current_chunk_size:
                    counter = 0
                    # write to hdf5 mrd
                    dataset._dataset["data"][
                        write_start : write_start + current_chunk_size
                    ] = acq_chunk
                    write_start += current_chunk_size
                pbar.update(1)

        pbar.close()
        return dataset


class LoadTrajectorySampler(NonCartesianAcquisitionSampler):
    """Load a trajectory from a file.

    Parameters
    ----------
    constant: bool
        If True, the trajectory is constant.
    obs_time_ms: int
        Time spent to acquire a single shot
    in_out: bool
        If true, the trajectory is acquired with a double join pattern
        from/to the periphery
    """

    __sampler_name__ = "load-trajectory"
    __engine__ = "NUFFT"

    path: str
    constant: bool = True
    obs_time_ms: int = 25
    raster_time: float = 0.01
    dwell_time: float = 0.01
    in_out: bool = True
    frame_index: int = 0
    slice_num: int = 1
    read_shots: bool = False

    def _single_frame(self, sim_conf: SimConfig) -> NDArray:
        """Load the trajectory."""
        data = read_trajectory(
            self.path,
            dwell_time=self.dwell_time,
            raster_time=self.raster_time,
            read_shots=self.read_shots
        )[0]
        if not self.constant:
            data = data[self.frame_index*self.slice_num:(self.frame_index+1)*self.slice_num, ...]
            self.frame_index += 1
        data = np.minimum(data, 0.5)
        data = np.maximum(data, -0.5)
        
        return np.float32(data) #single slice (temporary change)
    

class SequentialLoadTrajectorySampler(LoadTrajectorySampler):
    """Load a trajectory from a file.

    Parameters
    ----------
    constant: bool
        If True, the trajectory is constant.
    obs_time_ms: int
        Time spent to acquire a single shot
    in_out: bool
        If true, the trajectory is acquired with a double join pattern
        from/to the periphery
    """

    __sampler_name__ = "sequential-load-trajectory"
    __engine__ = "NUFFT"

    path: list[str]
    constant: bool = True
    obs_time_ms: int = 25
    raster_time: float = 0.01
    dwell_time: float = 0.01
    in_out: bool = True
    frame_index: int = 0

    def _single_frame(self, sim_conf: SimConfig) -> NDArray:
        """Load the trajectory."""
        if self.frame_index >= len(self.path):
            self.frame_index = 0
        data = read_trajectory(
            self.path[self.frame_index],
            dwell_time=self.dwell_time,
            raster_time=self.raster_time,
        )[0]
        data = np.minimum(data, 0.5)
        data = np.maximum(data, -0.5)
        self.frame_index += 1
        return np.float32(data) #single slice (temporary change)


class StackedSequentialLoadTrajectorySampler(LoadTrajectorySampler):
    """Load a  2D trajectory from a file and get a 3D trajectory from it
    Parameters
       Parameters
    ----------
    acsz: float | int
        Number/ proportion of lines to be acquired in the center of k-space.
    accelz: int
        Acceleration factor for the rest of the lines.
    directionz: Literal["center-out", "random"]
        Direction of the acquisition. Either "center-out" or "random".
    pdfz: Literal["gaussian", "uniform"]
        Probability density function of the sampling. Either "gaussian" or "uniform".
    obs_ms: int
        Time spent to acquire a single shot

    in_out: bool
        If true, the spiral is acquired with a double join pattern from/to the periphery
    **wargs:
      k  Extra arguments (smaps, n_jobs, backend etc...)
    """
    __sampler_name__ = "stacked-sequential-load-trajectory"
    __engine__ = "NUFFT"
    accelz : int = 0
    acsz : float | int  = 0.1                # lines or % for calibration 
    orderz : str = "top-down"
    path: list[str]
    constant: bool = True
    obs_time_ms: int 
    raster_time: float
    dwell_time: float
    in_out: bool = True
    frame_index: int = 0
    pdfz = VDSpdf.UNIFORM
    n_shot_slices: int = 1
    frame_index = 0
    constant_2d: bool = True

    max_noncore_overlap_ratio: float = 0.20   # 非中心相邻最大重合比例
    n_planes_per_frame: int = 20             # 你明确每帧 20/40，这里直接固定

  
    _prev_noncore: ClassVar[set[int] | None] = None
    _z_counts: ClassVar[np.ndarray | None] = None

    @property
    def z_counts(self):
        return None if self._z_counts is None else self._z_counts.copy()
    
    def _center_indices(self, sizeZ: int, acsz: int | float) -> list[int]:
        if isinstance(acsz, float):
            n_center = int(round(acsz * sizeZ))
        else:
            n_center = int(acsz)
        n_center = max(0, min(sizeZ, n_center))
        if n_center == 0:
            return []
        center = sizeZ // 2
        idx = np.arange(sizeZ)
        chosen = idx[np.argsort(np.abs(idx - center))[:n_center]]
        return list(map(int, chosen))

    def _select_z_indices(self, sizeZ: int, acsz: int | float, rng) -> np.ndarray:
        from .factories import flip2center
        # init state
        if self._prev_noncore is None:
            self._prev_noncore = set()
        if self._z_counts is None or len(self._z_counts) != sizeZ:
            self._z_counts = np.zeros(sizeZ, dtype=np.int64)
            self._prev_noncore = set()

        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

        center_idx = self._center_indices(sizeZ, acsz)
        center_set = set(center_idx)

        n_total = min(self.n_planes_per_frame, sizeZ)
        if n_total <= len(center_idx):
            z = np.array(center_idx[:n_total], dtype=np.int32)
            return z

        n_noncore = n_total - len(center_idx)
        overlap_limit = int(np.floor(self.max_noncore_overlap_ratio * n_noncore))

        # candidates = all noncore planes
        candidates = [i for i in range(sizeZ) if i not in center_set]

        # sort by (count asc, random tie-break)
        counts = self._z_counts[candidates]
        tie = rng.random(len(candidates))
        order = np.lexsort((tie, counts))
        candidates = [candidates[i] for i in order]

        chosen_noncore = []
        overlap_used = 0

        # pass 1: enforce overlap limit
        for z in candidates:
            if z in self._prev_noncore:
                if overlap_used >= overlap_limit:
                    continue
                overlap_used += 1
            chosen_noncore.append(z)
            if len(chosen_noncore) == n_noncore:
                break

        # pass 2: if not enough (constraint too tight), fill ignoring overlap
        if len(chosen_noncore) < n_noncore:
            chosen_set = set(chosen_noncore)
            for z in candidates:
                if z in chosen_set:
                    continue
                chosen_noncore.append(z)
                if len(chosen_noncore) == n_noncore:
                    break

        # update state (balance on noncore)
        self._z_counts[chosen_noncore] += 1
        self._prev_noncore = set(chosen_noncore)

        z_index = np.array(center_idx + chosen_noncore, dtype=np.int32)

        # apply order of lines
        orderz = VDSorder(self.orderz)
        if orderz == VDSorder.CENTER_OUT:
            line_locs = flip2center(sorted(z_index), sizeZ // 2)
        elif orderz == VDSorder.RANDOM:
            line_locs = rng.permutation(z_index)
        elif orderz == VDSorder.TOP_DOWN:
            line_locs = np.array(sorted(z_index))
        else:
            raise ValueError(f"Unknown direction '{orderz}'.")
        
        return line_locs

    # def stack_EPI_Sparkling_factory(
    #     self,
    #     shape: tuple[int, ...],
    #     accelz: int,
    #     acsz: int | float,
    #     n_samples: int,
    #     in_out: bool = True,
    #     n_shot_slices: int = 1,
    #     orderz: VDSorder = VDSorder.TOP_DOWN,
    #     pdfz: VDSpdf = VDSpdf.UNIFORM,
    #     raster_time: float = 0.01,
    #     dwell_time: float = 0.005,
    #     rng: int | None | np.random.Generator = None,
    #     sim_conf: SimConfig = None,
    #     path: list[str] = None,
    # ) -> np.ndarray:

    #     sizeZ = shape[-1]

    #     # --- 用受约束选择替换原 get_kspace_slice_loc ---
    #     z_index = self._select_z_indices(sizeZ=sizeZ, acsz=acsz, rng=rng)
    #     # ------------------------------------------------

    #     Traj_2D = self.load_2D_traj(sim_conf, path, raster_time, dwell_time)

    #     z_kspace = (z_index - sizeZ // 2) / sizeZ
    #     nsamples = Traj_2D.shape[1] // n_shot_slices
    #     Traj_2D = Traj_2D.reshape(n_shot_slices, nsamples, 2)

    #     nz = len(z_kspace)
    #     kspace_locs3d = np.zeros((nz * n_shot_slices, nsamples, 3), dtype=np.float32)
    #     for i in range(nz):
    #         kspace_locs3d[i * n_shot_slices : (i + 1) * n_shot_slices, :, :2] = Traj_2D
    #         kspace_locs3d[i * n_shot_slices : (i + 1) * n_shot_slices, :, 2] = z_kspace[i]

    #     return kspace_locs3d.astype(np.float32)

    def load_2D_traj(self, sim_conf: SimConfig, path, raster_time, dwell_time) -> NDArray:
        """Load the 2D trajectory."""
        if self.constant_2d:
            self.frame_index = 0
        if self.frame_index >= len(path):
            self.frame_index = 0 
        data = read_trajectory(
            path[self.frame_index],
            dwell_time=dwell_time,
            raster_time=raster_time,
        )[0]
        data = np.minimum(data, 0.5)
        data = np.maximum(data, -0.5)
        self.frame_index += 1
        return np.float32(data) #single slice (temporary change)


    def stack_EPI_Sparkling_factory(
        self,
        shape: tuple[int, ...],
        accelz: int,
        acsz: int | float,
        n_samples: int,
        in_out: bool = True,
        n_shot_slices: int = 1,
        orderz: VDSorder = VDSorder.TOP_DOWN,
        pdfz: VDSpdf = VDSpdf.UNIFORM,
        raster_time: float = 0.01,
        dwell_time: float = 0.005,
        rng: int | None | np.random.Generator = None,
        sim_conf: SimConfig = None,
        path :list[str] = {},

    ) -> np.ndarray:
        """Generate a trajectory of stack of spiral."""
        sizeZ = shape[-1]

        z_index = get_kspace_slice_loc(sizeZ, acsz, accelz, pdf=pdfz, rng=rng, order=orderz)
        Traj_EPI_Sparkling_2D = self.load_2D_traj(sim_conf, path, raster_time, dwell_time)    

        z_kspace = (z_index - sizeZ // 2) / sizeZ
        # create the equivalent 3d trajectory
        nsamples = Traj_EPI_Sparkling_2D.shape[1] // n_shot_slices

        Traj_EPI_Sparkling_2D = Traj_EPI_Sparkling_2D.reshape(n_shot_slices, nsamples, 2)
        nz = len(z_kspace)
        kspace_locs3d = np.zeros((nz * n_shot_slices, nsamples, 3), dtype=np.float32)
        # TODO use numpy api for this ?
        for i in range(nz):
            kspace_locs3d[i * n_shot_slices : (i + 1) * n_shot_slices, :, :2] = (
                Traj_EPI_Sparkling_2D
            )

            kspace_locs3d[i * n_shot_slices : (i + 1) * n_shot_slices, :, 2] = z_kspace[i]

        return kspace_locs3d.astype(np.float32)

    def _single_frame(self, sim_conf: SimConfig) -> NDArray:
        """Generate the sampling pattern."""
        n_samples = int(self.obs_time_ms / sim_conf.hardware.dwell_time_ms)
        return self.stack_EPI_Sparkling_factory(
            shape=sim_conf.shape,
            accelz=self.accelz,
            acsz=self.acsz,
            n_samples=n_samples,
            pdfz=self.pdfz,
            orderz=self.orderz,
            in_out=self.in_out,
            n_shot_slices=self.n_shot_slices,
            rng=sim_conf.rng,
            sim_conf = sim_conf,
            path = self.path,
            raster_time=self.raster_time,
            dwell_time=self.dwell_time,
        )
    
class StackOfSpiralSampler(NonCartesianAcquisitionSampler):
    """
    Spiral 2D Acquisition Handler to generate k-space data.

    Parameters
    ----------
    acsz: float | int
        Number/ proportion of lines to be acquired in the center of k-space.
    accelz: int
        Acceleration factor for the rest of the lines.
    directionz: Literal["center-out", "random"]
        Direction of the acquisition. Either "center-out" or "random".
    pdfz: Literal["gaussian", "uniform"]
        Probability density function of the sampling. Either "gaussian" or "uniform".
    obs_ms: int
        Time spent to acquire a single shot
    nb_revolutions: int
        Number of revolutions of the spiral.
    in_out: bool
        If true, the spiral is acquired with a double join pattern from/to the periphery
    **kwargs:
        Extra arguments (smaps, n_jobs, backend etc...)
    """

    __sampler_name__ = "stack-of-spiral"

    acsz: float | int
    accelz: int
    orderz: str | VDSorder = VDSorder.TOP_DOWN
    nb_revolutions: int = 10
    spiral_name: str = "archimedes"
    pdfz: str | VDSpdf = VDSpdf.GAUSSIAN
    constant: bool = False
    in_out: bool = True
    rotate_angle: AngleRotation = AngleRotation.ZERO
    obs_time_ms: int = 30
    n_shot_slices: int = 1

    def _single_frame(self, sim_conf: SimConfig) -> NDArray:
        """Generate the sampling pattern."""
        n_samples = int(self.obs_time_ms / sim_conf.hardware.raster_time_ms)
        trajs = stack_spiral_factory(
            shape=sim_conf.shape,
            accelz=self.accelz,
            acsz=self.acsz,
            n_samples=n_samples,
            nb_revolutions=self.nb_revolutions,
            pdfz=self.pdfz,
            orderz=self.orderz,
            spiral=self.spiral_name,
            rotate_angle=self.rotate_angle,
            in_out=self.in_out,
            n_shot_slices=self.n_shot_slices,
            rng=sim_conf.rng,
        )
        new_len = int(trajs.shape[1] * (sim_conf.hardware.raster_time_ms /sim_conf.hardware.dwell_time_ms))
        return np.stack([
                    np.stack([
                        np.interp(np.linspace(0, trajs.shape[1] - 1, new_len), np.arange(trajs.shape[1]), trajs[i, :, d])
                        for d in range(trajs.shape[2])
                    ], axis=-1)
                    for i in range(trajs.shape[0])
                ], axis=0)


class RotatedStackOfSpiralSampler(StackOfSpiralSampler):
    """
    Spiral 2D Acquisition Handler to generate k-space data.

    Parameters
    ----------
    rotate_frame_angle: AngleRotation | int
        Angle of rotation of the frame.
    frame_index: int
        Index of the frame.
    **kwargs:
        Extra arguments (smaps, n_jobs, backend etc...)
    """

    __sampler_name__ = "rotated-stack-of-spiral"
    rotate_frame_angle: AngleRotation | float = 0.0
    frame_index: int = 0

    def fix_angle_rotation(
        self, frame: Generator[np.ndarray, None, None], angle: AngleRotation | float = 0
    ) -> Generator[np.ndarray, None, None]:
        """Rotate the trajectory by a given angle."""
        for traj in frame:
            yield from rotate_trajectory((x for x in [traj]), angle)

    def get_next_frame(self, sim_conf: SimConfig) -> NDArray:
        """Generate the next rotated frame."""
        base_frame = self._single_frame(sim_conf)
        if self.constant or self.rotate_frame_angle == 0:
            return base_frame
        else:
            self.frame_index += 1
            rotate_frame_angle = np.pi * (self.rotate_frame_angle / 180)
            base_frame_gen = (traj[None, ...] for traj in base_frame)
            rotated_frame = self.fix_angle_rotation(
                base_frame_gen, float(rotate_frame_angle * self.frame_index)
            )
            return np.concatenate(
                [traj.astype(np.float32) for traj in rotated_frame], axis=0
            )


class EPI3dAcquisitionSampler(BaseSampler):
    """Sampling pattern for EPI-3D."""

    __sampler_name__ = "epi-3d"
    __engine__ = "EPI"

    in_out = True
    acsz: float | int
    accelz: int
    orderz: VDSorder = VDSorder.CENTER_OUT
    pdfz: VDSpdf = VDSpdf.GAUSSIAN

    def _single_frame(self, sim_conf: SimConfig) -> NDArray:
        """Generate the sampling pattern."""
        return stacked_epi_factory(
            shape=sim_conf.shape,
            accelz=self.accelz,
            acsz=self.acsz,
            orderz=self.orderz,
            pdfz=self.pdfz,
            rng=sim_conf.rng,
        )

    def TR_vol_ms(self, sim_conf: SimConfig) -> float:
        """Return the volumic TR in ms."""
        single_frame = self._single_frame(sim_conf)
        n_shots_frame = single_frame.shape[0]
        return sim_conf.seq.TR * n_shots_frame

    def add_all_acq_mrd(
        self,
        dataset: mrd.Dataset,
        sim_conf: SimConfig,
    ) -> mrd.Dataset:
        slice_dim = sim_conf.shape[2]
        phase_dim = sim_conf.shape[1]
        readout_dim = sim_conf.shape[0]
        
        """Create the acquisitions associated with this sampler."""
        single_frame = self._single_frame(sim_conf)
        n_shots_frame = single_frame.shape[0]
        n_lines = phase_dim

        n_samples = single_frame.shape[1]
        TR_vol_ms = sim_conf.seq.TR * single_frame.shape[0]
        n_ksp_frames_true = sim_conf.max_sim_time * 1000 / TR_vol_ms
        n_ksp_frames = int(n_ksp_frames_true)

        self.log.info("Generating %d frames", n_ksp_frames)
        self.log.info("Frame have %d shots", n_shots_frame)
        self.log.info("Tobs %.3f ms", n_samples * sim_conf.hardware.dwell_time_ms)
        self.log.info("Shot have %d samples", n_samples)
        self.log.info("volume TR: %f ms", TR_vol_ms)

        if n_ksp_frames == 0:
            raise ValueError(
                "No frame can be generated with the current configuration"
                " (TR/shot too long or max_sim_time too short)"
            )
        if n_ksp_frames != n_ksp_frames_true:
            self.log.warning(
                "Volumic TR does not align with max simulation time, "
                "last incomplete frame will be discarded."
            )
            self.log.warning("Updating the max_sim_time to match.")
            sim_conf.max_sim_time = TR_vol_ms * n_ksp_frames / 1000
        self.log.info("Start Sampling pattern generation")
        counter = 0
        zero_data = np.zeros(
            (sim_conf.hardware.n_coils, readout_dim), dtype=np.complex64
        )

        # Update the encoding limits.
        # step 0 : frequency (readout directionz)
        # step 1 : phase encoding (blip epi)
        #
        hdr = mrd.xsd.CreateFromDocument(dataset.read_xml_header())
        hdr.encoding[0].encodingLimits = mrd.xsd.encodingLimitsType(
            kspace_encoding_step_0=mrd.xsd.limitType(
                0, readout_dim, readout_dim // 2
            ),
            kspace_encoding_step_1=mrd.xsd.limitType(
                0, phase_dim, phase_dim // 2
            ),
            slice=mrd.xsd.limitType(0, slice_dim, slice_dim // 2),
            repetition=mrd.xsd.limitType(0, n_ksp_frames, 0),
        )

        dataset.write_xml_header(mrd.xsd.ToXML(hdr))  # write the updated header back

        acq_dtype = np.dtype(
            [
                ("head", mrd.hdf5.acquisition_header_dtype),
                (
                    "data",
                    np.float32,
                    (sim_conf.hardware.n_coils * readout_dim * 2,), 
                ),
                ("traj", np.uint32, (readout_dim * 3,)), 
            ]
        )

        acq_size = np.empty((1,), dtype=acq_dtype).nbytes
        chunk = int(
            np.ceil(
                (n_shots_frame * acq_size * n_lines)
                / EnvConfig["SNAKE_HDF5_CHUNK_SIZE"]
            )
        )
        chunk = min(chunk, n_shots_frame * n_lines)  # write at least a chunk per frmae.
        chunk_write_sizes = [
            len(c)
            for c in batched(
                range(n_lines * n_shots_frame * n_ksp_frames),
                int(
                    np.ceil(
                        EnvConfig["SNAKE_HDF5_CHUNK_SIZE"]
                        / (acq_size * n_lines * n_shots_frame * n_ksp_frames)
                    )
                ),
            )
        ]

        self.log.debug("chunk size for hdf5 %s, elem %s Bytes", chunk, acq_size)
        pbar = tqdm(total=n_ksp_frames * n_shots_frame)
        dataset._dataset.create_dataset(
            "data",
            shape=(n_ksp_frames * n_shots_frame * n_lines),
            dtype=acq_dtype,
            chunks=(chunk,),
        )
        write_start = 0
        counter = 0

        for i in range(n_ksp_frames):
            stack_epi3d = self.get_next_frame(sim_conf)  # of shape N_stack, N, 3
            for j, epi2d in enumerate(stack_epi3d):
                epi2d_r = epi2d.reshape(
                    phase_dim, readout_dim, 3
                )  
                for k, readout in enumerate(epi2d_r):
                    flags = 0
                    if k == 0:
                        flags |= ACQ.FIRST_IN_ENCODE_STEP1
                        flags |= ACQ.FIRST_IN_SLICE
                        if j == 0:
                            flags |= ACQ.FIRST_IN_REPETITION
                    if k == len(epi2d_r) - 1:
                        flags |= ACQ.LAST_IN_ENCODE_STEP1
                        flags |= ACQ.LAST_IN_SLICE
                        if j == len(stack_epi3d) - 1:
                            flags |= ACQ.LAST_IN_REPETITION
                            if i == n_ksp_frames - 1:
                                flags |= ACQ.LAST_IN_MEASUREMENT
                    if counter == 0:
                        current_chunk_size = chunk_write_sizes.pop()
                        acq_chunk = np.empty((current_chunk_size,), dtype=acq_dtype)

                    acq_chunk[counter]["head"] = np.frombuffer(
                        mrd.AcquisitionHeader(
                            version=1,
                            flags=flags,
                            scan_counter=counter,
                            sample_time_us=sim_conf.hardware.dwell_time_ms
                            * 1000
                            / n_samples,
                            center_sample=n_samples // 2 if self.in_out else 0,
                            idx=mrd.EncodingCounters(
                                repetition=i,
                                kspace_encode_step_1=readout[0, 1],
                                slice=readout[0, 0],
                            ),
                            read_dir=dir_cos(readout[0], readout[1]),
                            active_channels=sim_conf.hardware.n_coils,
                            available_channels=sim_conf.hardware.n_coils,
                            number_of_samples=len(readout),#what should readout be?? 
                            trajectory_dimensions=3,
                        ),
                        dtype=mrd.hdf5.acquisition_header_dtype,
                    ).copy()
                    acq_chunk[counter]["data"] = zero_data.view(np.float32).ravel()
                    acq_chunk[counter]["traj"] = readout[:, ::-1].astype(
                        np.uint32, copy=False
                    ).ravel() # [1, 72, 60] => [60, 72, 1]
                    counter += 1
                    if counter == current_chunk_size:
                        counter = 0
                        # write to hdf5 mrd
                        dataset._dataset["data"][
                            write_start : write_start + current_chunk_size
                        ] = acq_chunk
                        write_start += current_chunk_size
                pbar.update(1)

        pbar.close()

        dataset._file.flush()  # Empty all buffers to disk
        return dataset


class EVI3dAcquisitionSampler(BaseSampler):
    """SAmpler for EVI acquisition."""

    __sampler_name__ = "evi"
    __engine__ = "EVI"

    in_out = True

    def _single_frame(self, sim_conf: SimConfig) -> NDArray:
        """Generate the sampling pattern."""
        epi_coords = evi_factory(
            shape=sim_conf.shape,
        ).reshape(*sim_conf.shape, 3)
        return epi_coords

    def add_all_acq_mrd(
        self,
        dataset: mrd.Dataset,
        sim_conf: SimConfig,
    ) -> mrd.Dataset:
        """Create the acquisitions associated with this sampler."""
        single_frame = self._single_frame(sim_conf)
        n_samples = (
            single_frame.shape[1] * single_frame.shape[2] * single_frame.shape[0]
        )

        TR_vol_ms = sim_conf.seq.TR
        n_ksp_frames_true = sim_conf.max_sim_time * 1000 / TR_vol_ms
        n_ksp_frames = int(n_ksp_frames_true)

        self.log.info("Generating %d frames", n_ksp_frames)
        self.log.info("Frame have %d shots", 1)
        self.log.info("Tobs %.3f ms", n_samples * sim_conf.hardware.dwell_time_ms)
        self.log.info("Shot have %d samples", n_samples)
        self.log.info("volume TR: %f ms", TR_vol_ms)

        if n_ksp_frames == 0:
            raise ValueError(
                "No frame can be generated with the current configuration"
                " (TR/shot too long or max_sim_time too short)"
            )
        if n_ksp_frames != n_ksp_frames_true:
            self.log.warning(
                "Volumic TR does not align with max simulation time, "
                "last incomplete frame will be discarded."
            )
            self.log.warning("Updating the max_sim_time to match.")
            sim_conf.max_sim_time = TR_vol_ms * n_ksp_frames / 1000
        self.log.info("Start Sampling pattern generation")
        counter = 0
        zero_data = np.zeros(
            (sim_conf.hardware.n_coils, sim_conf.shape[2]), dtype=np.complex64
        )

        # Update the encoding limits.
        # step 0 : frequency (readout directionz)
        # step 1 : phase encoding (blip epi)
        #
        hdr = mrd.xsd.CreateFromDocument(dataset.read_xml_header())
        hdr.encoding[0].encodingLimits = mrd.xsd.encodingLimitsType(
            kspace_encoding_step_0=mrd.xsd.limitType(
                0, sim_conf.shape[2], sim_conf.shape[2] // 2
            ),
            kspace_encoding_step_1=mrd.xsd.limitType(
                0, sim_conf.shape[1], sim_conf.shape[1] // 2
            ),
            slice=mrd.xsd.limitType(0, sim_conf.shape[0], sim_conf.shape[0] // 2),
            repetition=mrd.xsd.limitType(0, n_ksp_frames, 0),
        )

        dataset.write_xml_header(mrd.xsd.ToXML(hdr))  # write the updated header back

        acq_dtype = np.dtype(
            [
                ("head", mrd.hdf5.acquisition_header_dtype),
                (
                    "data",
                    np.float32,
                    (sim_conf.hardware.n_coils * sim_conf.shape[2] * 2,),
                ),
                ("traj", np.uint32, (sim_conf.shape[2] * 3,)),
            ]
        )

        # Write the acquisition.
        # We create the dataset manually with custom dtype.
        # Compared to using mrd.Dataset.append_acquisition
        # - this is faster !
        # - uses fixed sized array (All shot have the same size !)
        # - allow for smart chunking (useful for reading/writing efficiently)
        acq = np.empty(
            (n_ksp_frames * sim_conf.shape[1] * sim_conf.shape[0],), dtype=acq_dtype
        )

        for i in range(n_ksp_frames):
            stack_epi3d = self._single_frame(sim_conf)  # of shape N_stack, N, 3
            for j, epi2d in enumerate(stack_epi3d):
                epi2d_r = epi2d.reshape(
                    sim_conf.shape[1], sim_conf.shape[2], 3
                )  # reorder to have
                for k, readout in enumerate(epi2d_r):
                    flags = 0
                    if k == 0:
                        flags |= ACQ.FIRST_IN_ENCODE_STEP1
                        flags |= ACQ.FIRST_IN_SLICE
                        if j == 0:
                            flags |= ACQ.FIRST_IN_REPETITION
                    if k == len(epi2d_r) - 1:
                        flags |= ACQ.LAST_IN_ENCODE_STEP1
                        flags |= ACQ.LAST_IN_SLICE
                        if j == len(stack_epi3d) - 1:
                            flags |= ACQ.LAST_IN_REPETITION
                            if i == n_ksp_frames - 1:
                                flags |= ACQ.LAST_IN_MEASUREMENT
                    acq[counter]["head"] = np.frombuffer(
                        mrd.AcquisitionHeader(
                            version=1,
                            flags=flags,
                            scan_counter=counter,
                            sample_time_us=sim_conf.hardware.dwell_time_ms
                            * 1000
                            / n_samples,
                            center_sample=n_samples // 2 if self.in_out else 0,
                            idx=mrd.EncodingCounters(
                                repetition=i,
                                kspace_encode_step_1=readout[0, 1],
                                slice=readout[0, 0],
                            ),
                            read_dir=dir_cos(readout[0], readout[1]),
                            active_channels=sim_conf.hardware.n_coils,
                            available_channels=sim_conf.hardware.n_coils,
                            number_of_samples=len(readout),
                            trajectory_dimensions=3,
                        ),
                        dtype=mrd.hdf5.acquisition_header_dtype,
                    ).copy()
                    acq[counter]["data"] = zero_data.view(np.float32).ravel()
                    acq[counter]["traj"] = np.float32(readout).view(np.float32).ravel()
                    counter += 1

        dataset._dataset.create_dataset(
            "data",
            data=acq,
            chunks=min(sim_conf.shape[1] * sim_conf.shape[0], len(acq)),
        )
        return dataset


def dir_cos(start: NDArray, end: NDArray) -> tuple[np.float32]:
    """Compute the directional cosine of the vector from beg to end point."""
    diff = np.float32(end) - np.float32(start)
    cos = diff / np.sqrt(np.sum(diff**2))
    return tuple(cos)
