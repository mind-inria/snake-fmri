"""Loader of MRD data."""

from __future__ import annotations

import logging
import os
import functools
from functools import cached_property
import ismrmrd as mrd
import h5py
import numpy as np
from numpy.typing import NDArray
from typing import Any, TYPE_CHECKING, overload
from collections.abc import Generator
from .._meta import LogMixin

if TYPE_CHECKING:
    from ..core import Phantom, DynamicData
    from ..core import SimConfig

from .utils import b64encode2obj

log = logging.getLogger(__name__)

GenericPath = os.PathLike | str


def read_mrd_header(filename: GenericPath | mrd.Dataset) -> mrd.xsd.ismrmrdHeader:
    """Read the header of the MRD file."""
    if isinstance(filename, mrd.Dataset):
        dataset = filename
    else:
        dataset = mrd.Dataset(filename, create_if_needed=False)

    header = mrd.xsd.CreateFromDocument(dataset.read_xml_header())

    if not isinstance(filename, mrd.Dataset):
        dataset.close()

    return header


class MRDLoader(LogMixin):
    """Base class for MRD data loader.

    This is to be used as a context manager.

    It reimplements most of the methods of the mrd.Dataset class, and adds some
    useful wrappers. With this dataloader, you can open the dataset in readonly mode,
    which is not possible with mrd.
    """

    def __init__(
        self,
        filename: GenericPath,
        dataset_name: str = "dataset",
        writeable: bool = False,
        swmr: bool = False,
        squeeze_dims: bool = True,
    ):
        self._filename = filename
        self._dataset_name = dataset_name
        self._writeable = writeable
        self._swmr = swmr
        self._level = 0
        self._file: h5py.File | None = None
        self._squeeze_dims = squeeze_dims

    def __enter__(self):
        # Track the number of times the dataloader is used as a context manager
        # to close the file only when the last context manager is closed.
        if self._file is None:
            self._file = h5py.File(
                self._filename,
                "r+" if self._writeable else "r",
                libver="latest",
                swmr=self._swmr,
            )
            try:
                header = self.header
            except LookupError:
                log.warning(
                    "No matrix size found in the header."
                    " The header is probably missing."
                )
                self._shape = None
            else:
                matrixSize = header.encoding[0].encodedSpace.matrixSize
                self._shape = matrixSize.x, matrixSize.y, matrixSize.z
                if self._squeeze_dims:
                    self._shape = tuple(filter(lambda x: x > 1, self._shape))
        self._level += 1
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        self._level -= 1
        if self._level == 0 and self._file is not None:
            self._file.close()
            self._file = None

    def iter_frames(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        shot_dim: bool = False,
    ) -> Generator[tuple[int, NDArray[np.float32], NDArray[np.complex64]], None, None]:
        """Iterate over kspace frames of the dataset.

        Parameters
        ----------
        start : int, optional
            Start index of the iteration.
        stop : int, optional
            Stop index of the iteration.
        step: int, optional
            Step of the iteration.
        shot_dim: bool, optional
            Return the data reshaped with the shot dimension first.

        Yields
        ------
        tuple[int, np.ndarray, np.ndarray]
            The index of the frame, the trajectory and the kspace data.

        """
        if start is None:
            start = 0
        if stop is None:
            stop = self.n_frames
        if step is None:
            step = 1
        with self:
            for i in np.arange(start, stop, step):
                yield i, *self.get_kspace_frame(i, shot_dim=shot_dim)

    @overload
    def get_kspace_frame(
        self, idx: int
    ) -> tuple[NDArray[np.float32], NDArray[np.complex64]]:
        # Get k-space frame trajectory/mask and data.
        raise NotImplementedError()

    ###########################
    # MRD interfaces methods  #
    ###########################

    def _read_xml_header(self) -> mrd.xsd.ismrmrdHeader:
        """Read the header of the MRD file."""
        if "xml" not in self._dataset:
            raise LookupError("XML header not found in the dataset.")
        return self._dataset["xml"][0]

    def _read_waveform(self, wavnum: int) -> mrd.Waveform:
        if "waveforms" not in self._dataset:
            raise LookupError("Acquisition data not found in the dataset.")

        # create a Waveform
        # and fill with the header for this waveform
        # We start with an array of zeros to avoid garbage in the padding bytes.
        header_array = np.zeros((1,), dtype=mrd.file.waveform_header_dtype)
        header_array[0] = self._dataset["waveforms"][wavnum]["head"]

        wav = mrd.Waveform(header_array)

        # copy the data as uint32
        wav.data[:] = (
            self._dataset["waveforms"][wavnum]["data"]
            .view(np.uint32)
            .reshape((wav.channels, wav.number_of_samples))[:]
        )

        return wav

    def _read_image(self, impath: str, imnum: int = 0) -> mrd.Image:
        # create an image
        # and fill with the header and attribute string for this image
        im = mrd.Image(
            self._dataset[impath]["header"][imnum],
            self._dataset[impath]["attributes"][imnum],
        )

        # copy the data
        # ismrmrd complex data is stored as pairs named real and imag
        # TODO do we need to store and reset or the config local to the module?
        cplxcfg = h5py.get_config().complex_names
        h5py.get_config().complex_names = ("real", "imag")
        im.data[:] = self._dataset[impath]["data"][imnum]
        h5py.get_config().complex_names = cplxcfg

        return im

    ######################
    # dataset properties #
    ######################
    @cached_property
    def header(self) -> mrd.xsd.ismrmrdHeader:
        """Get the header from the mrd file."""
        return mrd.xsd.CreateFromDocument(self._read_xml_header())

    @property
    def _dataset(self) -> h5py.Dataset:
        """Get MRD dataset."""
        if self._file is not None:
            return self._file[self._dataset_name]
        else:
            raise FileNotFoundError(
                "Dataset not opened. Use the dataloader as a context manager."
            )

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return self.header.encoding[0].encodingLimits.repetition.maximum

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the volume."""
        try:
            return self._shape
        except AttributeError as exc:
            raise RuntimeError(
                "You need to run the dataloader as a context manager."
            ) from exc

    @property
    def n_coils(self) -> int:
        """Number of coils."""
        return self.header.acquisitionSystemInformation.receiverChannels

    @property
    def n_acquisition(self) -> int:
        """Number of acquisition in the dataset."""
        return self._dataset["data"].size

    def __len__(self):
        return self.n_frames

    @property
    def n_sample(self) -> int:
        """Number of samples in a single acquisition."""
        return self.header.encoding[0].encodingLimits.kspace_encoding_step_0.maximum

    @property
    def n_shots(self) -> int:
        """Number of samples in a single acquisition.

        Notes
        -----
        for EPI this is the number of phase encoding lines in the EPI zigzag.
        """
        return self.header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum

    @property
    def engine_model(self) -> str:
        """Get the engine model."""
        return self.header.userParameters.userParameterString[0].value

    @property
    def slice_2d(self) -> bool:
        """Is the acquisition run on 2D slices."""
        return self.header.userParameters.userParameterString[1].value == "True"

    #############
    # Get data  #
    #############
    @functools.lru_cache  # noqa
    def get_phantom(self, imnum: int = 0) -> Phantom:
        """Load the phantom from the dataset."""
        from ..core import Phantom

        return Phantom.from_mrd_dataset(self, imnum)

    @cached_property
    def _all_waveform_infos(self) -> dict[int, dict]:
        return parse_waveform_information(self.header)

    def get_dynamic(self, waveform_num: int) -> DynamicData:
        """Get dynamic data."""
        waveform = self._read_waveform(waveform_num)
        wave_info = self._all_waveform_infos[waveform.waveform_id]
        from ..core import DynamicData

        return DynamicData._from_waveform(waveform, wave_info)

    def get_all_dynamic(self) -> list[DynamicData]:
        """Get all dynamic data."""
        from ..core import DynamicData

        all_dyn_data = []
        try:
            n_waves = self._dataset["waveforms"].size
        except Exception as e:
            log.debug(e)
            return []

        for i in range(n_waves):
            waveform = self._read_waveform(i)
            wave_info = self._all_waveform_infos[waveform.waveform_id]
            all_dyn_data.append(DynamicData._from_waveform(waveform, wave_info))
        return all_dyn_data

    def get_timeserie(
        self, name: str
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Get a dynamic timeseries and the associated sampled time (every TRshot).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The sampled time and the dynamic timeserie.
        """
        dyn_datas = self.get_all_dynamic()
        good_d = None
        for d in dyn_datas:
            if d.name == name:
                good_d = d
                break
        if good_d is None:
            raise ValueError("No dynamic data found matching waveform name")

        signal = good_d.data[0]
        sample_time = np.arange(len(signal)) * self.get_sim_conf().seq.TR / 1000
        return sample_time, signal

    @functools.lru_cache  # noqa
    def get_sim_conf(self) -> SimConfig:
        """Parse the sim config."""
        return parse_sim_conf(self.header)

    def _get_image_data(self, name: str, idx: int = 0) -> NDArray[np.complex64] | None:
        try:
            image = self._read_image(name, idx).data.astype(np.complex64)
        except LookupError:
            log.warning(f"No {name} found in the dataset.")
            return None
        return image

    def get_smaps(self, resample: bool = True) -> NDArray[np.complex64] | None:
        """Load the sensitivity maps from the dataset.

        Parameters
        ----------
        resample: bool
            If resample is True (default), the sensitivity maps are resampled using the
            affine transformation
        describe in the phantom and sim_conf.

        Returns
        -------
        None: if no Smaps is found
        NDArray: The coils sensitivity maps
        """
        sim_conf = self.get_sim_conf()
        try:
            smaps_im = self._read_image("smaps")
        except KeyError:
            return None
        else:
            smaps_affine = get_affine_from_image(smaps_im)
            smaps = smaps_im.data
            sim_conf = self.get_sim_conf()
            if resample:
                from snake.core.transform import apply_affine4d

                smaps = apply_affine4d(
                    smaps,
                    smaps_affine,
                    sim_conf.fov.affine,
                    new_shape=sim_conf.fov.shape,
                    use_gpu=True,
                )
            if self._squeeze_dims:
                return smaps.squeeze()
            return smaps

    def get_coil_cov(self) -> NDArray | None:
        """Load the coil covariance from the dataset."""
        return self._get_image_data("coil_cov")

class SliceDataloader(MRDLoader):
    """Load slice MRD files k-space frames iteratively."""

    def __init__(self,
                 frame_dl: MRDLoader,
                 ): 
        super().__init__(
            filename=frame_dl._filename,
            dataset_name=frame_dl._dataset_name,
            writeable=frame_dl._writeable,
            swmr=frame_dl._swmr,
            squeeze_dims=frame_dl._squeeze_dims
        )
        self.frame = frame_dl
        self.is_cartesian = isinstance(self.frame, CartesianFrameDataLoader)
        self.is_non_cartesian = isinstance(self.frame, NonCartesianFrameDataLoader)
        self.get_kspace_frame = self._get_kspace_frame

    def __getattr__(self, name: str) -> Any:
        return getattr(self.frame, name)

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return self.frame.n_acquisition 

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the volume."""
        return self.frame.shape[:2]
    
    def _get_kspace_frame(
        self, idx: int, shot_dim: bool = False
         ) -> tuple[NDArray[np.float32], NDArray[np.complex64]]:
        """Get the k-space frame."""
        n_acq_per_frame = self.frame.n_acquisition // self.frame.n_frames
        traj, data = self.frame.get_kspace_frame(idx // self.frame.n_shots)
        traj =  traj.reshape(n_acq_per_frame, -1, 3)
        data = data.reshape(self.frame.n_coils, n_acq_per_frame, -1)

        return traj[idx%self.frame.n_shots,:,:2], data[:,idx%self.frame.n_shots,:]
    
class CartesianFrameDataLoader(MRDLoader):
    """Load cartesian MRD files k-space frames iteratively.

    Parameters
    ----------
    filename: source for the MRD file.

    Examples
    --------
    >>> for mask, kspace in CartesianFrameDataLoader("test.mrd"):
            image = ifft(kspace)
    """

    def get_kspace_frame(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Get the k-space frame."""
        kspace = np.zeros((self.n_coils, *self.shape), dtype=np.complex64)
        mask = np.zeros(self.shape, dtype=bool)

        n_acq_per_frame = self.n_acquisition // self.n_frames
        start = idx * n_acq_per_frame
        end = (idx + 1) * n_acq_per_frame
        # Do a single read of the dataset much faster !
        acq = self._dataset["data"][start:end]
        traj = acq["traj"].reshape(-1, 3)
        data = acq["data"].view(np.complex64)
        data = data.reshape(-1, self.n_shots, self.n_coils, self.n_sample)
        data = np.moveaxis(data, 2, 0)  # putting the coil dimension first.
        data = data.reshape(self.n_coils, -1)
        traj_locs: tuple = tuple(np.int32(traj.T))  # type: ignore
        for c in range(self.n_coils):
            kspace[c][traj_locs] = data[c]
        mask[traj_locs] = True
        return mask, kspace


class NonCartesianFrameDataLoader(MRDLoader):
    """Non Cartesian Dataloader.

    Iterate over the acquisition of the MRD file.

    Examples
    --------
    >>> from mrinufft import get_operator
    >>> dataloader =  NonCartesianFrameDataLoader("test.mrd")
    >>> for mask, kspace in data_loader:
    ...     nufft = get_operator("finufft")(traj,
    ...     shape=dataloader.shape, n_coils=dataloader.n_coils)
    ...     image = nufft.adj_op(kspace)
    """

    def get_kspace_frame(
        self, idx: int, shot_dim: bool = False
    ) -> tuple[NDArray[np.float32], NDArray[np.complex64]]:
        """Get the k-space frame and the associated trajectory.

        Parameters
        ----------
        idx : int
            Index of the frame to get.
        shot_dim : bool, optional
            Return the data reshaped with the shot dimension first.

        Returns
        -------
        np.ndarray
            The trajectory.
        np.ndarray
            The kspace data.
        """
        n_acq_per_frame = self.n_acquisition // self.n_frames
        start = idx * n_acq_per_frame
        end = (idx + 1) * n_acq_per_frame
        # Do a single read of the dataset much faster !
        acq = self._dataset["data"][start:end]
        traj = acq["traj"].reshape(n_acq_per_frame, -1, 3)
        data = acq["data"].view(np.complex64)
        data = data.reshape(-1, self.n_coils, self.n_sample)
        data = np.moveaxis(data, 1, 0)
        data = data.reshape(self.n_coils, n_acq_per_frame, -1)
        if not shot_dim:
            return (
                traj.reshape(-1, 3),
                data.reshape(self.n_coils, -1),
            )
        return traj, data


def parse_sim_conf(header: mrd.xsd.ismrmrdHeader) -> SimConfig:
    """Parse the header to populate SimConfig from an MRD Header."""
    from ..core import GreConfig, HardwareConfig, SimConfig, FOVConfig

    n_coils = header.acquisitionSystemInformation.receiverChannels
    field = header.acquisitionSystemInformation.systemFieldStrength_T

    TR = header.sequenceParameters.TR[0]
    TE = header.sequenceParameters.TE[0]
    FA = header.sequenceParameters.flipAngle_deg[0]
    seq = GreConfig(TR=TR, TE=TE, FA=FA)

    caster = {
        "gmax": float,
        "smax": float,
        "dwell_time_ms": float,
        "max_sim_time": int,
        "rng_seed": int,
        "TR_eff": float,
    }

    parsed = {
        up.name: caster[up.name](up.value)
        for up in header.userParameters.userParameterDouble
        if up.name in caster.keys()
    }
    if set(caster.keys()) != set(parsed.keys()):
        raise ValueError(
            f"Missing parameters {set(caster.keys()) - set(parsed.keys())}"
        )
    caster_str = {
        "fov_config": str,
    }
    parsed_str = {
        up.name: caster_str[up.name](up.value)
        for up in header.userParameters.userParameterString
        if up.name in caster_str.keys()
    }
    if set(caster_str.keys()) != set(parsed_str.keys()):
        raise ValueError(
            f"Missing parameters {set(caster_str.keys()) - set(parsed_str.keys())}"
        )

    hardware = HardwareConfig(
        gmax=parsed.pop("gmax"),
        smax=parsed.pop("smax"),
        dwell_time_ms=parsed.pop("dwell_time_ms"),
        n_coils=n_coils,
        field=field,
    )

    seq.TR_eff = parsed.pop("TR_eff")

    fov_mm = header.encoding[0].encodedSpace.fieldOfView_mm
    fov_mm = (fov_mm.x, fov_mm.y, fov_mm.z)
    shape = header.encoding[0].encodedSpace.matrixSize
    shape = (shape.x, shape.y, shape.z)

    sim_conf = SimConfig(
        max_sim_time=parsed.pop("max_sim_time"),
        seq=seq,
        hardware=hardware,
        rng_seed=parsed.pop("rng_seed"),
    )
    sim_conf.fov: FOVConfig = eval(parsed_str.pop("fov_config"))

    return sim_conf


def parse_waveform_information(hdr: mrd.xsd.ismrmrdHeader) -> dict[int, dict]:
    """Parse the waveform information from the MRD file.

    Returns a dictionary with id as key and waveform information
    (name, parameters, etc.. ) as value.

    Base64 encoded parameters are decoded.
    """
    waveform_info = dict()
    for wi in hdr.waveformInformation:
        infos = {"name": wi.waveformName}
        for ptype, p in wi.userParameters.__dict__.items():
            for pp in p:
                if ptype == "userParameterBase64":
                    infos[pp.name] = b64encode2obj(pp.value)
                elif ptype == "userParameterString":
                    infos[pp.name] = pp.value
                elif ptype == "userParameterLong":
                    infos[pp.name] = pp.value
                elif ptype == "userParameterDouble":
                    infos[pp.name] = pp.value
                else:
                    raise ValueError(f"Unknown parameter type {ptype}")

        waveform_info[int(wi.waveformType)] = infos

    return waveform_info


def get_affine_from_image(image: mrd.Image) -> NDArray[np.float32]:
    """Extract the affine matrix from the header of an MRD.Image."""
    # Affine matrix from the header
    position = image._head.position
    read_dir = image._head.read_dir
    phase_dir = image._head.phase_dir
    slice_dir = image._head.slice_dir
    affine = np.eye(4, dtype=np.float32)
    # Reverse matrix size because the MRD file is messed up
    # See imsmrd/image.py:296 (matrix_size property warning)
    res = np.array(image._head.field_of_view) / np.array(image._head.matrix_size[::-1])
    affine[:3, 3] = -position[0], -position[1], position[2]
    affine[:3, 0] = (
        -read_dir[0] * res[0],
        -read_dir[1] * res[0],
        read_dir[2] * res[0],
    )
    affine[:3, 1] = (
        -phase_dir[0] * res[1],
        -phase_dir[1] * res[1],
        phase_dir[2] * res[1],
    )
    affine[:3, 2] = (
        -slice_dir[0] * res[2],
        -slice_dir[1] * res[2],
        slice_dir[2] * res[2],
    )
    return affine
