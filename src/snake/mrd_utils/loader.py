"""Loader of MRD data."""

import atexit
import logging
import os

from functools import cached_property
import ismrmrd as mrd
import numpy as np
from numpy.typing import NDArray

from snake._meta import LogMixin
from snake.phantom import Phantom, DynamicData
from snake.simulation import GreConfig, HardwareConfig, SimConfig

from .utils import b64encode2obj, ACQ

log = logging.getLogger(__name__)


def read_mrd_header(filename: os.PathLike | mrd.Dataset) -> mrd.xsd.ismrmrdHeader:
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
    """Base class for MRD data loader."""

    def __init__(self, filename: os.PathLike | mrd.Dataset):
        if isinstance(filename, mrd.Dataset):
            self.dataset = filename
            self.filename = filename._file.filename
        else:
            self.filename = filename
            self.dataset = mrd.Dataset(filename, create_if_needed=False)

        self.header = mrd.xsd.CreateFromDocument(self.dataset.read_xml_header())
        matrixSize = self.header.encoding[0].encodedSpace.matrixSize
        self.shape = matrixSize.x, matrixSize.y, matrixSize.z
        atexit.register(self._cleanup)

    def get_sim_conf(self) -> SimConfig:
        """Parse the header to populate SimConfig."""
        return parse_sim_conf(self.header)

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return self.header.encoding[0].encodingLimits.repetition.maximum

    @property
    def n_coils(self) -> int:
        """Number of coils."""
        return self.header.acquisitionSystemInformation.receiverChannels

    def __len__(self):
        return self.n_frames

    @property
    def n_sample(self) -> int:
        """Number of samples in a single acquisition."""
        return self.header.encoding[0].limits.kspace_encoding_step_0.maximum

    @property
    def n_shots(self) -> int:
        """Number of samples in a single acquisition.

        Notes
        -----
        for EPI this is the number of phase encoding lines in the EPI zigzag.
        """
        return self.header.encoding[0].limits.kspace_encoding_step_1.maximum

    def get_smaps(self) -> NDArray | None:
        """Load the sensitivity maps from the dataset."""
        return load_smaps(self.dataset)

    def get_coil_cov(self) -> NDArray | None:
        """Load the coil covariances from the dataset."""
        return load_coil_cov(self.dataset)

    def get_phantom(self) -> Phantom:
        """Load the phantom from the dataset."""
        return Phantom.from_mrd_dataset(self.dataset)

    @cached_property
    def _all_waveform_infos(self) -> dict[int, dict]:
        return parse_waveform_information(self.dataset)

    def get_dynamic(self, waveform_num: int) -> DynamicData:
        """Get dynamic data."""
        waveform = self.dataset.read_waveform(waveform_num)
        wave_info = self._all_waveform_infos[waveform.waveform_id]
        return DynamicData._from_waveform(waveform, wave_info)

    def get_all_dynamic(self) -> list[DynamicData]:
        """Get all dynamic data."""
        all_dyn_data = []
        try:
            n_waves = self.dataset.number_of_waveforms()
        except Exception as e:
            log.error(e)
            return []

        for i in range(n_waves):
            waveform = self.dataset.read_waveform(i)
            wave_info = self._all_waveform_infos[waveform.waveform_id]
            all_dyn_data.append(DynamicData._from_waveform(waveform, wave_info))
        return all_dyn_data

    def parse_sim_conf(self) -> SimConfig:
        """Parse the sim config."""
        return parse_sim_conf(self.header)

    def _cleanup(self) -> None:
        try:
            self.dataset.close()
        except Exception as e:
            self.log.error(e)
            pass

    def __iter__(self):
        raise NotImplementedError


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

    def __iter__(self):
        counter = 0
        yielded = False
        kspace = np.zeros((self.n_coils, *self.shape), dtype=np.complex64)
        mask = np.zeros(self.shape, dtype=bool)
        acq = self.dataset.read_acquisition(counter)
        n_acq = self.dataset.number_of_acquisitions()
        while counter < n_acq:
            traj_locs = tuple(np.int32(acq.traj.T))
            for c in range(self.n_coils):  # FIXME what is the good way of doing this ?
                kspace[c][traj_locs] = acq.data[c]

            mask[traj_locs] = True
            if (
                acq.flags & ACQ.LAST_IN_REPETITION
                or acq.flags & ACQ.LAST_IN_MEASUREMENT
            ):
                yield mask, kspace
                kspace[:] = 0
                mask[:] = False
                yielded = True
            counter += 1
            if counter < self.dataset.number_of_acquisitions():
                acq = self.dataset.read_acquisition(counter)
                if yielded:
                    yielded = False
                    if not (acq.flags & ACQ.FIRST_IN_REPETITION):
                        raise ValueError(
                            f"Flags error at {counter} {ACQ(acq.flags).__repr__()}"
                        )


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

    def __iter__(self):
        counter = 0
        shot_counter = 0
        yielded = False
        kspace = np.zeros(
            (self.n_coils, self.n_shots, self.n_sample), dtype=np.complex64
        )
        acq = self.dataset.read_acquisition(counter)
        n_acq = self.dataset.number_of_acquisitions()
        samples_locs = np.zeros(
            (self.n_shots, self.n_sample, len(self.shape)), dtype=bool
        )
        while counter < n_acq:
            for c in range(self.n_coils):  # FIXME what is the good way of doing this ?
                kspace[c, shot_counter] = acq.data[c]
                samples_locs[shot_counter] = acq.traj.reshape(-1, len(self.shape))
            if (
                acq.flags & ACQ.LAST_IN_REPETITION
                or acq.flags & ACQ.LAST_IN_MEASUREMENT
            ):
                yield samples_locs, kspace
                kspace[:] = 0
                samples_locs[:] = 0
                shot_counter = 0
                yielded = True
            counter += 1
            shot_counter += 1
            if counter < self.dataset.number_of_acquisitions():
                acq = self.dataset.read_acquisition(counter)
                if yielded:
                    yielded = False
                    if not (acq.flags & ACQ.FIRST_IN_REPETITION):
                        raise ValueError(
                            f"Flags error at {counter} {ACQ(acq.flags).__repr__()}"
                        )


def parse_sim_conf(header: mrd.xsd.ismrmrdHeader | mrd.Dataset) -> SimConfig:
    """Parse the header to populate SimConfig."""
    if isinstance(header, mrd.Dataset):
        header = mrd.xsd.CreateFromDocument(header.read_xml_header())

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

    hardware = HardwareConfig(
        gmax=parsed.pop("gmax"),
        smax=parsed.pop("smax"),
        dwell_time_ms=parsed.pop("dwell_time_ms"),
        n_coils=n_coils,
        field=field,
    )

    fov_mm = header.encoding[0].encodedSpace.fieldOfView_mm
    fov_mm = (fov_mm.x, fov_mm.y, fov_mm.z)
    shape = header.encoding[0].encodedSpace.matrixSize
    shape = (shape.x, shape.y, shape.z)

    return SimConfig(
        max_sim_time=parsed.pop("max_sim_time"),
        seq=seq,
        hardware=hardware,
        fov_mm=fov_mm,
        shape=shape,
        rng_seed=parsed.pop("rng_seed"),
    )


def parse_waveform_information(dataset: mrd.Dataset) -> dict[int, dict]:
    """Parse the waveform information from the MRD file.

    Returns a dictionary with id as key and waveform information
    (name, parameters, etc.. ) as value.

    Base64 encoded parameters are decoded.
    """
    hdr = mrd.xsd.CreateFromDocument(dataset.read_xml_header())
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


def _load_image(dataset: mrd.Dataset, name: str, idx: int = 0) -> NDArray | None:
    try:
        image = dataset.read_image(name, idx).data
    except LookupError:
        log.warning(f"No {name} found in the dataset.")
        return None
    return image


def load_smaps(dataset: mrd.Dataset) -> NDArray | None:
    """Load the sensitivity maps from the dataset."""
    return _load_image(dataset, "smaps")


def load_coil_cov(
    dataset: mrd.Dataset, default: NDArray | None = None
) -> NDArray | None:
    """Load the coil covariance from the dataset."""
    return _load_image(dataset, "coil_cov")
