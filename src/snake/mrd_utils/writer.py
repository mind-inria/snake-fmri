"""Export data to mrd format."""

from __future__ import annotations
from snake.core.engine.base import GenericPath

import logging
import os
from typing import TYPE_CHECKING

import ismrmrd as mrd
import numpy as np
from hydra_callbacks import PerfLogger
from mrinufft.trajectories.utils import Gammas
from numpy.typing import NDArray

from snake._version import __version__ as version

from .utils import get_waveform_id, obj2b64encode

if TYPE_CHECKING:
    from snake.core.handlers import AbstractHandler, HandlerList
    from snake.core.phantom import DynamicData, Phantom
    from snake.core.sampling import BaseSampler
    from snake.core.simulation import SimConfig

log = logging.getLogger(__name__)


def get_mrd_header(
    sim_conf: SimConfig, engine: str, model: str, slice_2d: bool
) -> mrd.xsd.ismrmrdHeader:
    """Create a MRD Header for snake-fmri data."""
    H = mrd.xsd.ismrmrdHeader()
    # Experimental conditions
    H.experimentalConditions = mrd.xsd.experimentalConditionsType(
        H1resonanceFrequency_Hz=int(Gammas.H * 1e3),
    )

    # Acquisition System Information
    H.acquisitionSystemInformation = mrd.xsd.acquisitionSystemInformationType(
        deviceID="SNAKE",
        systemVendor="SNAKE",
        systemModel=f"{version}-{engine}",
        deviceSerialNumber=42,
        systemFieldStrength_T=sim_conf.hardware.field,
        receiverChannels=sim_conf.hardware.n_coils,
    )

    # Encoding
    # FOV computation
    input_fov = mrd.xsd.fieldOfViewMm(*sim_conf.fov_mm)
    input_matrix = mrd.xsd.matrixSizeType(*sim_conf.shape)

    output_fov = mrd.xsd.fieldOfViewMm(*sim_conf.fov_mm)
    output_matrix = mrd.xsd.matrixSizeType(*sim_conf.shape)

    # FIXME: update the encoding in acquisition writer.
    encoding = mrd.xsd.encodingType(
        encodedSpace=mrd.xsd.encodingSpaceType(input_matrix, input_fov),
        reconSpace=mrd.xsd.encodingSpaceType(output_matrix, output_fov),
        trajectory=mrd.xsd.trajectoryType.OTHER,
        encodingLimits=mrd.xsd.encodingLimitsType(
            kspace_encoding_step_0=-1,
            kspace_encoding_step_1=-1,
            kspace_encoding_step_2=-1,
            repetition=-1,
        ),
    )
    H.encoding.append(encoding)

    # Sequence Parameters
    H.sequenceParameters = mrd.xsd.sequenceParametersType(
        TR=sim_conf.seq.TR,
        TE=sim_conf.seq.TE,
        flipAngle_deg=sim_conf.seq.FA,
    )
    H.userParameters = mrd.xsd.userParametersType(
        userParameterDouble=[
            mrd.xsd.userParameterDoubleType(name=name, value=value)
            for name, value in [
                ("gmax", sim_conf.hardware.gmax),
                ("smax", sim_conf.hardware.smax),
                ("dwell_time_ms", sim_conf.hardware.dwell_time_ms),
                ("rng_seed", sim_conf.rng_seed),
                ("max_sim_time", sim_conf.max_sim_time),
                ("TR_eff", sim_conf.seq.TR_eff),
            ]
        ],
        userParameterString=[
            mrd.xsd.userParameterStringType(name=name, value=value)
            for name, value in [
                ("engine_model", model),
                ("slice_2d", str(slice_2d)),
                # FIXME: better erialization ?
                ("fov_config", str(sim_conf.fov.__repr__())),
            ]
        ],
    )

    return H


def add_dynamic_mrd(
    dataset: mrd.Dataset, dynamic: DynamicData, sim_conf: SimConfig
) -> mrd.Dataset:
    """Add the dynamic data to the dataset."""
    waveform_id = get_waveform_id(dynamic.name)

    # add the type to the header.
    hdr = mrd.xsd.CreateFromDocument(dataset.read_xml_header())
    hdr.waveformInformation.append(
        mrd.xsd.waveformInformationType(
            waveformName=dynamic.name,
            waveformType=waveform_id,
            userParameters=mrd.xsd.userParametersType(
                userParameterBase64=[
                    mrd.xsd.userParameterBase64Type(
                        dynamic.name, obj2b64encode(dynamic.func)
                    )
                ],
                userParameterString=[
                    mrd.xsd.userParameterStringType(
                        "domain", "kspace" if dynamic.in_kspace else "image"
                    )
                ],
            ),
        )
    )
    dataset.write_xml_header(mrd.xsd.ToXML(hdr))

    if dynamic.data.ndim == 1:
        channels = 1
        nsamples = dynamic.data.shape[0]
    elif dynamic.data.ndim == 2:
        channels, nsamples = dynamic.data.shape
    else:
        raise ValueError(f"Invalid data shape: {dynamic.data.shape}")
    dataset.append_waveform(
        mrd.Waveform(
            mrd.WaveformHeader(
                waveform_id=waveform_id,
                number_of_samples=nsamples,
                channels=channels,
                sample_time_us=sim_conf.sim_tr_ms * 1000,
            ),
            data=np.float32(dynamic.data).view(np.uint32),
        )
    )
    return dataset


def add_coil_cov_mrd(
    dataset: mrd.Dataset,
    sim_conf: SimConfig,
    coil_cov: NDArray | None = None,
) -> mrd.Dataset:
    """Add the Smaps to the dataset."""
    n_coils = sim_conf.hardware.n_coils
    if coil_cov is None:
        return dataset
    elif coil_cov.shape != (n_coils, n_coils):
        raise ValueError(
            f"Incompatible coil_cov shape {coil_cov.shape} != {(n_coils, n_coils)} "
        )
    dataset.append_image(
        "coil_cov",
        mrd.image.Image(
            head=mrd.image.ImageHeader(
                matrixSize=mrd.xsd.matrixSizeType(*coil_cov.shape),
                fieldOfView_mm=mrd.xsd.fieldOfViewMm(*coil_cov.shape),
                channels=1,
                acquisition_time_stamp=0,
            ),
            data=coil_cov,
        ),
    )
    return dataset


def make_base_mrd(
    filename: GenericPath,
    sampler: BaseSampler,
    phantom: Phantom,
    sim_conf: SimConfig,
    handlers: list[AbstractHandler] | HandlerList | None = None,
    coil_cov: NDArray | None = None,
    model: str = "simple",
    slice_2d: bool = False,
) -> mrd.Dataset:
    """
    Create a base `.mrd` file from the simulation configurations.

    Parameters
    ----------
    filename : os.PathLike
        The output filename.
    sampler : BaseSampler
        The sampling pattern generator.
    phantom : Phantom
        The phantom object.
    sim_conf : SimConfig
        The simulation configurations.
    dynamic_data : list[DynamicData], optional
        The dynamic data, by default None
    smaps : NDArray, optional
        The coil sensitivity maps, by default None
    coil_covar : NDArray, optional
        The coil covariance matrix, by default None
    """
    # Apply the handlers and get the dynamic data!!
    if handlers is None:
        handlers = []
    for h in handlers:
        phantom = h.get_static(phantom, sim_conf)
    dynamic_data = [h.get_dynamic(phantom, sim_conf) for h in handlers]

    try:
        log.warning("Existing %s it will be overwritten", filename)
        os.remove(filename)
    except Exception as e:
        log.error(e)
        pass
    dataset = mrd.Dataset(filename, "dataset", create_if_needed=True)
    dataset.write_xml_header(
        mrd.xsd.ToXML(get_mrd_header(sim_conf, sampler.__engine__, model, slice_2d))
    )
    with PerfLogger(logger=log, name="acq"):
        sampler.add_all_acq_mrd(dataset, sim_conf)

    with PerfLogger(logger=log, name="phantom"):
        phantom.to_mrd_dataset(dataset)

    with PerfLogger(logger=log, name="dynamic"):
        if dynamic_data is not None:
            for dyn in dynamic_data:
                if dyn is not None:
                    add_dynamic_mrd(dataset, dyn, sim_conf)

    with PerfLogger(logger=log, name="coil_cov"):
        if sim_conf.hardware.n_coils > 1 and coil_cov is not None:
            add_coil_cov_mrd(dataset, sim_conf, coil_cov)

    dataset._file.flush()
    dataset.close()
    return dataset
