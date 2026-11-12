"""Acquisition engine using nufft."""

from collections.abc import Sequence
from typing import Any

import ismrmrd as mrd
import numpy as np
from mrinufft.operators import FourierOperatorBase, get_operator
from numpy.typing import NDArray

from snake.core.phantom import DynamicData, Phantom
from snake.core.phantom.utils import PropTissueEnum
from snake.core.simulation import SimConfig

from .base import BaseAcquisitionEngine
from .utils import get_phantom_state


class NufftAcquisitionEngine(BaseAcquisitionEngine):
    """Acquisition engine using nufft."""

    __engine_name__ = "NUFFT"
    __mp_mode__ = "spawn"
    model: str = "simple"
    snr: float = np.inf
    slice_2d: bool = False

    def _job_trajectories(
        self,
        dataset: mrd.Dataset,
        hdr: mrd.xsd.ismrmrdHeader,
        sim_conf: SimConfig,
        shot_idx: Sequence[int] | int,
    ) -> NDArray:
        """Get Non Cartesian trajectories from the dataset.

        Returns
        -------
        NDArray
            The trajectories.
        """
        if not isinstance(shot_idx, Sequence):
            shot_idx = [shot_idx]
        head = dataset._dataset["data"][0]["head"]
        n_samples = head["number_of_samples"]
        ndim = head["trajectory_dimensions"]
        trajectories = np.zeros((len(shot_idx), n_samples, ndim), dtype=np.float32)

        for i, s in enumerate(shot_idx):
            trajectories[i] = dataset._dataset["data"][s]["traj"].reshape(
                n_samples, ndim
            )
        return trajectories

    @staticmethod
    def _init_model_nufft(
        samples: NDArray,
        sim_conf: SimConfig,
        backend: str,
        slice_2d: bool = False,
    ) -> FourierOperatorBase:
        """Initialize the nufft operator."""
        kwargs = {}
        if slice_2d and "stacked" in backend:
            raise ValueError("Stacked NUFFT does not support 2D slice")

        if "stacked" in backend:
            kwargs["z_index"] = "auto"

        shape_ = sim_conf.shape
        if slice_2d:
            shape_ = sim_conf.shape[:-1]
        nufft = get_operator(backend)(
            samples,  # will be updated in the loop
            shape=shape_,
            n_coils=sim_conf.hardware.n_coils,
            smaps=None,
            density=False,
            squeeze_dims=False,
            **kwargs,
        )
        return nufft

    @staticmethod
    def _sample_times_ms(
        sim_conf: SimConfig,
        trajectories: NDArray,
        sample_times_ms: NDArray | None = None,
    ) -> NDArray:
        """Time of each sample of a shot relative to the echo (ms).

        Either given explicitly (e.g. ramp-sampled EPI, where the samples are not
        equally spaced in time), or from the dwell time, with the echo at the sample
        of the first shot closest to the k-space center.
        """
        n_samples = trajectories.shape[1]
        if sample_times_ms is not None:
            t = np.asarray(sample_times_ms, dtype=np.float32).ravel()
            if t.size != n_samples:
                raise ValueError(
                    f"sample_times_ms has {t.size} values for {n_samples} samples/shot."
                )
            return t
        echo_idx = int(np.argmin(np.sum(np.abs(trajectories[0]) ** 2, axis=-1)))
        return sim_conf.hardware.dwell_time_ms * (
            np.arange(n_samples, dtype=np.float32) - echo_idx
        )

    @staticmethod
    def _job_model_T2s(
        phantom: Phantom,
        dyn_datas: list[DynamicData],
        sim_conf: SimConfig,
        trajectories: NDArray,
        nufft_backend: str,
        slice_2d: bool = False,
        sample_times_ms: NDArray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Acquire k-space data with T2s relaxation effect.

        ``sample_times_ms`` optionally gives the time of each sample of a shot
        relative to the echo (default: from the dwell time).
        """
        chunk_size, n_samples, _ = trajectories.shape

        final_ksp = np.zeros(
            (chunk_size, sim_conf.hardware.n_coils, n_samples), dtype=np.complex64
        )
        # (n_tissues_true, n_samples) Filter the tissues that have NaN Values.
        nufft = NufftAcquisitionEngine._init_model_nufft(
            trajectories[0],
            sim_conf,
            backend=nufft_backend,
            slice_2d=slice_2d,
        )

        # time from the echo of each sample (the echo index used to be computed
        # without an axis, always giving 0: decay referenced to the readout start)
        t_ms = NufftAcquisitionEngine._sample_times_ms(
            sim_conf, trajectories, sample_times_ms
        )
        t2s_decay = np.exp(-t_ms[None, :] / phantom.props[:, PropTissueEnum.T2s, None])
        nufft.n_batchs = len(phantom.masks)  # number of tissues.
        for i, traj in enumerate(trajectories):
            phantom_state, smaps = get_phantom_state(
                phantom, dyn_datas, i, sim_conf, aggregate=False
            )
            if slice_2d:
                slice_loc = round((traj[0, -1] + 0.5) * sim_conf.shape[-1])
                nufft.samples = traj[:, :2]
                if smaps is not None:
                    nufft.smaps = smaps[..., slice_loc]
                phantom_state = phantom_state[:, None, ..., slice_loc]
            else:
                nufft.smaps = smaps
                phantom_state = phantom_state[:, None, ...]
                nufft.samples = traj
            ksp = nufft.op(phantom_state)
            # apply the T2s and sum over tissues
            # final_ksp[i] = np.sum(ksp * t2s_decay[:, None, :], axis=0)
            final_ksp[i] = np.einsum("kij, kj-> ij", ksp, t2s_decay)

        return final_ksp

    @staticmethod
    def _job_model_simple(
        phantom: Phantom,
        dyn_datas: list[DynamicData],
        sim_conf: SimConfig,
        trajectories: NDArray,
        nufft_backend: str,
        slice_2d: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Acquire k-space data. No T2s decay."""
        chunk_size, n_samples, _ = trajectories.shape

        final_ksp = np.zeros(
            (chunk_size, sim_conf.hardware.n_coils, n_samples), dtype=np.complex64
        )
        nufft = NufftAcquisitionEngine._init_model_nufft(
            trajectories[0],
            sim_conf,
            backend=nufft_backend,
            slice_2d=slice_2d,
        )
        # (n_tissues_true, n_samples) Filter the tissues that have NaN Values
        for i, traj in enumerate(trajectories):
            phantom_state, smaps = get_phantom_state(phantom, dyn_datas, i, sim_conf)
            nufft.n_batchs = 1  # number of tissues.
            if slice_2d:
                slice_loc = int((traj[0, -1] + 0.5) * sim_conf.shape[-1])
                nufft.samples = traj[:, :2]
                if smaps is not None:
                    nufft.smaps = smaps[..., slice_loc]
                phantom_state = phantom_state[None, ..., slice_loc]
            else:
                nufft.samples = traj
                nufft.smaps = smaps
                phantom_state = phantom_state[None, ...]
            final_ksp[i] = nufft.op(phantom_state)
        return final_ksp

    def _write_chunk_data(
        self, dataset: mrd.Dataset, chunk: Sequence[int], chunk_data: NDArray
    ) -> None:
        shot_idx = np.asarray(chunk)
        acq_chunk = dataset._dataset["data"][shot_idx]
        chunk_data = chunk_data.view(np.float32)
        acq_chunk["data"] = chunk_data.reshape(acq_chunk["data"].shape)
        dataset._dataset["data"][shot_idx] = acq_chunk
