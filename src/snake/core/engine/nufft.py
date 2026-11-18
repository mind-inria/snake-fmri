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
        smaps: NDArray | None = None,
    ) -> FourierOperatorBase:
        """Initialize the nufft operator.

        The coil maps are given at creation: some backends (gpunufft) cannot
        receive coil maps later when created without them.
        """
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
            smaps=smaps,
            density=False,
            squeeze_dims=False,
            **kwargs,
        )
        return nufft

    @staticmethod
    def _update_nufft(
        nufft: FourierOperatorBase | None,
        samples: NDArray,
        smaps: NDArray | None,
        sim_conf: SimConfig,
        backend: str,
        slice_2d: bool = False,
    ) -> FourierOperatorBase:
        """Create the operator at the first shot, else update its samples and maps."""
        if nufft is None:
            return NufftAcquisitionEngine._init_model_nufft(
                samples, sim_conf, backend=backend, slice_2d=slice_2d, smaps=smaps
            )
        nufft.samples = samples
        if smaps is not None:
            nufft.smaps = smaps
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
        nufft = None  # created at the first shot, with its coil maps

        # time from the echo of each sample (the echo index used to be computed
        # without an axis, always giving 0: decay referenced to the readout start)
        t_ms = NufftAcquisitionEngine._sample_times_ms(
            sim_conf, trajectories, sample_times_ms
        )
        t2s_decay = np.exp(-t_ms[None, :] / phantom.props[:, PropTissueEnum.T2s, None])
        for i, traj in enumerate(trajectories):
            phantom_state, smaps = get_phantom_state(
                phantom, dyn_datas, i, sim_conf, aggregate=False
            )
            if slice_2d:
                slice_loc = round((traj[0, -1] + 0.5) * sim_conf.shape[-1])
                samples = traj[:, :2]
                if smaps is not None:
                    smaps = smaps[..., slice_loc]
                phantom_state = phantom_state[:, None, ..., slice_loc]
            else:
                samples = traj
                phantom_state = phantom_state[:, None, ...]
            nufft = NufftAcquisitionEngine._update_nufft(
                nufft, samples, smaps, sim_conf, nufft_backend, slice_2d
            )
            nufft.n_batchs = len(phantom.masks)  # number of tissues.
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
        nufft = None  # created at the first shot, with its coil maps
        for i, traj in enumerate(trajectories):
            phantom_state, smaps = get_phantom_state(phantom, dyn_datas, i, sim_conf)
            if slice_2d:
                slice_loc = int((traj[0, -1] + 0.5) * sim_conf.shape[-1])
                samples = traj[:, :2]
                if smaps is not None:
                    smaps = smaps[..., slice_loc]
                phantom_state = phantom_state[None, ..., slice_loc]
            else:
                samples = traj
                phantom_state = phantom_state[None, ...]
            nufft = NufftAcquisitionEngine._update_nufft(
                nufft, samples, smaps, sim_conf, nufft_backend, slice_2d
            )
            nufft.n_batchs = 1  # number of tissues.
            final_ksp[i] = nufft.op(phantom_state)
        return final_ksp

    @staticmethod
    def _job_model_T2s_B0(
        phantom: Phantom,
        dyn_datas: list[DynamicData],
        sim_conf: SimConfig,
        trajectories: NDArray,
        nufft_backend: str,
        slice_2d: bool = False,
        field_map: NDArray | None = None,
        sample_times_ms: NDArray | None = None,
        orc_interpolator: str | dict | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Acquire k-space data with T2s relaxation and static B0 off-resonance.

        The signal of each voxel evolves as exp(-R2*(r) t + 2i pi dB0(r) t), with t
        the time from the echo, using mri-nufft's MRIFourierCorrected (time-segmented
        model). The sign convention is mri-nufft's, so that data simulated here are
        consistent with off-resonance corrected reconstructions using mri-nufft.

        Parameters
        ----------
        field_map: NDArray
            Off-resonance map in Hz, on the simulation grid (``sim_conf.shape``).
        sample_times_ms: NDArray, optional
            Time of each sample of a shot relative to the echo, in ms.
            Default: from the dwell time (see ``_sample_times_ms``).
        orc_interpolator: str or dict, optional
            Factorization of the field (see ``mrinufft.extras.get_orc_factorization``),
            e.g. ``{"name": "svd", "L": 100}``. Default: SVD with 16 x 1024 histogram
            bins (R2* x B0). Give ``n_bins`` explicitly: with an integer, mri-nufft
            splits it between the two axes (e.g. 21 x 49), which quantizes B0 in
            ~30 Hz bins for a typical brain field map.

        Notes
        -----
        R2* is per voxel: the contrast-weighted mean of the tissues' 1/T2s (a
        mono-exponential approximation in voxels mixing tissues). The factorization
        is computed once per chunk from the first shot's phantom state: dynamic data
        changing T2s (e.g. BOLD) only change the amplitudes, not the decay rates.
        """
        from mrinufft.operators.off_resonance import MRIFourierCorrected

        if slice_2d:
            raise NotImplementedError("T2s_B0 model is only implemented in 3D.")
        if field_map is None:
            raise ValueError("The T2s_B0 model needs a field_map (Hz).")
        if orc_interpolator is None or orc_interpolator == "svd":
            orc_interpolator = {"name": "svd", "n_bins": (16, 1024), "partial_svd": False}
        field_map = np.asarray(field_map, dtype=np.float32)
        if field_map.shape != tuple(sim_conf.shape):
            raise ValueError(
                f"field_map shape {field_map.shape} != simulation shape {sim_conf.shape}"
            )
        chunk_size, n_samples, _ = trajectories.shape
        t_s = 1e-3 * NufftAcquisitionEngine._sample_times_ms(
            sim_conf, trajectories, sample_times_ms
        )
        final_ksp = np.zeros(
            (chunk_size, sim_conf.hardware.n_coils, n_samples), dtype=np.complex64
        )
        rates = 1e3 / phantom.props[:, PropTissueEnum.T2s]  # R2* of each tissue (Hz)

        def r2star_map(tissues: NDArray) -> NDArray:
            weights = np.abs(tissues)
            r2s = np.einsum("k...,k->...", weights, rates) / np.maximum(
                weights.sum(axis=0), np.finfo(np.float32).tiny
            )
            return r2s.astype(np.float32)

        if not dyn_datas:
            # static phantom: all the shots of the chunk at once (one factorization
            # and one operator instead of one per shot)
            tissues, smaps = get_phantom_state(
                phantom, dyn_datas, 0, sim_conf, aggregate=False
            )
            nufft = NufftAcquisitionEngine._init_model_nufft(
                trajectories.reshape(-1, trajectories.shape[-1]),
                sim_conf,
                backend=nufft_backend,
                smaps=smaps,
            )
            orc_op = MRIFourierCorrected(
                nufft,
                b0_map=field_map,
                readout_time=t_s,  # per shot: the operator repeats it over the shots
                r2star_map=r2star_map(tissues),
                interpolator=orc_interpolator,
            )
            ksp = orc_op.op(np.sum(tissues, axis=0)[None].astype(np.complex64))
            ksp = np.asarray(ksp).reshape(sim_conf.hardware.n_coils, chunk_size, n_samples)
            return np.ascontiguousarray(np.moveaxis(ksp, 1, 0))

        nufft, orc_op = None, None
        for i, traj in enumerate(trajectories):
            tissues, smaps = get_phantom_state(
                phantom, dyn_datas, i, sim_conf, aggregate=False
            )
            image = np.sum(tissues, axis=0)
            nufft = NufftAcquisitionEngine._update_nufft(
                nufft, traj, smaps, sim_conf, nufft_backend
            )
            nufft.n_batchs = 1
            if orc_op is None:
                orc_op = MRIFourierCorrected(
                    nufft,
                    b0_map=field_map,
                    readout_time=t_s,
                    r2star_map=r2star_map(tissues),
                    interpolator=orc_interpolator,
                )
            ksp = orc_op.op(image[None].astype(np.complex64))
            final_ksp[i] = np.asarray(ksp).reshape(sim_conf.hardware.n_coils, n_samples)
        return final_ksp

    def _write_chunk_data(
        self, dataset: mrd.Dataset, chunk: Sequence[int], chunk_data: NDArray
    ) -> None:
        shot_idx = np.asarray(chunk)
        acq_chunk = dataset._dataset["data"][shot_idx]
        chunk_data = chunk_data.view(np.float32)
        acq_chunk["data"] = chunk_data.reshape(acq_chunk["data"].shape)
        dataset._dataset["data"][shot_idx] = acq_chunk
