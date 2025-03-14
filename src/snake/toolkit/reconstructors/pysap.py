"""Reconstructors using PySAP-fMRI toolbox."""

import copy
import os

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Local imports
from snake.mrd_utils import (
    CartesianFrameDataLoader,
    MRDLoader,
    NonCartesianFrameDataLoader,
    SliceDataloader,
)
from snake.core.parallel import (
    ArrayProps,
    SharedMemoryManager,
    array_from_shm,
    array_to_shm,
)
from snake._meta import NoCaseEnum
from snake.core.simulation import SimConfig
from tqdm.auto import tqdm

from .base import BaseReconstructor
from .fourier import ifft, init_nufft


def _reconstruct_cartesian_frame(
    filename: os.PathLike,
    idx: int,
    smaps_props: ArrayProps | None,
    final_props: ArrayProps,
) -> int:
    """Reconstruct a single frame."""
    with (
        array_from_shm(final_props) as final_images,
        CartesianFrameDataLoader(filename) as data_loader,
    ):
        mask, kspace = data_loader.get_kspace_frame(idx)
        if data_loader.slice_2d:
            axes = (-3, -2)
        else:
            axes = tuple(range(len(data_loader.shape), 0, -1))
        adj_data = ifft(kspace, axis=axes)
        if smaps_props is not None and data_loader.n_coils > 1:
            with array_from_shm(smaps_props) as smaps_info:
                smaps = smaps_info[0]
                adj_data_smaps_comb = abs(
                    np.sum(adj_data * smaps.conj(), axis=0)
                    / np.sum(smaps * smaps.conj(), axis=0)
                ).astype(np.float32, copy=False)
        elif data_loader.n_coils > 1:
            adj_data_smaps_comb = np.sqrt(np.sum(abs(adj_data) ** 2, axis=0)).astype(
                np.float32, copy=False
            )
        else:
            adj_data_smaps_comb = abs(adj_data).astype(np.float32, copy=False)

        final_images[0][idx] = adj_data_smaps_comb
    return idx


class ZeroFilledReconstructor(BaseReconstructor):
    """Zero Filled Reconstructor."""

    __reconstructor_name__ = "adjoint"
    n_jobs: int = 10
    nufft_backend: str = "gpunufft"
    density_compensation: str | bool = "pipe"

    def setup(self, sim_conf: SimConfig) -> None:
        """Initialize Reconstructor."""
        pass

    def reconstruct(
        self,
        data_loader: MRDLoader,
        sim_conf: SimConfig = None,
    ) -> NDArray:
        """Reconstruct data with zero-filled method."""
        with data_loader:
            if isinstance(data_loader, SliceDataloader):
                reconstruct_method = self._reconstruct_nufft if data_loader.is_non_cartesian else self._reconstruct_cartesian
            elif isinstance(data_loader, CartesianFrameDataLoader | NonCartesianFrameDataLoader):
                reconstruct_method = self._reconstruct_cartesian if isinstance(data_loader, CartesianFrameDataLoader) else self._reconstruct_nufft
            else:
                raise ValueError("Unknown dataloader")
            return reconstruct_method(data_loader)

    def _reconstruct_cartesian(
        self,
        data_loader: CartesianFrameDataLoader,
    ) -> NDArray:
        """Reconstruct Cartesian data with zero-filled method."""
        smaps = data_loader.get_smaps()
        if smaps is None and data_loader.n_coils > 1:
            raise NotImplementedError("Missing coil combine code.")

        final_images = np.zeros(
            (data_loader.n_frames, *data_loader.shape), dtype=np.float32
        )

        with (
            SharedMemoryManager() as smm,
            ProcessPoolExecutor(self.n_jobs) as executor,
            tqdm(total=data_loader.n_frames) as pbar,
        ):
            smaps_props = None
            if smaps is not None:
                smaps_props, smaps_shared, smaps_sm = array_to_shm(smaps, smm)
            final_props, final_shared, final_sm = array_to_shm(final_images, smm)

            futures = {
                executor.submit(
                    _reconstruct_cartesian_frame,
                    data_loader._filename,
                    idx,
                    smaps_props,
                    final_props,
                ): idx
                for idx in range(data_loader.n_frames)
            }
            for future in as_completed(futures):
                future.result()
                pbar.update(1)
            final_images[:] = final_shared.copy()
            final_sm.close()
            if smaps_props is not None:
                smaps_sm.close()
            smm.shutdown()
        return final_images

    def _reconstruct_nufft(
        self,
        data_loader: NonCartesianFrameDataLoader,
    ) -> NDArray:
        """Reconstruct data with nufft method."""
        nufft_operator = init_nufft(
            data_loader, self.nufft_backend, self.density_compensation
        )
        final_images = np.empty(
            (data_loader.n_frames, *data_loader.shape), dtype=np.float32
        )
        smaps = data_loader.get_smaps()

        for i in tqdm(range(data_loader.n_frames)):
            traj, data = data_loader.get_kspace_frame(i)
            #fix: density compensation should update every frame when rotating(writer: dyn_traj?)
            if smaps is not None and data_loader.slice_2d:
                nufft_operator.smaps = smaps[...,i % data_loader.frame.n_shots]
            nufft_operator.samples = traj
            final_images[i] = abs(nufft_operator.adj_op(data))
        if data_loader.slice_2d:
            final_images = np.moveaxis(final_images.reshape(
                (data_loader.frame.n_frames, -1, *final_images.shape[-2:])
            ), 1, -1)
        return final_images


class RestartStrategy(NoCaseEnum):
    """Restart strategies for the reconstruction."""

    WARM = "warm"
    COLD = "cold"
    REFINE = "refine"


class SequentialReconstructor(BaseReconstructor):
    """Use a sequential Reconstruction.

    Parameters
    ----------
    max_iter_frame
        Number of iteration to allow per frame.
    optimizer
        Optimizer name, available are pogm and fista.
    threshold
        Threshold value for the wavelet regularisation.
    """

    __reconstructor_name__ = "sequential"

    max_iter_per_frame: int = 15
    optimizer: str = "pogm"
    wavelet: str = "db4"
    threshold: float | str = "sure"
    nufft_backend: str = "gpunufft"
    density_compensation: None | str | bool = "pipe"
    restart_strategy: RestartStrategy = RestartStrategy.WARM
    compute_backend: str = "cupy"

    def __str__(self) -> str:
        """Return a string representation of the reconstructor."""
        return f"{self.__reconstructor_name__}-{self.restart_strategy}"

    def setup(self, sim_conf: SimConfig = None, shape: tuple[int] = None) -> None:
        """Set up the reconstructor."""
        from fmri.operators.weighted import AutoWeightedSparseThreshold
        from modopt.opt.linear import Identity
        from modopt.opt.linear.wavelet import WaveletTransform
        from modopt.opt.proximity import SparseThreshold
        from modopt.base.backend import get_backend

        if sim_conf is None and shape is None:
            raise ValueError("SimConfig or shape must be provided.")
        elif sim_conf is not None and shape is not None:
            raise ValueError("SimConfig and shape cannot be provided at the same time.")
        elif sim_conf is not None:
            shape = sim_conf.shape

        self.space_linear_op = WaveletTransform(
            self.wavelet,
            shape=shape,
            level=3,
            mode="zero",
            compute_backend=self.compute_backend,
        )
        xp, _ = get_backend(self.compute_backend)
        _ = self.space_linear_op.op(xp.zeros(shape, dtype=np.complex64))

        if not hasattr(self, "space_prox_op"):
            if self.threshold == "sure":
                self.space_prox_op = AutoWeightedSparseThreshold(
                    self.space_linear_op.coeffs_shape,
                    linear=None,
                    threshold_estimation="hybrid-sure",
                    threshold_scaler=0.6,
                )
            else:
                self.threshold = float(self.threshold)
                self.space_prox_op = SparseThreshold(
                    linear=Identity(), weights=self.threshold
                )

    def reconstruct(self, data_loader: MRDLoader) -> np.ndarray:
        """Reconstruct with Sequential."""
        shape = data_loader.shape
        self.setup(shape=shape)
        from fmri.operators.gradient import (
            GradAnalysis,
            GradSynthesis,
        )
        from modopt.base.backend import get_backend
        from mrinufft import get_operator
        from mrinufft.density import pipe

        xp, _ = get_backend(self.compute_backend)

        traj, data = data_loader.get_kspace_frame(0)

        smaps = data_loader.get_smaps()

        density_compensation = self.density_compensation
        if (
            isinstance(self.density_compensation, str)
            and "first" in self.density_compensation
        ):
            density_compensation = False

        kwargs = {}
        if "stacked" in self.nufft_backend:
            kwargs["z_index"] = "auto"
        if self.nufft_backend == "cufinufft":
            kwargs["smaps_cached"] = True

        fourier_op = get_operator(
            self.nufft_backend,
            samples=traj,
            shape=data_loader.shape,
            n_coils=data_loader.n_coils,
            smaps=smaps.squeeze() if data_loader.slice_2d else smaps,
            # smaps=xp.array(smaps) if smaps is not None else None,
            density=density_compensation,
            squeeze_dims=True,
            **kwargs,
        )

        final_estimate = np.zeros(
            (data_loader.n_frames, *data_loader.shape), dtype=np.float32
        )
        grad_kwargs = dict(
            fourier_op=fourier_op,
            input_data_writeable=True,
            dtype=np.complex64,
            compute_backend=self.compute_backend,
            num_check_lips=0,
            verbose=0,
        )
        if self.optimizer in ["fista"]:
            grad_op = GradAnalysis(**grad_kwargs)
        if self.optimizer in ["pogm"]:
            grad_op = GradSynthesis(linear_op=self.space_linear_op, **grad_kwargs)

        x_init = xp.zeros(shape, dtype=np.complex64)
        if (
            isinstance(self.density_compensation, str)
            and "first" in self.density_compensation
        ):
            density_comp_vector = pipe(traj, shape, self.nufft_backend)
            x_init = fourier_op.adj_op(xp.array(data * density_comp_vector, copy=False))
        else:
            x_init = fourier_op.adj_op(xp.array(data, copy=False))

        pbar_frames = tqdm(total=data_loader.n_frames, position=0)
        pbar_iter = tqdm(total=self.max_iter_per_frame, position=1)
        for i, traj, data in data_loader.iter_frames():
            grad_op.fourier_op.samples = traj
            spec_rad = grad_op.fourier_op.get_lipschitz_cst(20)
            grad_op._obs_data = xp.array(data)
            grad_op.spec_rad = spec_rad
            grad_op.inv_spec_rad = 1 / spec_rad
            x_iter = self._reconstruct_frame(
                grad_op,
                x_init,
                n_iter=self.max_iter_per_frame,
                progbar=pbar_iter,
            )
            # Prepare for next iteration and save results
            x_init = (
                x_iter.copy()
                if self.restart_strategy != RestartStrategy.COLD
                else x_init.copy()
            )
            if self.compute_backend == "cupy":
                final_estimate[i, ...] = abs(x_iter).get()  # type: ignore
            else:
                final_estimate[i, ...] = abs(x_iter)

            pbar_frames.update(1)
        if self.restart_strategy != RestartStrategy.REFINE:
            if data_loader.slice_2d:
                final_estimate = np.moveaxis(final_estimate.reshape(
                    (data_loader.frame.n_frames, -1, *final_estimate.shape[-2:])
                ), 1, -1)
            return final_estimate
        # else, we do a second pass on the data using the last iteration as a slotion.
        pbar_frames.reset()
        pbar_iter.reset()
        x_init = x_iter.copy()  # last iteration results.
        for i, traj, data in data_loader.iter_frames():
            grad_op.fourier_op.samples = traj
            spec_rad = grad_op.fourier_op.get_lipschitz_cst()
            grad_op._obs_data = xp.array(data)
            grad_op.spec_rad = spec_rad
            grad_op.inv_spec_rad = 1 / spec_rad
            x_iter = self._reconstruct_frame(
                grad_op,
                x_init,
                n_iter=self.max_iter_per_frame,
                progbar=pbar_iter,
            )
            if self.compute_backend == "cupy":
                final_estimate[i, ...] = abs(x_iter).get()  # type: ignore
            else:
                final_estimate[i, ...] = abs(x_iter)
            pbar_frames.update(1)
        if data_loader.slice_2d:
            final_estimate = np.moveaxis(final_estimate.reshape(
                (data_loader.frame.n_frames, -1, *final_estimate.shape[-2:])
            ), 1, -1) 
        return final_estimate

    def _reconstruct_frame(
        self,
        grad_op: Any,
        x_init: NDArray,
        n_iter: int = 15,
        progbar: tqdm | None = None,
    ) -> NDArray:
        from fmri.reconstructors.utils import initialize_opt
        from modopt.base.backend import get_backend

        xp, _ = get_backend(self.compute_backend)
        # only recreate gradient if the trajectory change.
        # reset Smaps and optimizer if required.
        opt = initialize_opt(
            opt_name=self.optimizer,
            grad_op=grad_op,
            linear_op=copy.deepcopy(self.space_linear_op),
            prox_op=copy.deepcopy(self.space_prox_op),
            x_init=x_init.get(),
            synthesis_init=False,
            metric_kwargs={},
            compute_backend=self.compute_backend,
            opt_kwargs=dict(
                verbose=0,
                cost="auto",
            ),
        )
        # if no reset, the internal state is kept.
        if progbar is not None:
            progbar.reset(total=n_iter)
        opt.iterate(max_iter=n_iter, progbar=progbar)
        if hasattr(grad_op, "linear_op"):
            img = grad_op.linear_op.adj_op(opt.x_final)
        else:
            img = opt.x_final
        return img
