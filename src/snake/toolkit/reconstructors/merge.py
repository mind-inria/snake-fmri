"""Conjugate Gradient descent solver."""

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm
from mrinufft.extras.optim import loss_l2_reg
from snake.mrd_utils import (
    CartesianFrameDataLoader,
    NonCartesianFrameDataLoader,
    MRDLoader
)

from .fourier import init_nufft
from .pysap import BaseReconstructor
from .pysap import RestartStrategy  

class MergeGlobalReconstructor(BaseReconstructor):
    """Merge Global Reconstructor.

    Parameters
    ----------
    max_iter : int
            Maximum number of iterations.
    tol : float
            Tolerance for the solver.
    """   

    __reconstructor_name__ = "merge-global"

    max_iter: int
    tol: float
    density_compensation: str | bool | None = False
    nufft_backend: str = "gpunufft"
    restart_strategy: str = "global"
    off_resonance_correction: bool = False


    def reconstruct(self, data_loader: MRDLoader) -> NDArray:
        """Reconstruct the data using the NUFFT operator."""
        from mrinufft import get_operator
        from scipy.optimize import minimize

        actual_dwell_s = 0.002 / 1e3     
        actual_TE_s = 29 / 1e3         
        te_pos = 0.5                 
        n_samples = 15000
        t_obs = n_samples * actual_dwell_s
        readout_time = np.arange(n_samples) * actual_dwell_s + (actual_TE_s - t_obs * te_pos)
        readout_time = np.float32(readout_time)

        field_map = np.load("/volatile/Caini/field_map_dim.npy")
        mask = np.load("/volatile/Caini/mask_dim.npy")

        nufft_operator = init_nufft(
            data_loader,
            density_compensation=self.density_compensation,
            nufft_backend=self.nufft_backend,
        )
        if self.off_resonance_correction:
            nufft_orc = nufft_operator.with_off_resonance_correction(
                b0_map=field_map,
                mask=mask,
                readout_time=readout_time,
                interpolator=dict(name='mti', L=20) #mti
            )
            nufft_operator = nufft_orc
        pg = tqdm(total=20, position=0, leave=True)
        def mixed_cb(*args, **kwargs):
            """A compound callback function, to track iterations time and convergence."""
            return [
                loss_l2_reg(*args, **kwargs),
            ]
        final_images = np.empty(
            (data_loader.n_frames, *data_loader.shape), dtype=np.complex64
        )
        x_init_0 = np.zeros(data_loader.shape, dtype=np.complex64)
        
        smaps = data_loader.get_smaps().squeeze() if data_loader.get_smaps() is not None else None
        if self.restart_strategy == "global":
            frames = list(tqdm(data_loader.iter_frames(),
                    total=data_loader.n_frames, position=0))
            trajs, datas = zip(*[(traj, data) for _, traj, data in frames])

            trajs = np.array(trajs)
            data_merge = np.array(datas)
            data_merge = np.moveaxis(data_merge, 0, 1)
            data_merge = data_merge.reshape(data_merge.shape[0], -1, data_merge.shape[-1])
            nufft_op_merge = get_operator(self.nufft_backend)(
                samples=trajs,
                shape=data_loader.shape,
                density=self.density_compensation,
                n_batchs=1,
                n_coils=data_loader.n_coils,
                smaps=smaps,   
            )
            recon_merge = nufft_op_merge.adj_op(data_merge.reshape(data_merge.shape[0], -1))
            x_init = recon_merge.copy() 
        else:
            x_init = x_init_0.copy()
        x_iter = x_init.copy()
        pbar_frames = tqdm(total=data_loader.n_frames, position=0)
        for i, traj, data in data_loader.iter_frames():
            if data_loader.slice_2d:
                nufft_operator.samples = traj.reshape(
                    data_loader.n_shots, -1, traj.shape[-1]
                )[0, :, :2]
                data = np.reshape(data, (data.shape[0], data_loader.n_shots, -1))
                for j in range(data.shape[1]):
                    recon = nufft_operator.pinv_solver(
                        kspace_data=np.ascontiguousarray(data[:,j]),
                        max_iter=self.max_iter,
                        callback=mixed_cb,
                        damp=0.0,
                        x_init=np.array(x_init[:,:,j]),
                        optim='cg',
                        progressbar=pg,
                        )
                    x_iter[:,:,j] = recon[0].copy()    
            else:
                nufft_operator.samples = traj
                recon = nufft_operator.pinv_solver(
                    kspace_data=np.ascontiguousarray(data),
                    max_iter=self.max_iter,
                    callback=mixed_cb,
                    damp=0.0,
                    x_init=np.array(x_init),
                    optim='cg',
                    progressbar=pg,
                    )
                x_iter = recon[0].copy()

            final_images[i, ...] = x_iter.squeeze()
            pbar_frames.update(1)
        return final_images

    # def _reconstruct_cartesian(self, data_loader: CartesianFrameDataLoader) -> NDArray:
    #     """Reconstruct the data for Cartesian Settings."""
    #     from mrinufft.extras.fft import (
    #         CartesianFourierOperator,
    #     )  # TODO this does not exists yet
    #     from mrinufft.extras.gradient import cg

    #     mask, data = data_loader.get_kspace_frame(0)
    #     nufft_operator = CartesianFourierOperator(mask, data_loader.shape)

    #     final_images = np.empty(
    #         (data_loader.n_frames, *data_loader.shape), dtype=np.float32
    #     )

    #     for i in tqdm(range(data_loader.n_frames)):
    #         traj, data = data_loader.get_kspace_frame(i)
    #         if data_loader.slice_2d:
    #             nufft_operator.samples = traj.reshape(
    #                 data_loader.n_shots, -1, traj.shape[-1]
    #             )[0, :, :2]
    #             data = np.reshape(data, (data.shape[0], data_loader.n_shots, -1))
    #             for j in range(data.shape[1]):
    #                 final_images[i, :, :, j] = cg(nufft_operator, data[:, j])
    #         else:
    #             final_images[i] = cg(
    #                 nufft_operator, data, num_iter=self.max_iter, tol=self.tol
    #             )
    #     return final_images