"""Conjugate Gradient descent solver."""

import cupy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from snake.mrd_utils import (
    CartesianFrameDataLoader,
    NonCartesianFrameDataLoader,
)

from .fourier import init_nufft
from .pysap import ZeroFilledReconstructor
from .pysap import RestartStrategy  

class ConjugateGradientReconstructor(ZeroFilledReconstructor):
    """Conjugate Gradient descent solver.

    Parameters
    ----------
    max_iter : int
            Maximum number of iterations.
    tol : float
            Tolerance for the solver.
    """

    __reconstructor_name__ = "cg"

    max_iter: int
    tol: float
    density_compensation: str | bool | None = False
    nufft_backend: str = "cufinufft"
    restart_strategy: RestartStrategy = RestartStrategy.REFINE
    traj_2d: bool = False

    def _reconstruct_nufft(self, data_loader: NonCartesianFrameDataLoader) -> NDArray:
        """Reconstruct the data using the NUFFT operator."""
        from mrinufft.extras.gradient import cg
    
        from scipy.optimize import minimize
        nufft_operator = init_nufft(
            data_loader,
            density_compensation=self.density_compensation,
            nufft_backend=self.nufft_backend,
            traj_2d=self.traj_2d,
        )

        final_images = np.empty(
            (data_loader.n_frames, *data_loader.shape), dtype=np.complex64
        )
        x_init_0 = np.zeros(data_loader.shape, dtype=np.complex64)

        # traj, data = data_loader.get_kspace_frame(0, traj_2d=True)
        x_init = x_init_0.copy()
        x_iter = x_init.copy()
        pbar_frames = tqdm(total=data_loader.n_frames, position=0)
        for i, traj, data in data_loader.iter_frames(traj_2d=self.traj_2d):
            if data_loader.slice_2d:
                nufft_operator.samples = traj.reshape(
                    data_loader.n_shots, -1, traj.shape[-1]
                )[0, :, :2]
                data = np.reshape(data, (data.shape[0], data_loader.n_shots, -1))
                for j in range(data.shape[1]):
                    x_iter[:,:,j] = cg(
                        nufft_operator, 
                        data[:,j], 
                        x_init=x_init[:,:,j], 
                        num_iter=self.max_iter, 
                        tol=self.tol,
                        compute_backend="cupy"  
                    )
            else:
                nufft_operator.samples = traj.reshape(
                    data_loader.n_shots, -1, traj.shape[-1]
                ) # fix: update traj when 3D
                x_iter = cg(
                    nufft_operator, 
                    data, 
                    x_init=np.array(x_init), 
                    num_iter=self.max_iter, 
                    tol=self.tol,
                    compute_backend="cupy"  
          
                )
            x_iter = x_iter.copy() 
            x_init = (
                    x_iter.copy()
                    if self.restart_strategy != RestartStrategy.COLD
                    else x_init.copy()
            )
            final_images[i, ...] = x_iter
            pbar_frames.update(1)
        if self.restart_strategy != RestartStrategy.REFINE:
            return final_images.get()  # Removed .get() to match previous logic
        pbar_frames.reset()
        x_init = x_iter.copy() 
        for i, traj, data in data_loader.iter_frames(traj_2d=self.traj_2d):
            if data_loader.slice_2d:
                nufft_operator.samples = traj.reshape(
                    data_loader.n_shots, -1, traj.shape[-1]
                )[0, :, :2]
                data = np.reshape(data, (data.shape[0], data_loader.n_shots, -1))
                for j in range(data.shape[1]):
                    x_iter[:,:,j] = cg(
                        nufft_operator, 
                        data[:,j], 
                        x_init=x_init[:,:,j], 
                        num_iter=self.max_iter, 
                        tol=self.tol,
                        compute_backend="cupy"  
                    )    
            else:
                nufft_operator.samples = traj.reshape(
                    data_loader.n_shots, -1, traj.shape[-1]
                )
                x_iter = cg(
                    nufft_operator, 
                    data, 
                    x_init=np.array(x_init),
                    num_iter=self.max_iter, 
                    tol=self.tol,
                    compute_backend="cupy"  
           
                )
                #loss_list.append(loss)

            final_images[i, ...] = x_iter
            pbar_frames.update(1)
        return final_images.get()

    def _reconstruct_cartesian(self, data_loader: CartesianFrameDataLoader) -> NDArray:
        """Reconstruct the data for Cartesian Settings."""
        from mrinufft.extras.fft import (
            CartesianFourierOperator,
        )  # TODO this does not exists yet
        from mrinufft.extras.gradient import cg

        mask, data = data_loader.get_kspace_frame(0)
        nufft_operator = CartesianFourierOperator(mask, data_loader.shape)

        final_images = np.empty(
            (data_loader.n_frames, *data_loader.shape), dtype=np.float32
        )

        for i in tqdm(range(data_loader.n_frames)):
            traj, data = data_loader.get_kspace_frame(i)
            if data_loader.slice_2d:
                nufft_operator.samples = traj.reshape(
                    data_loader.n_shots, -1, traj.shape[-1]
                )[0, :, :2]
                data = np.reshape(data, (data.shape[0], data_loader.n_shots, -1))
                for j in range(data.shape[1]):
                    final_images[i, :, :, j] = cg(nufft_operator, data[:, j])
            else:
                final_images[i] = cg(
                    nufft_operator, data, num_iter=self.max_iter, tol=self.tol
                )
        return final_images