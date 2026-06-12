"""Temporal TV PDHG reconstructor.

This wraps the reconstruction flow prototyped in
``/volatile/Caini/stimulate/snake/notebook/temporal_tv_test.ipynb`` as a
SNAKE-fMRI reconstructor so it can be selected from Hydra YAML files.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import field
from types import ModuleType
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from snake.mrd_utils import (
    CartesianFrameDataLoader,
    MRDLoader,
    NonCartesianFrameDataLoader,
)

from .base import BaseReconstructor


def _load_module_from_path(module_name: str, path: str | Path) -> ModuleType:
    """Load a Python module from an explicit file path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot load {module_name!r}; missing file: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {module_name!r} at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TemporalTVReconstructor(BaseReconstructor):
    """Joint frame reconstruction with spatial and temporal TV regularisation.

    The implementation mirrors ``temporal_tv_test.ipynb``:

    1. collect all selected frames from the MRD file,
    2. build one frame-wise NUFFT physics object,
    3. create either frame-wise or global shared adjoint initialization,
    4. solve the whole time block with ``PDHG_TV``.
    """

    __reconstructor_name__ = "temporal-tv"

    pdhg_tv_module_path: str = (
        "/volatile/Caini/stimulate/snake/notebook/pdhg_tv.py"
    )
    temporal_tv_utils_module_path: str = (
        "/volatile/Caini/stimulate/snake/notebook/temporal_tv_utils.py"
    )

    frame_indices: list[int] | None = None
    nufft_backend: str = "auto"
    density: bool | str | None = False
    shot_dim: bool = True
    power_iters: int = 8
    init_mode: str = "global_shared_adjoint"
    global_init_density: bool | str | None = "pipe"
    global_init_frame_indices: list[int] | None = None

    lambda_spatial: float = 1e-5
    lambda_temporal: float = 5e-4
    temporal_penalty: str = "l1"
    tv_dims_groups: list[list[int]] = field(
        default_factory=lambda: [[-3, -2, -1], [0]]
    )
    tv_lambdas: list[float] | None = None
    tv_penalties: list[str] | None = None

    max_iter: int = 30
    stopping_criterion: float = 1e-3
    relaxation_param: float = 1.0
    init_l2_lambda: float = 1e-1
    init_l1_lambda: float = 1e-4
    l2_norm_lambda: float = 0.0
    spatial_lambda_reg: float = 0.0

    return_complex: bool = False
    device: str = "auto"

    def __str__(self) -> str:
        """Return a compact identifier used by the CLI output filenames."""
        return (
            f"{self.__reconstructor_name__}"
            f"-{self.nufft_backend}"
            f"-ls{self.lambda_spatial:g}"
            f"-lt{self.lambda_temporal:g}"
            f"-{self.temporal_penalty}"
            f"-it{self.max_iter}"
        )

    def reconstruct(self, data_loader: MRDLoader) -> NDArray:
        """Reconstruct the MRD data as a temporal TV block."""
        if isinstance(data_loader, CartesianFrameDataLoader):
            raise NotImplementedError(
                "Temporal TV currently expects non-Cartesian data."
            )
        if not isinstance(data_loader, NonCartesianFrameDataLoader):
            raise ValueError(f"Unsupported dataloader type: {type(data_loader)!r}")

        import torch

        pdhg_tv_module = _load_module_from_path(
            "snake_temporal_tv_pdhg", self.pdhg_tv_module_path
        )
        tv_utils_module = _load_module_from_path(
            "snake_temporal_tv_utils", self.temporal_tv_utils_module_path
        )

        backend = self._resolve_backend(torch)
        device = self._resolve_device(torch)
        frame_indices = self._resolve_frame_indices(data_loader)
        tv_lambdas = self.tv_lambdas or (self.lambda_spatial, self.lambda_temporal)
        tv_penalties = self.tv_penalties or ("l1", self.temporal_penalty)

        block, physics, y_torch, x0_torch = tv_utils_module.prepare_temporal_pdhg_block(
            data_loader=data_loader,
            frame_indices=frame_indices,
            backend=backend,
            density=self.density,
            shot_dim=self.shot_dim,
            power_iters=self.power_iters,
            init_mode=self.init_mode,
            global_init_density=self.global_init_density,
            global_init_frame_indices=self.global_init_frame_indices,
            **self.nufft_kwargs,
        )

        y_torch = y_torch.to(device)
        x0_torch = x0_torch.to(device)

        solver = pdhg_tv_module.PDHG_TV(
            lambda_reg=1.0,
            max_iter=self.max_iter,
            lipschitz=block.lipschitz,
            data_fidelity=tv_utils_module.ComplexL2DataFidelity(),
            stopping_criterion=self.stopping_criterion,
            relaxation_param=self.relaxation_param,
            tv_dims_groups=self._as_tuple_groups(self.tv_dims_groups),
            tv_lambdas=tuple(float(value) for value in tv_lambdas),
            tv_penalties=tuple(tv_penalties),
            init_l2_lambda=self.init_l2_lambda,
            init_l1_lambda=self.init_l1_lambda,
            l2_norm_lambda=self.l2_norm_lambda,
            spatial_lambda_reg=self.spatial_lambda_reg,
        )

        recon_torch = solver(
            y_torch,
            physics,
            init=x0_torch,
            compute_metrics=False,
        )
        recon_complex = recon_torch.squeeze(1).detach().cpu().numpy()
        if self.return_complex:
            return recon_complex.astype(np.complex64, copy=False)
        return np.abs(recon_complex).astype(np.float32, copy=False)

    def _resolve_backend(self, torch_module: Any) -> str:
        if self.nufft_backend != "auto":
            return self.nufft_backend
        return "cufinufft" if torch_module.cuda.is_available() else "finufft"

    def _resolve_device(self, torch_module: Any) -> Any:
        if self.device != "auto":
            return torch_module.device(self.device)
        device = "cuda" if torch_module.cuda.is_available() else "cpu"
        return torch_module.device(device)

    def _resolve_frame_indices(
        self,
        data_loader: NonCartesianFrameDataLoader,
    ) -> np.ndarray:
        if self.frame_indices is None:
            return np.arange(data_loader.n_frames, dtype=int)
        return np.asarray(self.frame_indices, dtype=int)

    @staticmethod
    def _as_tuple_groups(groups: Any) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(group) for group in groups)
