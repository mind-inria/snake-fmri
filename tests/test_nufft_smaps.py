"""NUFFT engine: operators are created with their coil maps."""

import numpy as np
import pytest
from mrinufft import get_operator
from mrinufft.operators import check_backend

from snake.core.engine.nufft import NufftAcquisitionEngine
from snake.core.phantom import Phantom
from snake.core.simulation import FOVConfig, GreConfig, HardwareConfig, SimConfig

SHAPE = (24, 24, 16)
N_COILS = 3


def _setup(n_shots=3, n_samples=200):
    masks = np.zeros((1, *SHAPE), np.float32)
    masks[0, 6:18, 6:18, 4:12] = 1.0
    g = np.indices(SHAPE).astype(np.float32)
    smaps = np.stack(
        [np.exp(1j * 0.2 * c * g[0]) * (1 + 0.02 * c * g[1]) for c in range(N_COILS)]
    ).astype(np.complex64)
    props = np.array([[1500, 80, 30, 1.0, 0]], np.float32)
    phantom = Phantom(
        "test",
        masks,
        np.array(["a"]),
        props,
        smaps=smaps,
        affine=np.eye(4, dtype=np.float32),
    )
    sim_conf = SimConfig(
        seq=GreConfig(TR=50, TE=20, FA=15),
        hardware=HardwareConfig(n_coils=N_COILS, dwell_time_ms=0.05),
        fov=FOVConfig(size=SHAPE, offset=(0, 0, 0), res_mm=(1, 1, 1)),
    )
    sim_conf.seq.TR_eff = sim_conf.seq.TR
    t = np.linspace(-1, 1, n_samples)
    traj = np.stack(
        [
            np.stack(
                [0.4 * t * np.cos(5 * t + s), 0.4 * t * np.sin(5 * t + s), 0.3 * t], -1
            )
            for s in range(n_shots)
        ]
    ).astype(np.float32)
    return phantom, sim_conf, traj


@pytest.mark.parametrize("backend", ["finufft", "gpunufft"])
def test_simple_model_with_coil_maps(backend):
    """Multi-coil acquisition matches a direct NUFFT with the same coil maps."""
    if not check_backend(backend):
        pytest.skip(f"{backend} not available")
    phantom, sim_conf, traj = _setup()
    out = NufftAcquisitionEngine._job_model_simple(phantom, [], sim_conf, traj, backend)
    image = phantom.contrast(sim_conf=sim_conf, resample=False)
    for i, tr in enumerate(traj):
        op = get_operator(backend)(
            tr, shape=SHAPE, n_coils=N_COILS, smaps=phantom.smaps, squeeze_dims=False
        )
        ref = np.asarray(op.op(image.astype(np.complex64)[None])).reshape(N_COILS, -1)
        np.testing.assert_allclose(
            out[i], ref, rtol=1e-3, atol=1e-4 * np.abs(ref).max()
        )
