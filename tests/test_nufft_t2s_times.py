"""NUFFT engine: T2s decay referenced to the echo, explicit sample times."""

import numpy as np
import pytest

from snake.core.engine.nufft import NufftAcquisitionEngine
from snake.core.phantom import Phantom
from snake.core.simulation import FOVConfig, GreConfig, HardwareConfig, SimConfig

BACKEND = "finufft"
pytest.importorskip(BACKEND)

SHAPE = (24, 24, 16)


def _setup(n_shots=3, n_samples=200):
    masks = np.zeros((2, *SHAPE), np.float32)
    masks[0, 6:18, 6:18, 4:12] = 1.0
    masks[1, 9:15, 9:15, 6:10] = 1.0
    masks[0] -= masks[1]
    # T1, T2, T2s, rho, chi
    props = np.array([[1500, 80, 30, 1.0, 0], [1000, 60, 12, 0.8, 0]], np.float32)
    phantom = Phantom(
        "test", masks, np.array(["a", "b"]), props, affine=np.eye(4, dtype=np.float32)
    )
    sim_conf = SimConfig(
        seq=GreConfig(TR=50, TE=20, FA=15),
        hardware=HardwareConfig(n_coils=1, dwell_time_ms=0.05),
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


def test_t2s_decay_referenced_to_echo():
    """The decay must be 1 at the echo (sample closest to k=0), not at sample 0."""
    phantom, sim_conf, traj = _setup()
    t = NufftAcquisitionEngine._sample_times_ms(sim_conf, traj)
    echo = np.argmin(np.sum(np.abs(traj[0]) ** 2, axis=-1))
    assert t[echo] == 0 and echo > 0
    np.testing.assert_allclose(np.diff(t), sim_conf.hardware.dwell_time_ms, rtol=1e-5)


def test_explicit_sample_times():
    phantom, sim_conf, traj = _setup()
    times = np.linspace(-3, 7, traj.shape[1])
    np.testing.assert_allclose(
        NufftAcquisitionEngine._sample_times_ms(sim_conf, traj, times), times, rtol=1e-6
    )
    with pytest.raises(ValueError):
        NufftAcquisitionEngine._sample_times_ms(sim_conf, traj, times[:-1])
