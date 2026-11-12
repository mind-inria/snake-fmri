"""NUFFT engine: T2s_B0 model (static off-resonance and per-voxel R2*)."""

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


@pytest.mark.parametrize("df", [0.0, 40.0])
def test_t2s_b0_uniform_field_matches_t2s(df):
    """A uniform field multiplies the T2s signal by exp(+2i pi df t) (mri-nufft convention)."""
    phantom, sim_conf, traj = _setup()
    t_ms = NufftAcquisitionEngine._sample_times_ms(sim_conf, traj)
    ref = NufftAcquisitionEngine._job_model_T2s(phantom, [], sim_conf, traj, BACKEND)
    fmap = np.full(SHAPE, df, np.float32)
    out = NufftAcquisitionEngine._job_model_T2s_B0(
        phantom,
        [],
        sim_conf,
        traj,
        BACKEND,
        field_map=fmap,
        orc_interpolator={"name": "svd", "L": 40, "partial_svd": False},
    )
    expected = ref * np.exp(2j * np.pi * df * t_ms * 1e-3)[None, None]
    # the T2s_B0 model uses one (contrast-weighted) R2* per voxel: exact here, as
    # the two tissues do not overlap
    err = np.linalg.norm(out - expected) / np.linalg.norm(expected)
    assert err < 2e-2, err


def test_t2s_b0_requires_field_map():
    phantom, sim_conf, traj = _setup()
    with pytest.raises(ValueError):
        NufftAcquisitionEngine._job_model_T2s_B0(phantom, [], sim_conf, traj, BACKEND)


def test_t2s_b0_static_and_per_shot_paths_agree():
    """The chunked (static phantom) path equals the per-shot path."""
    from snake.core.phantom import DynamicData

    phantom, sim_conf, traj = _setup()
    rng = np.random.default_rng(1)
    fmap = (30 * rng.standard_normal(SHAPE)).astype(np.float32)
    kw = dict(
        field_map=fmap, orc_interpolator={"name": "svd", "L": 40, "partial_svd": False}
    )
    static = NufftAcquisitionEngine._job_model_T2s_B0(
        phantom, [], sim_conf, traj, BACKEND, **kw
    )
    # a dynamic data that does not change the phantom forces the per-shot path
    identity = DynamicData(
        "identity", data=np.zeros((1, traj.shape[0])), func=lambda ph, d, i: ph
    )
    per_shot = NufftAcquisitionEngine._job_model_T2s_B0(
        phantom, [identity], sim_conf, traj, BACKEND, **kw
    )
    np.testing.assert_allclose(
        static, per_shot, rtol=1e-3, atol=1e-6 * np.abs(static).max()
    )


def test_t2s_b0_nonuniform_field_matches_exact():
    """Against an exact per-time-sample computation (non-uniform B0 and R2*)."""
    from mrinufft import get_operator

    from snake.core.engine.utils import get_phantom_state

    # brain-like: 28 ms readout, field over ~1400 Hz (with an integer n_bins,
    # mri-nufft would quantize B0 in ~30 Hz bins here)
    phantom, sim_conf, traj = _setup(n_shots=2, n_samples=560)
    g = np.indices(SHAPE).astype(np.float32)
    fmap = (500 * np.sin(g[0] / 5) + 200 * (g[2] - 8) / 8).astype(np.float32)  # Hz
    out = NufftAcquisitionEngine._job_model_T2s_B0(
        phantom, [], sim_conf, traj, BACKEND, field_map=fmap
    )
    t_s = 1e-3 * NufftAcquisitionEngine._sample_times_ms(sim_conf, traj)
    tissues, _ = get_phantom_state(phantom, [], 0, sim_conf, aggregate=False)
    rates = 1e3 / phantom.props[:, 2]
    # exact: per tissue its own R2*, B0 per voxel, one NUFFT per time sample
    ref = np.zeros_like(out)
    for j, tj in enumerate(t_s):
        img = np.einsum("k...,k->...", tissues, np.exp(-rates * tj)) * np.exp(
            2j * np.pi * fmap * tj
        )
        op = get_operator(BACKEND)(
            traj[:, j], shape=SHAPE, n_coils=1, squeeze_dims=False
        )
        ref[:, :, j] = np.asarray(op.op(img.astype(np.complex64))).reshape(
            traj.shape[0], 1
        )
    err = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    assert err < 1e-2, err
