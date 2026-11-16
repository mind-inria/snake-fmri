"""Field drift: a uniform off-resonance changing linearly over the shots."""

import numpy as np

from snake.core.engine.utils import apply_field_drift


def test_apply_field_drift_matches_analytic_phase():
    n_shots, n_coils, n_samples = 5, 2, 7
    ksp = np.ones((n_shots, n_coils, n_samples), dtype=np.complex64)
    t_ms = np.linspace(-3, 3, n_samples)
    te_ms = 18.0
    out = apply_field_drift(ksp, np.arange(n_shots), n_shots, t_ms, te_ms, (0.0, 30.0))
    assert out.shape == ksp.shape and out.dtype == ksp.dtype
    df = np.linspace(0, 30, n_shots)
    expected = np.exp(2j * np.pi * df[:, None] * 1e-3 * (te_ms + t_ms)[None])
    np.testing.assert_allclose(out[:, 1], expected, atol=1e-5)
    # first shot unchanged, magnitude preserved
    np.testing.assert_allclose(out[0], ksp[0], atol=1e-6)
    np.testing.assert_allclose(np.abs(out), 1, atol=1e-6)


def test_apply_field_drift_uses_acquisition_index():
    """A chunk of shots gets the offsets of its own position in the acquisition."""
    ksp = np.ones((2, 1, 3), dtype=np.complex64)
    t_ms = np.zeros(3)
    out = apply_field_drift(ksp, np.array([2, 4]), 5, t_ms, 10.0, (0.0, 40.0))
    # shots 2 and 4 of 5: 20 Hz and 40 Hz, phase 2 pi df TE
    np.testing.assert_allclose(np.angle(out[:, 0, 0]), np.angle(np.exp(2j * np.pi * np.array([20.0, 40.0]) * 0.01)), atol=1e-5)
