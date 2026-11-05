"""Unit tests for the imaging utilities (no heavy deps required)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avail_omics.imaging import sanitize_ts  # noqa: E402


def test_sanitize_ts_zeroes_constant_and_nan():
    ts = np.array([[1.0, 0.0, 2.0],
                   [2.0, 0.0, np.nan],
                   [3.0, 0.0, 4.0]])
    out, nbad = sanitize_ts(ts)
    assert nbad == 1
    assert np.all(np.isfinite(out))
    assert np.all(out[:, 1] == 0.0)
    assert out[0, 0] == 1.0


def test_sanitize_ts_clean_input_untouched():
    rng = np.random.default_rng(0)
    ts = rng.normal(size=(50, 10))
    out, nbad = sanitize_ts(ts)
    assert nbad == 0
    assert np.allclose(out, ts)
