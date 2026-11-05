"""Paper-protocol connectivity feature tests."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_fc_protocol_features_shape_and_keys():
    rng = np.random.default_rng(0)
    ts = rng.normal(size=(120, 100))
    from avail_omics.imaging import fc_protocol_features, sanitize_ts
    ts, _ = sanitize_ts(ts)
    feats = fc_protocol_features(ts)
    assert len(feats) == 342
    assert sum(k.startswith("full_") for k in feats) == 171
    assert sum(k.startswith("part_") for k in feats) == 171
    assert all(np.isfinite(list(feats.values())))


def test_fc_protocol_features_guard_constant_parcel():
    rng = np.random.default_rng(1)
    ts = rng.normal(size=(120, 100))
    ts[:, 7] = 0.0
    from avail_omics.imaging import fc_protocol_features, sanitize_ts
    ts, nbad = sanitize_ts(ts)
    assert nbad == 1
    feats = fc_protocol_features(ts)
    assert all(np.isfinite(list(feats.values())))
