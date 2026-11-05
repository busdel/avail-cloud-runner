"""Unit tests for the avail-omics core (Paper 2 software evidence)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avail_omics.preprocess import (CLRTransformer, LogCenterTransformer,  # noqa: E402
                                    preproc, apply_preproc)
from avail_omics.learners import (base_predictions_loo,  # noqa: E402
                                  build_meta_features)
from avail_omics.generic import load_generic  # noqa: E402


def test_clr_handles_zeros_and_centers_rows():
    X = np.array([[0.5, 0.3, 0.2], [0.0, 0.6, 0.4]])
    t = CLRTransformer().fit(X)
    Z = t.transform(X)
    assert Z.shape == X.shape
    assert np.all(np.isfinite(Z))
    assert np.allclose(Z.mean(axis=1), 0, atol=1e-6)


def test_logcenter_finite_on_nonnegative():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Z = LogCenterTransformer().fit_transform(X)
    assert Z.shape == X.shape
    assert np.all(np.isfinite(Z))


def test_preproc_apply_roundtrip_shapes():
    rng = np.random.default_rng(0)
    X = rng.gamma(2, 1, size=(40, 30))
    y = np.r_[np.zeros(20), np.ones(20)].astype(int)
    C = rng.normal(size=(40, 3))
    sel, sel2, res, sc, Xtr = preproc(X, y, C, True)
    Xte = apply_preproc(sel, sel2, res, sc, X[:5], C[:5])
    assert Xte.shape[0] == 5
    assert np.all(np.isfinite(Xte))


def test_representation_flags_and_masks():
    piv = {"s1": {"a": np.array([0.7, 0.2, 0.1])},
           "s2": {"a": np.array([0.1, 0.2, 0.7])}}
    ids = ["s1", "s2", "s3"]
    label = {"s1": 0, "s2": 2, "s3": 1}
    X, y = build_meta_features(piv, ids, label, n_mods=1,
                               mod_order=["a"], K=3)
    assert X.shape == (3, 4)
    assert np.allclose(X[0, :3], [0.7, 0.2, 0.1])
    assert X[0, 3] == 1.0
    assert np.allclose(X[2, :3], 0.0)
    assert X[2, 3] == 0.0
    assert y.tolist() == [0, 2, 1]


def test_base_predictions_loo_leakfree_smoke():
    rng = np.random.default_rng(42)
    n, K = 24, 2
    ids = [f"s{i}" for i in range(n)]
    lab = {s: (i % 2) for i, s in enumerate(ids)}
    Xa = {s: rng.normal(size=8) + (2.0 if lab[s] else 0.0) for s in ids}
    Xb = {s: rng.normal(size=8) for s in ids[:12]}
    conf = {s: np.zeros(3) for s in ids}
    piv = base_predictions_loo({"a": Xa, "b": Xb}, ids, lab, conf, K,
                               micro=set(), n_estimators=50)
    assert set(piv.keys()) == set(ids)
    for s in ids:
        assert abs(piv[s]["a"].sum() - 1.0) < 1e-6
    assert "b" in piv["s1"]
    assert "b" not in piv["s20"]


def test_generic_layout_loading(tmp_path):
    d = tmp_path
    (d / "features").mkdir()
    pd.DataFrame({"SampleID": ["x1", "x2", "x3", "x4"],
                  "Group": ["A", "A", "B", "B"]}).to_csv(
        d / "metadata.tsv", sep="\t", index=False)
    pd.DataFrame({"SampleID": ["x1", "x2", "x3"],
                  "f1": [0.1, 0.2, 0.3], "f2": [1.0, 2.0, 3.0]}).to_csv(
        d / "features" / "mod1.tsv", sep="\t", index=False)
    pd.DataFrame({"SampleID": ["x1", "x3", "x4"],
                  "g1": [5.0, 6.0, 7.0]}).to_csv(
        d / "features" / "mod2.tsv", sep="\t", index=False)
    (d / "compositional.txt").write_text("mod1\n")
    group, conf, mods, names = load_generic(d)
    assert group == {"x1": "A", "x2": "A", "x3": "B", "x4": "B"}
    assert set(mods.keys()) == {"mod1", "mod2"}
    assert names["mod1"] == ["f1", "f2"]
    assert len(conf["x1"]) == 3


def test_generic_layout_requires_metadata(tmp_path):
    (tmp_path / "features").mkdir()
    try:
        load_generic(tmp_path)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass
