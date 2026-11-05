"""Regression tests for the 1.0.2 review fixes."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avail_omics.generic import detect_compositional, load_generic  # noqa: E402
from avail_omics.radiomics import match_sid  # noqa: E402


def test_compositional_heuristic_and_override(tmp_path):
    ids = ["a", "b"]
    comp = {i: np.array([0.4, 0.6]) for i in ids}       # rowsum 1.0
    cont = {i: np.array([350.0, 40.0]) for i in ids}    # clinical-like
    mods = {"taxa": comp, "clin": cont}
    assert detect_compositional(None, mods, ids, verbose=False) == {"taxa"}
    (tmp_path / "compositional.txt").write_text("clin\n")
    assert detect_compositional(tmp_path, mods, ids, verbose=False) == {"clin"}


def test_match_sid_exact_token(tmp_path):
    ids = {"S1", "S10"}
    assert match_sid(tmp_path / "S10_T1.nii.gz", ids) == "S10"
    assert match_sid(tmp_path / "S1_T1.nii.gz", ids) == "S1"
    assert match_sid(tmp_path / "S1" / "t1_mni.nii.gz", ids) == "S1"
    assert match_sid(tmp_path / "other" / "t1.nii.gz", ids) is None


def test_confounder_parse_tolerates_junk(tmp_path):
    (tmp_path / "features").mkdir()
    pd.DataFrame({"SampleID": ["x1", "x2"], "Group": ["A", "B"],
                  "Age": ["65", "N/A"]}).to_csv(
        tmp_path / "metadata.tsv", sep="\t", index=False)
    pd.DataFrame({"SampleID": ["x1", "x2"], "f1": [1.0, 2.0]}).to_csv(
        tmp_path / "features" / "m1.tsv", sep="\t", index=False)
    group, conf, mods, names = load_generic(tmp_path)
    assert conf["x1"][0] == 65.0
    assert np.isnan(conf["x2"][0])
