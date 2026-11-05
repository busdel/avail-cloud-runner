"""Unit tests for the radiomics module (light parts; no heavy extraction)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_radiomics_available_flag():
    from avail_omics.radiomics import radiomics_available
    assert radiomics_available() in (True, False)


def test_network_labels_load():
    from avail_omics.radiomics import load_network_labels
    labs = load_network_labels()
    # "17-network" Yeo naming yields 18 named groups here (SomMotB splits
    # into auditory + rest), 100 parcels total.
    assert len(labs) == 18
    assert sum(len(v) for v in labs.values()) == 100


def test_extract_requires_extras_cleanly():
    from avail_omics.radiomics import extract_subject, radiomics_available
    if radiomics_available():
        pytest.skip("radiomics installed; covered by integration runs")
    with pytest.raises(RuntimeError, match="radiomics"):
        extract_subject(t1="nonexistent.nii.gz")
