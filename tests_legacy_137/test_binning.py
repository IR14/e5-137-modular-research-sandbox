import numpy as np
import pandas as pd
import pytest

from cosmo_gradient.binning import assign_redshift_bins, make_redshift_bins, subset_by_redshift_bin


def test_make_bins_requires_monotonic_edges():
    with pytest.raises(ValueError):
        make_redshift_bins([0.0, 1.0, 0.5])


def test_assign_redshift_bins_includes_final_edge():
    edges = [0.4, 0.6, 0.8, 1.0]
    z = np.array([0.39, 0.4, 0.59, 0.6, 0.99, 1.0, 1.01])

    assigned = assign_redshift_bins(z, edges)

    np.testing.assert_array_equal(assigned, [-1, 0, 0, 1, 2, 2, -1])


def test_subset_by_redshift_bin_respects_final_bin_flag():
    frame = pd.DataFrame({"z": [0.4, 0.5, 0.6], "value": [1, 2, 3]})
    first, second = make_redshift_bins([0.4, 0.5, 0.6])

    assert subset_by_redshift_bin(frame, first)["value"].tolist() == [1]
    assert subset_by_redshift_bin(frame, second, final_bin=True)["value"].tolist() == [2, 3]
