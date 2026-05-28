import numpy as np
import pandas as pd

from cosmo_gradient.coords import radec_to_unit
from cosmo_gradient.statistics import (
    add_multiple_testing_corrections,
    compare_to_reference_axes,
    overlapping_redshift_axis_separations,
    region_axis_consistency,
)


def test_multiple_testing_corrections_are_monotone():
    results = pd.DataFrame({"null_p_value": [0.20, 0.01, 0.04]})

    corrected = add_multiple_testing_corrections(results)

    assert corrected["null_p_value"].tolist() == [0.01, 0.04, 0.20]
    np.testing.assert_allclose(corrected["bonferroni_p"], [0.03, 0.12, 0.60])
    assert corrected["bh_fdr_p"].is_monotonic_increasing


def test_overlapping_redshift_axis_separations_require_overlap():
    first = radec_to_unit([0.0], [0.0])[0]
    second = radec_to_unit([90.0], [0.0])[0]
    third = radec_to_unit([180.0], [0.0])[0]
    results = pd.DataFrame(
        [
            {
                "tracer": "LRG",
                "region": "NGC",
                "z_min": 0.4,
                "z_max": 0.8,
                "null_p_value": 0.2,
                "vector_x": first[0],
                "vector_y": first[1],
                "vector_z": first[2],
            },
            {
                "tracer": "ELG",
                "region": "NGC",
                "z_min": 0.6,
                "z_max": 1.0,
                "null_p_value": 0.3,
                "vector_x": second[0],
                "vector_y": second[1],
                "vector_z": second[2],
            },
            {
                "tracer": "QSO",
                "region": "NGC",
                "z_min": 1.1,
                "z_max": 1.4,
                "null_p_value": 0.4,
                "vector_x": third[0],
                "vector_y": third[1],
                "vector_z": third[2],
            },
        ]
    )

    separations = overlapping_redshift_axis_separations(results)

    assert len(separations) == 1
    assert np.isclose(separations.loc[0, "z_overlap"], 0.2)
    assert np.isclose(separations.loc[0, "axis_separation_deg"], 90.0)


def test_reference_axis_comparison_uses_axis_antipode_symmetry():
    opposite = radec_to_unit([180.0], [0.0])[0]
    results = pd.DataFrame(
        [
            {
                "tracer": "LRG",
                "region": "NGC",
                "z_min": 0.4,
                "z_max": 0.6,
                "ra_deg": 180.0,
                "dec_deg": 0.0,
                "null_p_value": 0.2,
                "amplitude": 0.1,
                "vector_x": opposite[0],
                "vector_y": opposite[1],
                "vector_z": opposite[2],
            }
        ]
    )

    compared = compare_to_reference_axes(results, {"test_axis": (0.0, 0.0, "test")})

    assert len(compared) == 1
    assert np.isclose(compared.loc[0, "axis_separation_deg"], 0.0)


def test_region_axis_consistency_matches_same_tracer_bin():
    first = radec_to_unit([0.0], [0.0])[0]
    second = radec_to_unit([60.0], [0.0])[0]
    results = pd.DataFrame(
        [
            {
                "tracer": "ELG",
                "region": "NGC",
                "z_min": 0.8,
                "z_max": 1.1,
                "null_p_value": 0.4,
                "amplitude": 0.02,
                "vector_x": first[0],
                "vector_y": first[1],
                "vector_z": first[2],
            },
            {
                "tracer": "ELG",
                "region": "SGC",
                "z_min": 0.8,
                "z_max": 1.1,
                "null_p_value": 0.1,
                "amplitude": 0.08,
                "vector_x": second[0],
                "vector_y": second[1],
                "vector_z": second[2],
            },
        ]
    )

    consistency = region_axis_consistency(results)

    assert len(consistency) == 1
    assert np.isclose(consistency.loc[0, "axis_separation_deg"], 60.0)
