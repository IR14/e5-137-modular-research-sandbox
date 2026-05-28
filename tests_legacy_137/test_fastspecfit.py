from __future__ import annotations

import numpy as np
import pandas as pd

from cosmo_gradient.fastspecfit import (
    FastSpecQualityConfig,
    _add_mgfe_proxy,
    _dn4000_ivar_weight,
    _ivar_weight,
    _read_table_columns,
    apply_quality_mask,
    inspect_table,
    residualize_observable,
    spatial_block_permutation_null_amplitudes,
)
from cosmo_gradient.maps import SkyMap, pixel_vectors
from cosmo_gradient.theory import (
    dark_matter_closure_check,
    fourth_neutrino_prediction,
    gce_resonance_audit,
    tau_prime_prediction,
)


def test_fastspecfit_inventory_sees_population_columns_in_later_hdu(tmp_path):
    import fitsio

    path = tmp_path / "fastspec-mini.fits"
    metadata = np.array(
        [(1, 10.0, 20.0)],
        dtype=[("TARGETID", "i8"), ("RA", "f8"), ("DEC", "f8")],
    )
    fastspec = np.array(
        [(1, 1.55, 10.2, 0.9)],
        dtype=[
            ("TARGETID", "i8"),
            ("DN4000", "f8"),
            ("LOGMSTAR", "f8"),
            ("ZZSUN", "f8"),
        ],
    )
    fitsio.write(path, metadata, extname="METADATA")
    fitsio.write(path, fastspec, extname="FASTSPEC")

    record = inspect_table(path)
    frame = _read_table_columns(path, ["TARGETID", "DN4000", "LOGMSTAR", "ZZSUN"])

    assert {"DN4000", "LOGMSTAR", "ZZSUN"}.issubset(set(record.population_columns))
    assert list(frame.columns) == ["TARGETID", "DN4000", "LOGMSTAR", "ZZSUN"]
    assert frame["DN4000"].iloc[0] == 1.55
    assert frame["TARGETID"].dtype.byteorder in ("=", "|")
    assert frame["DN4000"].dtype.byteorder in ("=", "|")


def test_fastspecfit_reader_merges_quality_columns_across_hdus(tmp_path):
    import fitsio

    path = tmp_path / "fastspec-quality.fits"
    fastspec = np.array(
        [(1, 1.4, 7.0, 1.2), (2, 1.6, 8.0, 1.4)],
        dtype=[
            ("TARGETID", "i8"),
            ("DN4000_MODEL", "f8"),
            ("DN4000_IVAR", "f8"),
            ("RCHI2_CONT", "f8"),
        ],
    )
    metadata = np.array(
        [(1, 0, 40.0), (2, 4, 12.0)],
        dtype=[("TARGETID", "i8"), ("ZWARN", "i8"), ("DELTACHI2", "f8")],
    )
    fitsio.write(path, fastspec, extname="FASTSPEC")
    fitsio.write(path, metadata, extname="METADATA")

    frame = _read_table_columns(
        path,
        ["TARGETID", "DN4000_MODEL", "DN4000_IVAR", "RCHI2_CONT", "ZWARN", "DELTACHI2"],
    )

    assert list(frame.columns) == [
        "TARGETID",
        "DN4000_MODEL",
        "DN4000_IVAR",
        "RCHI2_CONT",
        "ZWARN",
        "DELTACHI2",
    ]
    assert frame.loc[frame["TARGETID"] == 1, "ZWARN"].iloc[0] == 0
    assert frame.loc[frame["TARGETID"] == 2, "DELTACHI2"].iloc[0] == 12.0


def test_apply_quality_mask_filters_and_winsorizes_dn4000_model():
    frame = pd.DataFrame(
        {
            "DN4000_MODEL": [1.0, 2.0, 100.0, 4.0, 5.0, 6.0, 7.0],
            "DN4000_IVAR": [10.0, 12.0, 8.0, 9.0, 11.0, 13.0, 0.0],
            "ZWARN": [0, 0, 0, 1, 0, 0, 0],
            "RCHI2_CONT": [1.0, 1.0, 1.0, 1.0, 2.5, 1.0, 1.0],
            "DELTACHI2": [40.0, 40.0, 40.0, 40.0, 40.0, 5.0, 40.0],
        }
    )

    filtered, stats = apply_quality_mask(
        frame,
        "DN4000_MODEL",
        FastSpecQualityConfig(winsor_lower_percentile=1.0, winsor_upper_percentile=99.0),
    )

    assert stats["n_input"] == 7
    assert stats["n_after"] == 3
    assert stats["n_removed"] == 4
    assert filtered["DN4000_MODEL"].max() < 100.0
    assert np.all(_dn4000_ivar_weight(filtered) > 0.0)


def test_apply_quality_mask_uses_line_quality_and_line_ivar_for_qso_observable():
    frame = pd.DataFrame(
        {
            "CIV_1549_EW": [10.0, 12.0, 1000.0, 16.0, 18.0, 20.0],
            "CIV_1549_EW_IVAR": [5.0, 6.0, 7.0, 0.0, 9.0, 10.0],
            "ZWARN": [0, 0, 0, 0, 0, 1],
            "RCHI2_LINE": [1.0, 1.0, 1.0, 1.0, 3.0, 1.0],
            "RCHI2_CONT": [9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            "DELTA_LINECHI2": [40.0, 40.0, 40.0, 40.0, 5.0, 40.0],
            "DELTACHI2": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    filtered, stats = apply_quality_mask(
        frame,
        "CIV_1549_EW",
        FastSpecQualityConfig(winsor_lower_percentile=1.0, winsor_upper_percentile=99.0),
    )

    assert stats["n_after"] == 4
    assert "n_fail_delta_linechi2" not in stats
    assert stats["n_fail_civ_1549_ew_ivar"] == 1
    assert filtered["CIV_1549_EW"].max() < 1000.0
    assert np.all(_ivar_weight(filtered, "CIV_1549_EW") > 0.0)


def test_add_mgfe_proxy_from_lick_indices():
    frame = pd.DataFrame(
        {
            "MGB": [4.0, 9.0],
            "FE5270": [3.0, 2.0],
            "FE5335": [2.0, 1.0],
        }
    )

    enriched = _add_mgfe_proxy(frame)

    expected = np.sqrt(frame["MGB"] * (0.72 * frame["FE5270"] + 0.28 * frame["FE5335"]))
    assert "MGFE" in enriched
    assert np.allclose(enriched["MGFE"], expected)


def test_spatial_block_permutation_null_returns_finite_amplitudes():
    nside = 2
    vectors = pixel_vectors(nside, backend="healpy")
    npix = len(vectors)
    sky_map = SkyMap(
        nside=nside,
        backend="healpy",
        data_counts=np.ones(npix),
        random_counts=np.ones(npix),
        alpha=1.0,
        delta=np.linspace(-0.1, 0.1, npix),
        valid=np.ones(npix, dtype=bool),
        pixel_vectors=vectors,
    )

    amplitudes = spatial_block_permutation_null_amplitudes(
        sky_map,
        rng=np.random.default_rng(42),
        n_permutations=8,
        block_nside=1,
        weights=np.ones(npix),
    )

    assert amplitudes.shape == (8,)
    assert np.all(np.isfinite(amplitudes))


def test_residualize_observable_removes_linear_population_trend():
    z = np.linspace(0.1, 1.0, 100)
    mass = np.linspace(9.5, 11.5, 100)
    observable = 1.2 + 0.7 * z - 0.15 * mass
    frame = pd.DataFrame({"dn4000": observable, "z": z, "mass": mass})

    residual, coefficients = residualize_observable(
        frame,
        observable_column="dn4000",
        control_columns=["z", "mass"],
        weights=np.ones(len(frame)),
    )

    assert coefficients["term"].tolist() == ["intercept", "z", "mass"]
    assert np.nanmax(np.abs(residual)) < 1e-12


def test_tau_prime_mass_prediction_invariant():
    prediction = tau_prime_prediction()

    assert prediction.i5_topological == 42
    assert prediction.generation_coefficient == 130
    assert np.isclose(prediction.q, 0.008860611874290873)
    assert np.isclose(prediction.mass_gev, 47.417730830364825)
    assert prediction.above_lep_z_width_reference
    assert not prediction.above_sequential_charged_lepton_limit


def test_fourth_neutrino_mass_prediction_invariant():
    prediction = fourth_neutrino_prediction()

    assert np.isclose(prediction.s, 0.0011614097328858596)
    assert np.isclose(prediction.delta_phi, 0.05900558383835652)
    assert np.isclose(prediction.bracket, 3.4388321110039723)
    assert np.isclose(prediction.mass_ev, 1556646342.7697074)
    assert np.isclose(prediction.mass_gev, 1.5566463427697075)
    assert prediction.in_sterile_dm_mass_window
    assert not prediction.above_z_width_reference


def test_dark_matter_closure_requires_model_assumptions():
    check = dark_matter_closure_check()

    assert np.isclose(check.observed_omega_c_h2, 0.120)
    assert np.isclose(check.canonical_thermal_cross_section_cm3_s, 2.2e-26)
    assert check.geV_scale_cold_candidate
    assert not check.relic_density_calculable_from_mass_only
    assert not check.gamma_lifetime_calculable_from_mass_only
    assert check.active_neutrino_excluded_by_z_width
    assert check.conditionally_viable_if_fully_sterile


def test_gce_resonance_audit_rejects_simple_phase_modulation():
    audit = gce_resonance_audit()

    assert np.isclose(audit.phase_modulation, 0.0005228255768076074)
    assert np.isclose(audit.sigma_v_linear_cm3_s, 1.0456511536152148e-29)
    assert np.isclose(audit.sigma_v_quadratic_cm3_s, 5.4669316752841484e-33)
    assert audit.linear_fraction_of_reference < 1e-3
    assert audit.quadratic_fraction_of_reference < 1e-6
    assert audit.in_gce_photon_energy_band
    assert not audit.in_common_gce_dm_mass_window
    assert not audit.p_value_available
    assert audit.p_value is None
