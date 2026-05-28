"""Small deterministic theory-side calculations used by exploratory notes.

These helpers are intentionally separate from the DESI observational pipeline.
They provide reproducible numerical bookkeeping for speculative formula audits
without mixing those assumptions into the survey-analysis code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


CODATA_ALPHA_INV = 137.035999084
ELECTRON_MASS_MEV = 0.51099895069
ELECTRON_MASS_EV = ELECTRON_MASS_MEV * 1_000_000.0
PROTON_MASS_MEV = 938.27208816
PI0_MASS_MEV = 134.9768
N_TOPOLOGICAL = 5
D_STRING = 26
F26 = 121_393
LEP_Z_WIDTH_REFERENCE_GEV = 45.0
PDG_SEQUENTIAL_CHARGED_LEPTON_LIMIT_GEV = 100.8
PLANCK_OMEGA_C_H2 = 0.120
CANONICAL_THERMAL_RELIC_CROSS_SECTION_CM3_S = 2.2e-26
GCE_REFERENCE_CROSS_SECTION_CM3_S = 2.0e-26
UNIVERSE_AGE_SECONDS = 13.787e9 * 365.25 * 24 * 3600


@dataclass(frozen=True)
class TauPrimePrediction:
    n: int
    d: int
    alpha_inv: float
    electron_mass_mev: float
    q: float
    i5_topological: int
    generation_coefficient: int
    bracket: float
    mass_ratio_to_electron: float
    mass_mev: float
    mass_gev: float
    above_lep_z_width_reference: bool
    above_sequential_charged_lepton_limit: bool


@dataclass(frozen=True)
class FourthNeutrinoPrediction:
    n: int
    d: int
    f26: int
    alpha_inv: float
    electron_mass_ev: float
    s: float
    delta_phi: float
    bracket: float
    mass_ev: float
    mass_kev: float
    mass_mev: float
    mass_gev: float
    in_sterile_dm_mass_window: bool
    above_z_width_reference: bool


@dataclass(frozen=True)
class DarkMatterClosureCheck:
    neutrino: FourthNeutrinoPrediction
    observed_omega_c_h2: float
    canonical_thermal_cross_section_cm3_s: float
    universe_age_seconds: float
    target_stability_lifetime_seconds: float
    geV_scale_cold_candidate: bool
    relic_density_calculable_from_mass_only: bool
    gamma_lifetime_calculable_from_mass_only: bool
    active_neutrino_excluded_by_z_width: bool
    conditionally_viable_if_fully_sterile: bool


@dataclass(frozen=True)
class GceResonanceAudit:
    neutrino: FourthNeutrinoPrediction
    q: float
    delta_phi: float
    phase_modulation: float
    reference_cross_section_cm3_s: float
    sigma_v_linear_cm3_s: float
    sigma_v_quadratic_cm3_s: float
    linear_fraction_of_reference: float
    quadratic_fraction_of_reference: float
    induced_theta2_toy: float
    gamma_line_energy_gev: float
    in_gce_photon_energy_band: bool
    in_common_gce_dm_mass_window: bool
    p_value: Optional[float]
    p_value_available: bool


@dataclass(frozen=True)
class Phase9HadronUpgradeAudit:
    q: float
    invariant_117: int
    pdg_pi0_mass_mev: float
    baseline_pi0_mass_mev: float
    baseline_pi0_ppm: float
    literal_117_over_137_mass_mev: float
    literal_117_over_137_ppm: float
    modulated_q_mass_mev: float
    modulated_q_ppm: float
    literal_replacement_improves: bool
    modulated_q_improves: bool


@dataclass(frozen=True)
class Phase9ProtonUpgradeAudit:
    experimental_ratio: float
    baseline_ratio: float
    baseline_ppm: float
    upgraded_ratio: float
    upgraded_ppm: float
    correction: float
    phi_gap: int
    complexity_score: int
    passes_threshold: bool


def vacuum_compression_operator(n: int = N_TOPOLOGICAL) -> float:
    """Return q = N(N-2)/(e^4 pi^3)."""

    return n * (n - 2) / (math.e**4 * math.pi**3)


def schwinger_s(alpha_inv: float = CODATA_ALPHA_INV) -> float:
    """Return s = alpha/(2 pi)."""

    return (1.0 / alpha_inv) / (2.0 * math.pi)


def delta_phi(n: int = N_TOPOLOGICAL) -> float:
    """Return cos(1/N) - cos(2/N)."""

    return math.cos(1.0 / n) - math.cos(2.0 / n)


def tau_prime_prediction(
    *,
    alpha_inv: float = CODATA_ALPHA_INV,
    electron_mass_mev: float = ELECTRON_MASS_MEV,
    n: int = N_TOPOLOGICAL,
    d: int = D_STRING,
    lep_z_width_reference_gev: float = LEP_Z_WIDTH_REFERENCE_GEV,
    charged_lepton_limit_gev: float = PDG_SEQUENTIAL_CHARGED_LEPTON_LIMIT_GEV,
) -> TauPrimePrediction:
    """Compute the proposed fourth-generation heavy-lepton mass benchmark.

    The expression is

        m_tau_prime / m_e = alpha^-1 * [D^2 + q * (D N)].

    The returned experimental flags are threshold comparisons only; they are not
    collider reinterpretations.
    """

    q = vacuum_compression_operator(n)
    i5_topological = 2 * (d - n)
    generation_coefficient = d * n
    bracket = d**2 + q * generation_coefficient
    mass_ratio = alpha_inv * bracket
    mass_mev = electron_mass_mev * mass_ratio
    mass_gev = mass_mev / 1000.0
    return TauPrimePrediction(
        n=n,
        d=d,
        alpha_inv=alpha_inv,
        electron_mass_mev=electron_mass_mev,
        q=q,
        i5_topological=i5_topological,
        generation_coefficient=generation_coefficient,
        bracket=bracket,
        mass_ratio_to_electron=mass_ratio,
        mass_mev=mass_mev,
        mass_gev=mass_gev,
        above_lep_z_width_reference=mass_gev > lep_z_width_reference_gev,
        above_sequential_charged_lepton_limit=mass_gev > charged_lepton_limit_gev,
    )


def fourth_neutrino_prediction(
    *,
    alpha_inv: float = CODATA_ALPHA_INV,
    electron_mass_ev: float = ELECTRON_MASS_EV,
    n: int = N_TOPOLOGICAL,
    d: int = D_STRING,
    f26: int = F26,
    z_width_reference_gev: float = LEP_Z_WIDTH_REFERENCE_GEV,
) -> FourthNeutrinoPrediction:
    """Compute the proposed sterile fourth-neutrino mass benchmark.

    The expression is

        m_nu_tau_prime = m_e * (F26 / alpha^-1)
                         * [1 + (DN/pi) delta_phi (1 - s)].

    The mass-window flag only checks that the value lies in a broad keV-to-GeV
    sterile-dark-matter benchmark interval. It does not validate relic density,
    lifetime, thermal history, or active-sterile mixing constraints.
    """

    s_value = schwinger_s(alpha_inv)
    delta = delta_phi(n)
    bracket = 1.0 + ((d * n) / math.pi) * delta * (1.0 - s_value)
    mass_ev = electron_mass_ev * (f26 / alpha_inv) * bracket
    mass_gev = mass_ev / 1_000_000_000.0
    return FourthNeutrinoPrediction(
        n=n,
        d=d,
        f26=f26,
        alpha_inv=alpha_inv,
        electron_mass_ev=electron_mass_ev,
        s=s_value,
        delta_phi=delta,
        bracket=bracket,
        mass_ev=mass_ev,
        mass_kev=mass_ev / 1_000.0,
        mass_mev=mass_ev / 1_000_000.0,
        mass_gev=mass_gev,
        in_sterile_dm_mass_window=1_000.0 <= mass_ev <= 1_000_000_000_000.0,
        above_z_width_reference=mass_gev > z_width_reference_gev,
    )


def dark_matter_closure_check(
    *,
    observed_omega_c_h2: float = PLANCK_OMEGA_C_H2,
    canonical_cross_section_cm3_s: float = CANONICAL_THERMAL_RELIC_CROSS_SECTION_CM3_S,
    universe_age_seconds: float = UNIVERSE_AGE_SECONDS,
    stability_margin: float = 1.0e12,
) -> DarkMatterClosureCheck:
    """Return a conservative viability audit for the fourth-neutrino benchmark.

    The check intentionally distinguishes mass-scale compatibility from model
    completion. A GeV sterile particle can be kinematically cold, but relic
    abundance and radiative lifetime require couplings, branching ratios, and a
    production history.
    """

    neutrino = fourth_neutrino_prediction()
    return DarkMatterClosureCheck(
        neutrino=neutrino,
        observed_omega_c_h2=observed_omega_c_h2,
        canonical_thermal_cross_section_cm3_s=canonical_cross_section_cm3_s,
        universe_age_seconds=universe_age_seconds,
        target_stability_lifetime_seconds=universe_age_seconds * stability_margin,
        geV_scale_cold_candidate=neutrino.mass_gev >= 1.0,
        relic_density_calculable_from_mass_only=False,
        gamma_lifetime_calculable_from_mass_only=False,
        active_neutrino_excluded_by_z_width=not neutrino.above_z_width_reference,
        conditionally_viable_if_fully_sterile=True,
    )


def gce_resonance_audit(
    *,
    reference_cross_section_cm3_s: float = GCE_REFERENCE_CROSS_SECTION_CM3_S,
) -> GceResonanceAudit:
    """Audit the fourth-neutrino benchmark against broad GCE expectations.

    Two phase-modulated cross-section estimates are reported:

    - linear rate modulation: sigma_v = sigma_ref * (q delta_phi)
    - quadratic amplitude modulation: sigma_v = sigma_ref * (q delta_phi)^2

    Neither replaces a gamma-ray likelihood fit. The p-value is intentionally
    absent unless external Fermi-LAT spectral/likelihood data are supplied.
    """

    neutrino = fourth_neutrino_prediction()
    q = vacuum_compression_operator()
    delta = delta_phi()
    phase = q * delta
    sigma_linear = reference_cross_section_cm3_s * phase
    sigma_quadratic = reference_cross_section_cm3_s * phase**2
    return GceResonanceAudit(
        neutrino=neutrino,
        q=q,
        delta_phi=delta,
        phase_modulation=phase,
        reference_cross_section_cm3_s=reference_cross_section_cm3_s,
        sigma_v_linear_cm3_s=sigma_linear,
        sigma_v_quadratic_cm3_s=sigma_quadratic,
        linear_fraction_of_reference=sigma_linear / reference_cross_section_cm3_s,
        quadratic_fraction_of_reference=sigma_quadratic / reference_cross_section_cm3_s,
        induced_theta2_toy=(phase / D_STRING) ** 2,
        gamma_line_energy_gev=neutrino.mass_gev,
        in_gce_photon_energy_band=1.0 <= neutrino.mass_gev <= 4.0,
        in_common_gce_dm_mass_window=7.0 <= neutrino.mass_gev <= 200.0,
        p_value=None,
        p_value_available=False,
    )


def _ppm(model: float, target: float) -> float:
    return (model - target) / target * 1_000_000.0


def phase9_hadron_upgrade_audit(
    *,
    pi0_mass_mev: float = PI0_MASS_MEV,
    alpha_inv: float = CODATA_ALPHA_INV,
    electron_mass_mev: float = ELECTRON_MASS_MEV,
    n: int = N_TOPOLOGICAL,
    d: int = D_STRING,
) -> Phase9HadronUpgradeAudit:
    """Audit the 117-invariant modification of the neutral-pion ansatz.

    The literal replacement requested in Phase 9, `q -> 117/137`, is evaluated
    explicitly and retained as a failed branch when it worsens the residual.
    A low-complexity modulation of the original compression operator is also
    recorded:

        q_117 = q * (1 - 2^2 / (117 * 137)).

    This keeps the original scale of `q` while allowing the discovered
    `117 = DN - 13` invariant to act as a small correction.
    """

    q = vacuum_compression_operator(n)
    invariant_117 = d * n - 13
    delta = delta_phi(n)
    bracket_tail = 1.0 + 1.0 / n + 1.0 / math.pi
    baseline_ratio = alpha_inv * (2.0 - delta - q * bracket_tail)
    literal_ratio = alpha_inv * (2.0 - delta - (invariant_117 / 137.0) * bracket_tail)
    modulated_q = q * (1.0 - 4.0 / (invariant_117 * 137.0))
    modulated_ratio = alpha_inv * (2.0 - delta - modulated_q * bracket_tail)
    baseline_mass = electron_mass_mev * baseline_ratio
    literal_mass = electron_mass_mev * literal_ratio
    modulated_mass = electron_mass_mev * modulated_ratio
    baseline_ppm = _ppm(baseline_mass, pi0_mass_mev)
    literal_ppm = _ppm(literal_mass, pi0_mass_mev)
    modulated_ppm = _ppm(modulated_mass, pi0_mass_mev)
    return Phase9HadronUpgradeAudit(
        q=q,
        invariant_117=invariant_117,
        pdg_pi0_mass_mev=pi0_mass_mev,
        baseline_pi0_mass_mev=baseline_mass,
        baseline_pi0_ppm=baseline_ppm,
        literal_117_over_137_mass_mev=literal_mass,
        literal_117_over_137_ppm=literal_ppm,
        modulated_q_mass_mev=modulated_mass,
        modulated_q_ppm=modulated_ppm,
        literal_replacement_improves=abs(literal_ppm) < abs(baseline_ppm),
        modulated_q_improves=abs(modulated_ppm) < abs(baseline_ppm),
    )


def phase9_proton_upgrade_audit(
    *,
    proton_mass_mev: float = PROTON_MASS_MEV,
    electron_mass_mev: float = ELECTRON_MASS_MEV,
    alpha_inv: float = CODATA_ALPHA_INV,
    n: int = N_TOPOLOGICAL,
    d: int = D_STRING,
    f26: int = F26,
) -> Phase9ProtonUpgradeAudit:
    """Audit a compact GF(137)-style proton-ratio correction.

    Baseline:

        K0 = [2 F26/(DN)] cos(1/N) [1 + e s].

    Phase 9 correction uses Euler's phi decomposition
    `phi(137)=136=DN+6`, so the gap `6` enters as

        6 / [137^2 (D + N + 2^2)].

    The expression is intentionally reported as an empirical low-complexity
    candidate, not as a derivation from QCD.
    """

    experimental_ratio = proton_mass_mev / electron_mass_mev
    s_value = schwinger_s(alpha_inv)
    base = 2.0 * f26 / (d * n) * math.cos(1.0 / n)
    baseline_ratio = base * (1.0 + math.e * s_value)
    phi_gap = (137 - 1) - d * n
    correction = math.e * s_value + phi_gap / (137.0**2 * (d + n + 4.0))
    upgraded_ratio = base * (1.0 + correction)
    baseline_ppm = _ppm(baseline_ratio, experimental_ratio)
    upgraded_ppm = _ppm(upgraded_ratio, experimental_ratio)
    return Phase9ProtonUpgradeAudit(
        experimental_ratio=experimental_ratio,
        baseline_ratio=baseline_ratio,
        baseline_ppm=baseline_ppm,
        upgraded_ratio=upgraded_ratio,
        upgraded_ppm=upgraded_ppm,
        correction=correction,
        phi_gap=phi_gap,
        complexity_score=11,
        passes_threshold=abs(upgraded_ppm) < 0.5,
    )
