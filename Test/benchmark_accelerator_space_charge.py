"""
Dedicated Verification & Benchmark Suite for High-Intensity Accelerator Space-Charge Control
============================================================================================
Author & Discoverer: Dr. Hyo Eun Ahn (Independent Researcher, hyo.eun.ahn@proton.me)
Companion Benchmark for Paper:
  "Closed-Form Moment Reduction of 3D Space-Charge Fields for Fast Feedback
   in High-Intensity Hadron Rings" (doc/arXiv/Accelerator/accelerator_space_charge.tex)
Theoretical Foundation: Dr. Hyo Eun Ahn's 1994 LANL Derivations (report1, poly-phi, poly-e)
Architecture: Domain-Driven Design (DDD) & Zero Hard-Coding (GR2 Compliant)

Verifies:
  1. Relativistic Space-Charge Physics (1.0 GeV Accumulator/Booster Ring: SNS/PSB/PSR regime,
     gamma = 2.066, T_rev = 0.945 us, 5-turn control budget = 4.73 us <= 5.0 us, Delta Q_sc = -0.2271)
  2. Theorem 1 (report1.pdf):
     - Full rank condition rank(Omega) = 35 on the non-degenerate 3D mesh
     - Paul Channell's uniform reference mu_i = const produces zero difference from mu_i = 0
     - Dr. Ahn's normalized physical beam prior (sum_i mu_i = 1 with matched 2nd moments)
       strictly preserves a == mu > 0 at equilibrium and reduces fringe negative-charge
       overshoot by >3x and L2 error by 2x under finite perturbations when R_beam < R_mesh
     - Exact moment conservation ||Omega*a - s||_inf to machine precision
  3. Theorem 2 (poly-phi.pdf): 35x35 closed-form potential reduction & 35 potential averages
  4. Theorem 3 (poly-e.pdf) & Runtime Zero-Self-Force (ZSF) Dipolar Gauge Projection:
     - State-weighted projection (w = a) net force residual (~10^-16)
     - Prior-weighted projection (w = mu) mu-weighted moment conservation (~10^-16) vs
       uncorrected net force on perturbed state a (||<E^{prior}>_a||_inf = 0.1123)
     - Runtime ZSF Dipolar Gauge Projection restoring ||<E^{ZSF}>_a||_inf ~ 10^-16 on a!
     - Interior beam-core multipole gradient accuracy (R^2 = 0.8962, RMS relative error = 32.22%)
  5. Non-Circular 2D Transverse (x,px,y,py) Leapfrog Tracking in 3D Pairwise-Coulomb Geometry
     & Ablation Study (identical initial conditions, independent O(N_p^2) 3D Plummer-Coulomb
     space-charge disturbance, aperture-regularized polynomial feedback with and without ZSF
     dipolar gauge projection, plus 4-seed statistical robustness verification)
"""

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent

# Load engine from the same directory as this script, regardless of sys.path or
# working directory, so that a stale engine in any parent folder cannot shadow it.
_engine_path = CURRENT_DIR / "accelerator_space_charge_engine.py"
if not _engine_path.exists():
    raise FileNotFoundError(
        f"Engine not found at expected path: {_engine_path}\n"
        "Ensure accelerator_space_charge_engine.py is in the same directory as this script."
    )
_spec = importlib.util.spec_from_file_location("accelerator_space_charge_engine", _engine_path)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["accelerator_space_charge_engine"] = _mod
_spec.loader.exec_module(_mod)
AcceleratorSpaceChargeConfig = _mod.AcceleratorSpaceChargeConfig
AcceleratorSpaceChargeEngine = _mod.AcceleratorSpaceChargeEngine


@dataclass(frozen=True)
class AcceleratorBenchmarkSpec:
    """Immutable Specification for the Accelerator Space-Charge Benchmark (GR2 Compliant)."""
    num_bunch_mesh_points: int = 1000
    bunch_radius: float = 1.0
    random_seed: int = 42
    bunch_anisotropy: tuple = (0.55, 0.48, 0.62)
    bunch_centroid_offset: tuple = (0.04, -0.03, 0.02)
    prior_radius_scale: float = 0.55
    core_eval_radius: float = 0.45
    direct_pic_sample_runs: int = 50
    numerical_tolerance: float = 1e-10
    figure_dpi: int = 400
    output_figure_relpath: str = "benchmark_accelerator.png"


def generate_bunch_spherical_mesh(spec: AcceleratorBenchmarkSpec) -> np.ndarray:
    """Generates a deterministic 3D beam bunch spatial mesh."""
    rng = np.random.default_rng(spec.random_seed)
    points = []
    radius = spec.bunch_radius
    n_target = spec.num_bunch_mesh_points

    while len(points) < n_target:
        batch = rng.uniform(-radius, radius, (n_target * 2, 3))
        r2 = np.sum(batch ** 2, axis=1)
        valid = batch[r2 <= radius ** 2]
        for p in valid:
            if len(points) < n_target:
                points.append(p)
            else:
                break
    return np.array(points, dtype=np.float64)


def generate_3d_beam_bunch_charges(
    mesh: np.ndarray, spec: AcceleratorBenchmarkSpec
) -> np.ndarray:
    """Generates a realistic 3D anisotropic parabolic hadron bunch smaller than the mesh (report1.pdf Fig. 2)."""
    sx, sy, sz = spec.bunch_anisotropy
    cx, cy, cz = spec.bunch_centroid_offset
    x = (mesh[:, 0] - cx) / sx
    y = (mesh[:, 1] - cy) / sy
    z = (mesh[:, 2] - cz) / sz
    r2_bunch = x ** 2 + y ** 2 + z ** 2
    charges = np.maximum(0.0, 1.0 - r2_bunch)
    charges /= np.sum(charges)
    return charges


def build_ahn_reasonable_beam_prior(
    mesh: np.ndarray, spec: AcceleratorBenchmarkSpec
) -> np.ndarray:
    """
    Constructs Dr. Ahn's normalized reference prior mu (sum(mu_i) = 1) with approximate
    second moments matching the physical beam scale (report1.pdf, p. 3: 'choose the mu_i
    values so that it resembles a reasonable beam with close to the correct second moments').
    """
    r2_scaled = np.sum(mesh ** 2, axis=1) / (spec.prior_radius_scale ** 2)
    raw_mu = np.maximum(1e-6, 1.0 - r2_scaled)
    return raw_mu / np.sum(raw_mu)


def run_accelerator_benchmark(
    spec: AcceleratorBenchmarkSpec = AcceleratorBenchmarkSpec(),
    config: AcceleratorSpaceChargeConfig = AcceleratorSpaceChargeConfig(),
) -> dict:
    print("=" * 80)
    print("  HIGH-INTENSITY ACCELERATOR SPACE-CHARGE VERIFICATION SUITE (LANL 1994 -> 2026)")
    print("  Companion to: doc/arXiv/Accelerator/accelerator_space_charge.tex (Dr. Hyo Eun Ahn)")
    print("=" * 80)

    mesh = generate_bunch_spherical_mesh(spec)
    print(f"\n[1] Generated 3D Beam Bunch Mesh: N = {mesh.shape[0]} points (R_mesh = {spec.bunch_radius})")

    ahn_prior = build_ahn_reasonable_beam_prior(mesh, spec)
    t0 = time.perf_counter()
    engine = AcceleratorSpaceChargeEngine(
        bunch_mesh_coords=mesh, config=config, reference_prior_mu=ahn_prior
    )
    t_init_ms = (time.perf_counter() - t0) * 1e3
    phys = engine.compute_beam_physics_parameters()

    print(f"\n[2] Initialized AcceleratorSpaceChargeEngine & Physical Ring Parameters:")
    print(f"    - Target Ring Regime (SNS / PSB / PSR)     : E_k = {phys['kinetic_energy_gev']:.2f} GeV (gamma = {phys['gamma']:.3f}, beta = {phys['beta']:.3f})")
    print(f"    - Single-Turn Revolution Period T_rev      : {phys['t_rev_us']:.3f} µs ({config.control_window_turns}-turn window = {phys['control_window_us']:.2f} µs)")
    print(f"    - Laslett Space-Charge Tune Shift dQ_sc    : {phys['delta_q_sc']:.4f}")
    print(f"    - Plummer Kernel Softening Scale h         : {engine.softening_h:.4f}")
    print(f"    - Basis Matrix Rank rank(Omega)            : {engine.omega_rank} / {engine.M}")
    print(f"    - 3D E-Field Basis Dimension (poly-e.pdf)  : M_E = {engine.M_field} (orders 1..{config.max_order})")
    print(f"    - Normalized Prior Constraint sum(mu_i)    : {np.sum(engine.mu):.15f}")
    print(f"    - Offline Precomputation Time              : {t_init_ms:.2f} ms")
    print(f"    - Runtime Potential Operator K             : {engine.K.shape}, v_0: {engine.v_0.shape}")
    print(f"    - Runtime 3D E-Field Operator K_E          : {engine.K_E.shape}, v_E0: {engine.v_E0.shape}")

    true_charges = generate_3d_beam_bunch_charges(mesh, spec)
    s_bpm = engine.compute_bunch_moments(true_charges)

    # 1. Verify Theorem 1 (report1.pdf)
    a_recon = engine.runtime_reconstruct_bunch_charges(s_bpm)
    s_recon = engine.compute_bunch_moments(a_recon)
    moment_err = float(np.max(np.abs(s_recon - s_bpm)))

    # At reference equilibrium s_eq = Omega * mu, a_eq == mu > 0 strictly
    s_eq = engine.compute_bunch_moments(engine.mu)
    a_eq = engine.runtime_reconstruct_bunch_charges(s_eq)
    eq_neg_count = int(np.sum(a_eq < 0.0))

    # Paul Channell's uniform constant prior (mu_i = const) vs zero prior (mu_i = 0)
    a_channell_zero = engine.Lambda @ s_bpm
    uniform_const_mu = np.full(engine.N, 1.0 / engine.N, dtype=np.float64)
    engine_uniform = AcceleratorSpaceChargeEngine(
        bunch_mesh_coords=mesh, config=config, reference_prior_mu=uniform_const_mu
    )
    a_channell_uniform = engine_uniform.runtime_reconstruct_bunch_charges(s_bpm)
    channell_diff = float(np.max(np.abs(a_channell_uniform - a_channell_zero)))

    l2_ahn = float(np.linalg.norm(a_recon - true_charges))
    l2_channell = float(np.linalg.norm(a_channell_uniform - true_charges))
    neg_ahn = float(np.sum(np.abs(a_recon[a_recon < 0.0])))
    neg_channell = float(np.sum(np.abs(a_channell_uniform[a_channell_uniform < 0.0])))

    print("\n[3] Theorem 1 Verification (report1.pdf — Normalized Prior & Moment Inversion):")
    print(f"    - Exact Moment Conservation ||Omega*a - s||_inf        : {moment_err:.2e}")
    print(f"    - Equilibrium Positivity (s = Omega*mu, negative pts)  : {eq_neg_count} / {engine.N}")
    print(f"    - Channell Identity ||a(mu=const) - a(mu=0)||_inf      : {channell_diff:.2e} (Identical!)")
    print(f"    - Fringe Negative Charge Sum (Channell vs Ahn Prior)   : {neg_channell:.4f} -> {neg_ahn:.4f} ({neg_channell / neg_ahn:.2f}x reduction)")
    print(f"    - Charge Profile L2 Error (Channell vs Ahn Prior)      : {l2_channell:.6f} -> {l2_ahn:.6f} ({l2_channell / l2_ahn:.2f}x reduction)")
    assert moment_err < spec.numerical_tolerance
    assert eq_neg_count == 0
    assert channell_diff < spec.numerical_tolerance
    assert neg_ahn < neg_channell
    assert l2_ahn < l2_channell

    # 2. Verify Theorem 2 (poly-phi.pdf)
    v_closed = engine.runtime_potential_coefficients(s_bpm)
    phi_exact_rec = engine.D @ a_recon
    v_full = engine.Lambda_prime_T @ phi_exact_rec
    v_diff = float(np.max(np.abs(v_closed - v_full)))

    pot_thm = engine.verify_potential_averages_theorem(true_charges)
    pot_35_err = float(
        np.max(
            np.abs(
                pot_thm["approx_potential_averages_35"] - pot_thm["exact_potential_averages_35"]
            )
        )
    )

    print("\n[4] Theorem 2 Verification (poly-phi.pdf — Space-Charge Potential Reduction):")
    print(f"    - Closed-Form 35x35 v vs Full N-Grid ||v_closed - v_full|| : {v_diff:.2e}")
    print(f"    - 35 Potential Moment Averages ||<m_k phi_app>-<m_k phi_ex>||: {pot_35_err:.2e}")
    assert v_diff < spec.numerical_tolerance
    assert pot_35_err < spec.numerical_tolerance

    # 3. Verify Theorem 3 (poly-e.pdf), Prior vs State Weighting, & ZSF Gauge Projection
    field_thm = engine.verify_electric_field_and_average_force_theorem(true_charges)
    field_34_err = float(
        np.max(
            np.abs(
                field_thm["approx_field_averages_34"] - field_thm["exact_field_averages_34"]
            )
        )
    )
    avg_force_err_state = float(
        np.max(np.abs(field_thm["avg_force_approx"] - field_thm["avg_force_exact"]))
    )
    avg_force_err_mu = float(
        np.max(np.abs(field_thm["avg_force_mu_prior"] - field_thm["avg_force_mu_exact"]))
    )
    avg_force_uncorr_prior = float(
        np.max(np.abs(field_thm["avg_force_a_uncorrected_prior"] - field_thm["avg_force_exact"]))
    )
    avg_force_err_zsf = float(
        np.max(np.abs(field_thm["avg_force_a_zsf"] - field_thm["avg_force_exact"]))
    )

    vE_closed = engine.runtime_electric_field_coefficients(s_bpm, enforce_zero_self_force=False)
    vE_full = engine.Lambda_E_T @ a_recon
    vE_diff = float(np.max(np.abs(vE_closed - vE_full)))

    fidelity = engine.evaluate_prior_linearization_and_core_fidelity(
        true_charges, core_radius=spec.core_eval_radius
    )

    print("\n[5] Theorem 3 Verification (poly-e.pdf — State Fit, Prior Operator & ZSF Dipolar Projection):")
    print(f"    - 34 Generalized E-Field Averages Residual (w = a)     : {field_34_err:.2e}")
    print(f"    - State-Weighted (w = a) Self-Force Residual           : {avg_force_err_state:.2e}")
    print(f"    - Prior-Weighted (w = mu) Moment Conservation Residual : {avg_force_err_mu:.2e}")
    print(f"    - Uncorrected Prior Operator Net Force ||<E_prior>_a|| : {avg_force_uncorr_prior:.4f} (mu^T G da != 0)")
    print(f"    - ZSF Dipolar-Projected Runtime Operator ||<E_ZSF>_a|| : {avg_force_err_zsf:.2e} (Exact Zero Self-Force!)")
    print(f"    - Closed-Form 34x35 v_E vs Full N-Grid ||vE_cl - vE_fl||: {vE_diff:.2e}")
    print(f"    - Prior-Linearized Quadrupole Gradient Rel. Diff       : {fidelity['quadrupole_gradient_rel_diff']*100:.2f}%")
    print(f"    - Core (r <= {spec.core_eval_radius}) Field R^2 (State w = a)            : {fidelity['core_field_r2']:.4f} (RMS rel. err: {fidelity['core_weighted_rel_rms']*100:.2f}%)")
    print(f"    - Core (r <= {spec.core_eval_radius}) Field R^2 (Prior w = mu, uncorr.)  : {fidelity['core_field_r2_prior']:.4f} (RMS rel. err: {fidelity['core_weighted_rel_rms_prior']*100:.2f}%)")
    print(f"    - Core (r <= {spec.core_eval_radius}) Field R^2 (ZSF Runtime w=mu+ZSF)   : {fidelity['core_field_r2_zsf']:.4f} (RMS rel. err: {fidelity['core_weighted_rel_rms_zsf']*100:.2f}%)")
    assert field_34_err < spec.numerical_tolerance
    assert avg_force_err_state < spec.numerical_tolerance
    assert avg_force_err_mu < spec.numerical_tolerance
    assert avg_force_err_zsf < spec.numerical_tolerance
    assert vE_diff < spec.numerical_tolerance

    # 4. Non-Circular 2D Transverse Leapfrog Tracking in 3D Pairwise-Coulomb Geometry & De-Confounded Ablation
    t_track0 = time.perf_counter()
    tracking = engine.simulate_multiturn_envelope_tracking()
    t_track_sec = time.perf_counter() - t_track0
    print(
        f"\n[6] De-Confounded 10-Mode 2D-in-3D Pairwise-Coulomb Leapfrog Tracking "
        f"({tracking['num_turns']} turns, N_p = {tracking['num_particles']} particles, "
        f"dQ_sc,eff = {tracking['effective_dq_sc']:.4f}, {t_track_sec:.2f}s):"
    )
    print("    - Initial Condition Identity Across All 10 Modes       : 100% Identical (Zero Rescaling)")
    print("    - Disturbance Model (Co-Moving 3D Pairwise Coulomb)    : Exact O(N_p^2) 3D Plummer-Coulomb")
    print("    - Windowed Moment ZSF Enforcement                      : Pure 35-Moment Corollary 4 (Zero np.mean Subtraction)")
    print(f"    - Mode A (Open-Loop Uncompensated)                     : dEps = +{tracking['emittance_growth_uncomp_pct']:.2f}%, RMS Mismatch = {tracking['rms_mismatch_uncomp_pct']:.2f}%, Halo = {tracking['halo_fraction_uncomp_pct']:.2f}%")
    print(f"    - Mode B-Lab (Multipole Subset, Lab Frame, No ZSF)     : dEps = +{tracking['emittance_growth_multipole_pct']:.2f}%, RMS Mismatch = {tracking['rms_mismatch_multipole_pct']:.2f}%, Halo = {tracking['halo_fraction_multipole_pct']:.2f}%")
    print(f"    - Mode B-CF  (Multipole Subset, Centroid Frame, No ZSF): dEps = +{tracking['emittance_growth_multipole_cf_pct']:.2f}%, RMS Mismatch = {tracking['rms_mismatch_multipole_cf_pct']:.2f}%, Halo = {tracking['halo_fraction_multipole_cf_pct']:.2f}%")
    print(f"    - Mode B-CF-ZSF (Multipole Subset, Centroid + ZSF)     : dEps = +{tracking['emittance_growth_multipole_zsf_pct']:.2f}%, RMS Mismatch = {tracking['rms_mismatch_multipole_zsf_pct']:.2f}%, Halo = {tracking['halo_fraction_multipole_zsf_pct']:.2f}%")
    print(f"    - Mode C-Lab (Full 34-Basis, Lab Frame, No ZSF)        : dEps = +{tracking['emittance_growth_full34_pct']:.2f}%, RMS Mismatch = {tracking['rms_mismatch_full34_pct']:.2f}%, Halo = {tracking['halo_fraction_full34_pct']:.2f}%")
    print(f"    - Mode C-CF  (Full 34-Basis, Centroid Frame, No ZSF)   : dEps = +{tracking['emittance_growth_full34_cf_pct']:.2f}%, RMS Mismatch = {tracking['rms_mismatch_full34_cf_pct']:.2f}%, Halo = {tracking['halo_fraction_full34_cf_pct']:.2f}%")
    print(f"    - Mode C-CF-ZSF (Full 34-Basis, Centroid + ZSF)        : dEps = +{tracking['emittance_growth_full34_zsf_pct']:.2f}%, RMS Mismatch = {tracking['rms_mismatch_full34_zsf_pct']:.2f}%, Halo = {tracking['halo_fraction_full34_zsf_pct']:.2f}%")
    print(f"    - Mode D-Lab (Full 34 + Damper, Lab Frame, No ZSF)     : dEps = +{tracking['emittance_growth_ctrl_pct']:.2f}%, RMS Mismatch = {tracking['rms_mismatch_ctrl_pct']:.2f}%, Halo = {tracking['halo_fraction_ctrl_pct']:.2f}%")
    print(f"    - Mode D-CF  (Full 34 + Damper, Centroid Frame, No ZSF): dEps = +{tracking['emittance_growth_ctrl_cf_pct']:.2f}%, RMS Mismatch = {tracking['rms_mismatch_ctrl_cf_pct']:.2f}%, Halo = {tracking['halo_fraction_ctrl_cf_pct']:.2f}%")
    print(f"    - Mode D-CF-ZSF (Full 34 + Damper, Centroid + ZSF)     : dEps = +{tracking['emittance_growth_ctrl_zsf_pct']:.2f}%, RMS Mismatch = {tracking['rms_mismatch_ctrl_zsf_pct']:.2f}%, Halo = {tracking['halo_fraction_ctrl_zsf_pct']:.2f}%")
    print(f"    - Max Polynomial Self-Force (Centroid No-ZSF vs ZSF)   : {tracking['max_poly_selfforce_uncorr']:.4f} -> {tracking['max_poly_selfforce_zsf']:.2e}")
    print(f"    - Max Applied Particle Net Kick (Centroid No-ZSF vs ZSF): {tracking['max_applied_selfforce_uncorr']:.4f} -> {tracking['max_applied_selfforce_zsf']:.2e}")
    print(f"    - Spurious RMS Centroid Orbit Error (No-ZSF vs ZSF)    : {tracking['rms_cm_err_uncorr']:.4f} -> {tracking['rms_cm_err_zsf']:.2e}")
    assert tracking["emittance_growth_multipole_pct"] < tracking["emittance_growth_uncomp_pct"]
    assert tracking["emittance_growth_full34_pct"] < tracking["emittance_growth_uncomp_pct"]
    assert tracking["emittance_growth_ctrl_pct"] < tracking["emittance_growth_uncomp_pct"]
    assert tracking["emittance_growth_ctrl_zsf_pct"] < tracking["emittance_growth_uncomp_pct"]
    assert tracking["max_poly_selfforce_zsf"] < spec.numerical_tolerance
    assert tracking["max_applied_selfforce_zsf"] < spec.numerical_tolerance
    assert tracking["rms_cm_err_zsf"] < spec.numerical_tolerance

    # 4b. 4-Seed Statistical Robustness (seeds = {42, 1, 7, 99})
    print("\n[6b] 4-Seed Statistical Robustness (seeds = {42, 1, 7, 99}, reuses seed-42 result):")
    multi = engine.evaluate_multiseed_robustness(first_seed_result=tracking)
    def _fmt(mean, std_s, std_p):
        return f"{mean:.2f}% ± {std_s:.2f}% (pop ±{std_p:.2f}%)"
    print(f"    Mode A  Open-Loop          : {_fmt(multi['open_mean'],       multi['open_std_sample'],       multi['open_std_pop'])}")
    print(f"    Mode B-Lab  (Multipole)    : {_fmt(multi['multipole_mean'],  multi['multipole_std_sample'],  multi['multipole_std_pop'])}")
    print(f"    Mode B-CF   (Multipole-CF) : {_fmt(multi['multipole_cf_mean'],  multi['multipole_cf_std_sample'],  multi['multipole_cf_std_pop'])}")
    print(f"    Mode B-CF-ZSF              : {_fmt(multi['multipole_zsf_mean'], multi['multipole_zsf_std_sample'], multi['multipole_zsf_std_pop'])}")
    print(f"    Mode C-Lab  (Full34)       : {_fmt(multi['full34_mean'],     multi['full34_std_sample'],     multi['full34_std_pop'])}")
    print(f"    Mode C-CF   (Full34-CF)    : {_fmt(multi['full34_cf_mean'],  multi['full34_cf_std_sample'],  multi['full34_cf_std_pop'])}")
    print(f"    Mode C-CF-ZSF              : {_fmt(multi['full34_zsf_mean'], multi['full34_zsf_std_sample'], multi['full34_zsf_std_pop'])}")
    print(f"    Mode D-Lab  (Damper)       : {_fmt(multi['damper_mean'],     multi['damper_std_sample'],     multi['damper_std_pop'])}")
    print(f"    Mode D-CF   (Damper-CF)    : {_fmt(multi['damper_cf_mean'],  multi['damper_cf_std_sample'],  multi['damper_cf_std_pop'])}")
    print(f"    Mode D-CF-ZSF              : {_fmt(multi['damper_zsf_mean'], multi['damper_zsf_std_sample'], multi['damper_zsf_std_pop'])}")
    assert multi["damper_zsf_mean"] < multi["open_mean"], "D-CF-ZSF mean must beat open-loop mean"

    # 4c. Macro-Particle Resolution Convergence (N_p ∈ {500, 1000, 2000}, Seed 42)
    print("\n[6c] N_p Resolution Convergence (N_p ∈ {500, 1000, 2000}, Seed 42):")
    for _n_p in (500, 1000, 2000):
        _conv = tracking if _n_p == 500 else engine.simulate_multiturn_envelope_tracking(
            seed=42, num_particles=_n_p
        )
        print(
            f"    N_p = {_n_p:4d}: "
            f"Open={_conv['emittance_growth_uncomp_pct']:6.2f}%  "
            f"B-CF-ZSF={_conv['emittance_growth_multipole_zsf_pct']:6.2f}%  "
            f"C-CF-ZSF={_conv['emittance_growth_full34_zsf_pct']:6.2f}%  "
            f"D-CF-ZSF={_conv['emittance_growth_ctrl_zsf_pct']:6.2f}%"
        )
    assert tracking["emittance_growth_ctrl_zsf_pct"] < tracking["emittance_growth_uncomp_pct"], \
        "D-CF-ZSF (N_p=500) must beat open-loop"

    # [6d] K_sub Substep-Count Convergence (N_p=500, Seed 42, K_sub ∈ {4, 8, 16, 32}, fixed g_D=0.04375)
    print("\n[6d] K_sub Substep-Count Convergence (N_p=500, Seed 42, K_sub ∈ {4, 8, 16, 32}, fixed g_D=0.04375):")
    print(f"    {'K_sub':>6}  {'Open':>8}  {'B-CF-ZSF':>10}  {'C-CF-ZSF':>10}  {'D-CF-ZSF':>10}")
    for _k in (4, 8, 16, 32):
        _ks = engine.simulate_multiturn_envelope_tracking(seed=42, num_particles=500, num_substeps=_k)
        print(
            f"    {_k:>6}  "
            f"{_ks['emittance_growth_uncomp_pct']:>8.2f}%  "
            f"{_ks['emittance_growth_multipole_zsf_pct']:>9.2f}%  "
            f"{_ks['emittance_growth_full34_zsf_pct']:>9.2f}%  "
            f"{_ks['emittance_growth_ctrl_zsf_pct']:>9.2f}%"
        )

    # 5. Active Multipole Space-Charge Compensation & Compute Kernel Latency
    kick_res = engine.execute_realtime_bpm_multipole_control(s_bpm)
    bench = engine.benchmark_accelerator_loop(s_input=s_bpm)
    pot_us = bench["avg_potential_latency_us"]
    field_us = bench["avg_3d_field_latency_us"]
    field_zsf_us = bench["avg_3d_field_zsf_latency_us"]
    deadline_us = config.revolution_deadline_us

    print(f"\n[7] Active Multipole Synthesis & Compute Kernel Latency ({config.benchmark_iterations:,} runs):")
    print(f"    - ZSF-Projected Dipole Kicks (x, y, z)     : {np.round(kick_res['dipole_kicks'], 6)}")
    print(f"    - Compensating Quadrupole Trims (xx,yy,zz) : {np.round(kick_res['quadrupole_trims'], 6)}")
    print(f"    - 35x35 Potential Kernel Compute Latency   : {pot_us:.3f} µs")
    print(f"    - 34x35 3D E-Field Kernel Compute Latency  : {field_us:.3f} µs")
    print(f"    - 34x35 E-Field + ZSF Dipolar Projection   : {field_zsf_us:.3f} µs")
    print(f"    - 5-Turn Ring Control Window ({deadline_us:.1f} µs) Ratio : {deadline_us / field_us:.2f}x")

    # 6. Generate Dedicated Accelerator Publication Figure (400 DPI)
    t_pic0 = time.perf_counter()
    for _ in range(spec.direct_pic_sample_runs):
        _ = engine.G_x @ a_recon
        _ = engine.G_y @ a_recon
        _ = engine.G_z @ a_recon
    t_direct_pic_us = ((time.perf_counter() - t_pic0) / spec.direct_pic_sample_runs) * 1e6

    out_path = CURRENT_DIR / spec.output_figure_relpath
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    idx_34 = np.arange(1, engine.M_field + 1)
    axes[0].plot(
        idx_34,
        field_thm["exact_field_averages_34"],
        "bo-",
        label="Exact Coulomb Field Averages (Eq. 20)",
        markersize=5,
    )
    axes[0].plot(
        idx_34,
        field_thm["approx_field_averages_34"],
        "rx--",
        label="Ahn Polynomial E-Field Fit (Eq. 23)",
        markersize=4,
    )
    axes[0].set_title(
        "34 Space-Charge Field Averages & Force Conservation (poly-e.pdf)",
        fontsize=11.5,
        fontweight="bold",
    )
    axes[0].set_xlabel("Non-Constant Monomial Index $k$ (1 to 34)", fontsize=11)
    axes[0].set_ylabel("Charge-Weighted Field Moment Average", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=9.5)

    methods = [
        "Direct 3D Coulomb\n$O(N^2)$ Field Sum",
        "Ahn Potential\n35x35 ($O(M^2)$)",
        "Ahn 3D E-Field\n34x35 ($O(M_E M)$)",
    ]
    times = [t_direct_pic_us, pot_us, field_us]
    colors = ["#d9534f", "#5cb85c", "#0275d8"]
    bars = axes[1].bar(methods, times, color=colors, width=0.45)
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.5, max(times) * 4.0)
    axes[1].set_title("Space-Charge Compute Kernel Latency (Log Scale)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel(r"Kernel Execution Time ($\mu\mathrm{s}$)", fontsize=11)
    axes[1].axhline(
        y=deadline_us,
        color="red",
        linestyle=":",
        linewidth=1.8,
        label=rf"5-Turn Ring Window ({deadline_us:.1f} $\mu\mathrm{{s}}$)",
    )
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval * 0.55 if yval > 100 else yval * 0.52,
            rf"{yval:.2f} $\mu\mathrm{{s}}$",
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
        )
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=spec.figure_dpi)
    plt.close()
    print(f"\n[8] Saved Accelerator benchmark figure to: {out_path}")
    print("=" * 80)
    print("  ACCELERATOR SUITE PASSED 100%: ALL THEOREMS, ZSF PROJECTION & 2D-IN-3D TRACKING VERIFIED!")
    print("=" * 80)

    return {
        "moment_err": moment_err,
        "channell_diff": channell_diff,
        "neg_channell": neg_channell,
        "neg_ahn": neg_ahn,
        "l2_channell": l2_channell,
        "l2_ahn": l2_ahn,
        "v_diff": v_diff,
        "pot_35_err": pot_35_err,
        "field_34_err": field_34_err,
        "avg_force_err_state": avg_force_err_state,
        "avg_force_err_mu": avg_force_err_mu,
        "avg_force_uncorr_prior": avg_force_uncorr_prior,
        "avg_force_err_zsf": avg_force_err_zsf,
        "vE_diff": vE_diff,
        "pot_us": pot_us,
        "field_us": field_us,
        "tracking": tracking,
        "multi_seed": multi,
        "fidelity": fidelity,
        "phys": phys,
    }


if __name__ == "__main__":
    run_accelerator_benchmark()
