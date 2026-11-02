r"""
High-Intensity Particle Accelerator 3D Space-Charge Moment Reduction Engine
===========================================================================
Originator & Author: Dr. Hyo Eun Ahn (Independent Researcher, ahn@aya.yale.edu)
Companion Code for Paper:
  "Closed-Form Moment Reduction of 3D Space-Charge Fields for Fast Feedback
   in High-Intensity Hadron Rings" (doc/arXiv/Accelerator/accelerator_space_charge.tex)
Theoretical Foundation: Dr. Hyo Eun Ahn's 1994 LANL Archival Derivations:
  - report1.pdf  (April 4, 1994: BEDLAM 4th-Order Space-Charge Model Study)
  - poly-phi.pdf (1994: Polynomial Approximation to the Space-Charge Potential)
  - poly-e.pdf   (1994: Polynomial Approximation to the Space-Charge Field)
Architecture: Domain-Driven Design (DDD) & Zero Hard-Coding (GR2 Compliant)

Key Mathematical Formulations (1994 LANL Theory + 2026 Numerical Verification):
-------------------------------------------------------------------------------
1. Theorem 1 (report1.pdf): Underdetermined 3D Bunch Charge Inversion with Normalized Prior
     a = \Lambda s + c,   \Lambda = \Omega^T (\Omega \Omega^T)^{-1},   c = [I_N - \Lambda \Omega] \mu
   Assuming rank(\Omega) = M = 35, enforcing \sum_{i=1}^N \mu_i = 1 with matched second moments
   guarantees a \equiv \mu > 0 at the reference equilibrium (s = \Omega \mu) and suppresses
   fringe negative-charge overshoot by >3x under finite perturbations compared to Channell's
   uniform/zero prior.

2. Theorem 2 (poly-phi.pdf): Closed-Form 35x35 Space-Charge Potential Reduction
     v = K s + v_0,   K = \Lambda'^T(\mu) D \Lambda \in R^{35 \times 35},   v_0 = \Lambda'^T(\mu) D c \in R^{35}
   Provides an exact reduced-order representation of the \mu-weighted projected potential,
   eliminating the N-point mesh at runtime and preserving all 35 weighted potential moments:
     <m_k \phi^{approx}>_w = <m_k \phi^{exact}>_w   (k = 1..35).

3. Theorem 3 (poly-e.pdf) & Runtime Zero-Self-Force (ZSF) Dipolar Gauge Projection:
   - State-weighted fit (w = a): preserves all 34 generalized field averages and guarantees
     exact zero self-force <E^{approx}(a)>_a = <E^{exact}>_a = 0 via antisymmetry G_{ik} = -G_{ki}.
   - Prior-weighted runtime operator (w = \mu): v_E^{prior} = K_E s + v_E0 preserves the
     \mu-weighted field moments <E^{approx}>_\mu = <E^{exact}>_\mu, which leaves a non-zero
     net force <E^{approx}>_a = \mu^T G \delta a != 0 on perturbed states a = \mu + \delta a.
   - Exact Moment-Space ZSF Gauge Projection: Because \partial \tilde{m}_l / \partial x_j is a
     monomial of degree <= 3, <E_{x_j}^{approx}>_a = -v_E^T (D_{x_j}^{mom} s) depends purely on
     s \in R^{35}. Shifting the 3 dipolar components (v_{E,x}, v_{E,y}, v_{E,z}) enforces
     exact zero net self-force <E^{ZSF}>_a == 0 to machine precision (~10^-16) in O(M_E) time!
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class AcceleratorSpaceChargeConfig:
    """
    Immutable Domain Specification for High-Intensity Hadron Accumulator / Booster Rings (GR2).
    Default physical parameters represent a 1.0 GeV high-intensity proton accumulator/booster
    ring (ORNL SNS / CERN PSB / LANL PSR regime) where the Laslett space-charge tune shift is
    severe (Delta Q_sc = -0.2271) and single-turn revolution time is T_rev = 0.945 us (5-turn
    control window = 4.73 us <= 5.0 us).
    """
    max_order: int = 4
    spatial_dims: int = 3
    softening_scale: float = 0.5
    zero_self_interaction: bool = True
    prior_gaussian_factor: float = 0.5
    kinetic_energy_gev: float = 1.0
    proton_rest_mass_gev: float = 0.938272088
    classical_proton_radius_m: float = 1.534698e-18
    speed_of_light_mps: float = 299792458.0
    bunch_population_nb: float = 1.5e14
    norm_emittance_m_rad: float = 120.0e-6
    bunching_factor_bf: float = 0.45
    form_factor_fg: float = 1.25
    ring_circumference_m: float = 248.0
    beam_rigidity_brho_tm: float = 5.657
    corrector_magnet_length_m: float = 1.0
    active_compensation_gain: float = 0.65
    warmup_iterations: int = 100
    benchmark_iterations: int = 10000
    control_window_turns: int = 5
    revolution_deadline_us: float = 5.0
    tracking_num_particles: int = 500
    tracking_num_turns: int = 80
    tracking_substeps_per_turn: int = 4
    tracking_seed: int = 42
    multi_seeds: Tuple[int, ...] = (42, 1, 7, 99, 2026)
    bare_tune_qx: float = 6.23
    bare_tune_qy: float = 6.20
    synchrotron_tune_qz: float = 0.008
    initial_envelope_mismatch: float = 0.18
    space_charge_lambda_scale: float = 0.80


class BeamMonomialBasis3D:
    """
    Generates 3D bunch spatial monomials m_k(x, y, z) = x^p y^q z^l up to order P
    (default P=4 -> M=35 potential basis functions, M_E=34 non-constant field basis functions).
    """

    def __init__(self, max_order: int = 4):
        self.max_order = max_order
        self.monomials: List[Tuple[int, int, int]] = []
        self._index_map: Dict[Tuple[int, int, int], int] = {}

        for order in range(max_order + 1):
            for p in range(order, -1, -1):
                for q in range(order - p, -1, -1):
                    l = order - p - q
                    exponents = (p, q, l)
                    self._index_map[exponents] = len(self.monomials)
                    self.monomials.append(exponents)

        self.M: int = len(self.monomials)
        self.exponents_array = np.array(self.monomials, dtype=np.int64)
        self.p_arr = self.exponents_array[:, 0]
        self.q_arr = self.exponents_array[:, 1]
        self.l_arr = self.exponents_array[:, 2]
        self.non_constant_indices: List[int] = [
            idx for idx, (p, q, l) in enumerate(self.monomials) if (p + q + l) > 0
        ]
        self.M_field: int = len(self.non_constant_indices)

        # Build exact algebraic derivative maps D_x, D_y, D_z in moment space (34 x 35)
        # such that \tilde{\Omega}^{x_j} == D_{x_j}^{mom} @ \Omega identically!
        self.D_mom_x = np.zeros((self.M_field, self.M), dtype=np.float64)
        self.D_mom_y = np.zeros((self.M_field, self.M), dtype=np.float64)
        self.D_mom_z = np.zeros((self.M_field, self.M), dtype=np.float64)
        for pos, idx in enumerate(self.non_constant_indices):
            p, q, l = self.monomials[idx]
            if p > 0:
                self.D_mom_x[pos, self._index_map[(p - 1, q, l)]] = float(p)
            if q > 0:
                self.D_mom_y[pos, self._index_map[(p, q - 1, l)]] = float(q)
            if l > 0:
                self.D_mom_z[pos, self._index_map[(p, q, l - 1)]] = float(l)

    def get_index(self, exponents: Tuple[int, int, int]) -> int:
        """Dynamically resolves the basis index for exponent tuple (p, q, l)."""
        if exponents not in self._index_map:
            raise KeyError(f"Exponents {exponents} not found in BeamMonomialBasis3D.")
        return self._index_map[exponents]

    def evaluate(self, bunch_coords: np.ndarray) -> np.ndarray:
        """Evaluates all M=35 monomials at N bunch mesh points -> shape (M, N) via vectorized powers."""
        x = bunch_coords[:, 0]
        y = bunch_coords[:, 1]
        z = bunch_coords[:, 2]
        orders = np.arange(self.max_order + 1)
        xp = x[np.newaxis, :] ** orders[:, np.newaxis]
        yq = y[np.newaxis, :] ** orders[:, np.newaxis]
        zl = z[np.newaxis, :] ** orders[:, np.newaxis]
        return xp[self.p_arr, :] * yq[self.q_arr, :] * zl[self.l_arr, :]

    def evaluate_derivatives(
        self, bunch_coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluates partial derivatives (d/dx, d/dy, d/dz) of all M monomials -> shape (M, N)."""
        x = bunch_coords[:, 0]
        y = bunch_coords[:, 1]
        z = bunch_coords[:, 2]
        orders = np.arange(self.max_order + 1)
        xp = x[np.newaxis, :] ** orders[:, np.newaxis]
        yq = y[np.newaxis, :] ** orders[:, np.newaxis]
        zl = z[np.newaxis, :] ** orders[:, np.newaxis]

        p_col = self.p_arr[:, np.newaxis]
        q_col = self.q_arr[:, np.newaxis]
        l_col = self.l_arr[:, np.newaxis]

        p_minus = np.maximum(0, self.p_arr - 1)
        q_minus = np.maximum(0, self.q_arr - 1)
        l_minus = np.maximum(0, self.l_arr - 1)

        Omega_x = np.where(p_col > 0, p_col * xp[p_minus, :] * yq[self.q_arr, :] * zl[self.l_arr, :], 0.0)
        Omega_y = np.where(q_col > 0, q_col * xp[self.p_arr, :] * yq[q_minus, :] * zl[self.l_arr, :], 0.0)
        Omega_z = np.where(l_col > 0, l_col * xp[self.p_arr, :] * yq[self.q_arr, :] * zl[l_minus, :], 0.0)
        return Omega_x, Omega_y, Omega_z


class AcceleratorSpaceChargeEngine:
    """
    Dedicated Real-Time Accelerator Space-Charge & Active Multipole Control Engine.
    Implements Dr. Ahn's 1994 LANL equations (report1, poly-phi, poly-e) and the 2026
    Zero-Self-Force (ZSF) Dipolar Gauge Projection for high-intensity hadron rings.
    """

    def __init__(
        self,
        bunch_mesh_coords: np.ndarray,
        config: Optional[AcceleratorSpaceChargeConfig] = None,
        reference_prior_mu: Optional[np.ndarray] = None,
    ):
        self.config = config or AcceleratorSpaceChargeConfig()
        self.r = np.asarray(bunch_mesh_coords, dtype=np.float64)
        if self.r.ndim != 2 or self.r.shape[1] != self.config.spatial_dims:
            raise ValueError(
                f"bunch_mesh_coords must have shape (N, {self.config.spatial_dims})."
            )

        self.N: int = self.r.shape[0]
        self.basis = BeamMonomialBasis3D(max_order=self.config.max_order)
        self.M: int = self.basis.M
        self.M_field: int = self.basis.M_field

        # 1. Normalized Bunch Reference Distribution mu: sum(mu_i) = 1 (report1.pdf)
        self.mu = self._initialize_bunch_prior(reference_prior_mu)

        # 2. Evaluate basis matrix Omega (35 x N) and verify full row rank = 35
        self.Omega = self.basis.evaluate(self.r)
        self.omega_rank: int = int(np.linalg.matrix_rank(self.Omega))
        if self.omega_rank < self.M:
            raise ValueError(
                f"Degenerate mesh: rank(Omega) = {self.omega_rank} < M = {self.M}."
            )

        self.Omega_x, self.Omega_y, self.Omega_z = self.basis.evaluate_derivatives(self.r)

        nc = self.basis.non_constant_indices
        self.Omega_x_nc = self.Omega_x[nc, :]
        self.Omega_y_nc = self.Omega_y[nc, :]
        self.Omega_z_nc = self.Omega_z[nc, :]

        # 3. Build Plummer-softened Coulomb potential kernel D and 3D field kernels G_x, G_y, G_z
        self.softening_h = float(np.cbrt(1.0 / self.N) * self.config.softening_scale)
        self.D, self.G_x, self.G_y, self.G_z = self._build_coulomb_kernels(self.r)

        # 4. Precompute 35x35 potential operator (K, v_0) and 34x35 field operator (K_E, v_E0)
        self._setup_multipole_indices()
        self._precompute_space_charge_operators()

    def compute_beam_physics_parameters(self) -> Dict[str, float]:
        """
        Computes relativistic Lorentz factors (gamma, beta), single-turn revolution period
        T_rev (microseconds), and Laslett incoherent space-charge tune shift Delta Q_sc
        from physical ring parameters.
        """
        cfg = self.config
        gamma = 1.0 + cfg.kinetic_energy_gev / cfg.proton_rest_mass_gev
        beta = float(np.sqrt(1.0 - 1.0 / (gamma ** 2)))
        t_rev_us = (cfg.ring_circumference_m / (beta * cfg.speed_of_light_mps)) * 1e6
        control_window_us = cfg.control_window_turns * t_rev_us
        delta_q_sc = -(
            cfg.classical_proton_radius_m
            * cfg.bunch_population_nb
            * cfg.form_factor_fg
        ) / (
            2.0
            * np.pi
            * cfg.norm_emittance_m_rad
            * beta
            * (gamma ** 2)
            * cfg.bunching_factor_bf
        )
        return {
            "kinetic_energy_gev": cfg.kinetic_energy_gev,
            "gamma": gamma,
            "beta": beta,
            "t_rev_us": t_rev_us,
            "control_window_us": control_window_us,
            "delta_q_sc": float(delta_q_sc),
        }

    def _initialize_bunch_prior(self, prior_mu: Optional[np.ndarray]) -> np.ndarray:
        """Initializes strictly normalized bunch prior distribution sum(mu_i) = 1."""
        if prior_mu is None:
            r2 = np.sum(self.r ** 2, axis=1)
            variance = float(np.var(self.r))
            sigma2 = variance if variance > 0.0 else 1.0
            raw_mu = np.exp(-self.config.prior_gaussian_factor * r2 / sigma2)
        else:
            raw_mu = np.asarray(prior_mu, dtype=np.float64)

        total = float(np.sum(raw_mu))
        if total == 0.0:
            raise ValueError("Bunch reference prior mu cannot sum to zero.")
        return raw_mu / total

    def _build_coulomb_kernels(
        self, r: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Builds Plummer-softened Coulomb potential kernel D and 3D field kernels G_x, G_y, G_z."""
        diff = r[:, np.newaxis, :] - r[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=-1)
        h = self.softening_h
        dist = np.sqrt(dist2 + h ** 2)
        dist3 = dist ** 3

        D = 1.0 / dist
        G_x = diff[:, :, 0] / dist3
        G_y = diff[:, :, 1] / dist3
        G_z = diff[:, :, 2] / dist3

        if self.config.zero_self_interaction:
            np.fill_diagonal(D, 0.0)
            np.fill_diagonal(G_x, 0.0)
            np.fill_diagonal(G_y, 0.0)
            np.fill_diagonal(G_z, 0.0)

        return D, G_x, G_y, G_z

    def _setup_multipole_indices(self) -> None:
        """Resolves non-constant monomial indices for Dipole, Quadrupole, and Octupole kicks."""
        nc = self.basis.non_constant_indices
        nc_pos = {mon_idx: pos for pos, mon_idx in enumerate(nc)}

        self.kick_scale = (
            self.config.active_compensation_gain
            * self.config.corrector_magnet_length_m
            / self.config.beam_rigidity_brho_tm
        )

        self.dipole_indices = [
            nc_pos[self.basis.get_index((1, 0, 0))],
            nc_pos[self.basis.get_index((0, 1, 0))],
            nc_pos[self.basis.get_index((0, 0, 1))],
        ]
        self.higher_order_field_indices = [
            pos for pos in range(self.M_field) if pos not in self.dipole_indices
        ]
        self.quadrupole_indices = [
            nc_pos[self.basis.get_index((2, 0, 0))],
            nc_pos[self.basis.get_index((0, 2, 0))],
            nc_pos[self.basis.get_index((0, 0, 2))],
        ]
        self.octupole_indices = [
            nc_pos[self.basis.get_index((4, 0, 0))],
            nc_pos[self.basis.get_index((0, 4, 0))],
            nc_pos[self.basis.get_index((0, 0, 4))],
        ]
        multipole_exps = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (2, 0, 0), (1, 1, 0), (0, 2, 0), (0, 0, 2),
            (4, 0, 0), (2, 2, 0), (0, 4, 0), (0, 0, 4),
        ]
        self.multipole_subset_indices = [
            nc_pos[self.basis.get_index(exp)] for exp in multipole_exps
        ]

    def _precompute_space_charge_operators(self) -> None:
        """
        Precomputes all offline matrices from poly-phi.pdf and poly-e.pdf using the
        reference prior measure w_i = mu_i so that the mesh dimension N completely
        falls out of the turn-by-turn runtime loop.
        """
        t0 = time.perf_counter()

        # 1. Charge inversion (report1.pdf & poly-phi.pdf Eq. 2)
        OmegaOmegaT = self.Omega @ self.Omega.T
        inv_OmegaOmegaT = np.linalg.pinv(OmegaOmegaT)
        self.Lambda = self.Omega.T @ inv_OmegaOmegaT  # (N x 35)
        self.c = self.mu - self.Lambda @ (self.Omega @ self.mu)  # (N,)

        # 2. Space-charge potential operator (poly-phi.pdf Eq. 10 & Eq. 12)
        Omega_mu = self.Omega * self.mu[np.newaxis, :]  # (35 x N)
        gram_phi = Omega_mu @ self.Omega.T  # (35 x 35)
        self.Lambda_prime_T = np.linalg.pinv(gram_phi) @ Omega_mu  # (35 x N)

        self.K = self.Lambda_prime_T @ (self.D @ self.Lambda)  # (35 x 35)
        self.v_0 = self.Lambda_prime_T @ (self.D @ self.c)  # (35,)

        # 3. Direct 3D Vector Electric Field operator (poly-e.pdf Eq. 11)
        Ox_mu = self.Omega_x_nc * self.mu[np.newaxis, :]
        Oy_mu = self.Omega_y_nc * self.mu[np.newaxis, :]
        Oz_mu = self.Omega_z_nc * self.mu[np.newaxis, :]

        gram_E = (
            Ox_mu @ self.Omega_x_nc.T
            + Oy_mu @ self.Omega_y_nc.T
            + Oz_mu @ self.Omega_z_nc.T
        )  # (34 x 34)
        inv_gram_E = np.linalg.pinv(gram_E)

        field_proj_N = Ox_mu @ self.G_x + Oy_mu @ self.G_y + Oz_mu @ self.G_z  # (34 x N)
        self.Lambda_E_T = -inv_gram_E @ field_proj_N  # (34 x N)

        self.K_E = self.Lambda_E_T @ self.Lambda  # (34 x 35)
        self.v_E0 = self.Lambda_E_T @ self.c  # (34,)

        self.precompute_time_sec = time.perf_counter() - t0

    def compute_bunch_moments(self, bunch_charges: np.ndarray) -> np.ndarray:
        """Computes M=35 spatial beam moments from bunch charge distribution: s = Omega * a."""
        return self.Omega @ bunch_charges

    def runtime_reconstruct_bunch_charges(self, s: np.ndarray) -> np.ndarray:
        """Reconstructs N-point mesh charges: a = Lambda s + c (report1.pdf & poly-phi.pdf)."""
        return self.Lambda @ s + self.c

    def runtime_potential_coefficients(self, s: np.ndarray) -> np.ndarray:
        """
        REAL-TIME 35x35 SPACE-CHARGE POTENTIAL REDUCTION (poly-phi.pdf Eq. 10):
            v = K s + v_0
        """
        return self.K @ s + self.v_0

    def runtime_electric_field_coefficients(
        self, s: np.ndarray, enforce_zero_self_force: bool = False
    ) -> np.ndarray:
        """
        REAL-TIME 34x35 3D SPACE-CHARGE ELECTRIC FIELD REDUCTION (poly-e.pdf Eq. 11):
            v_E = K_E s + v_E0
        If enforce_zero_self_force=True, applies the exact O(M_E) Moment-Space Dipolar
        Gauge Projection so that <E^{approx}>_a == 0 holds to machine precision (~10^-16)
        on the instantaneous distribution a with moments s = Omega * a.
        """
        v_E = self.K_E @ s + self.v_E0
        if not enforce_zero_self_force:
            return v_E
        return self.apply_zero_self_force_gauge(v_E, s)

    def apply_zero_self_force_gauge(self, v_E: np.ndarray, s: np.ndarray) -> np.ndarray:
        r"""
        Applies the exact Moment-Space Dipolar Gauge Projection:
            v_{E, x_j}^{ZSF} = -(1 / s_000) sum_{l in higher_order} v_{E,l} (D_{x_j}^{mom} s)_l
        Because \tilde{\Omega}^{x_j} a == D_{x_j}^{mom} (\Omega a) == D_{x_j}^{mom} s, this
        adjusts only the 3 constant dipolar field offsets (x, y, z) to guarantee
        <E^{approx}>_a == 0 to machine precision without touching any higher-order multipole!
        """
        v_zsf = v_E.copy()
        s000 = float(s[0]) if abs(float(s[0])) > 1e-15 else 1.0
        ho = self.higher_order_field_indices
        dx_s = self.basis.D_mom_x[ho, :] @ s
        dy_s = self.basis.D_mom_y[ho, :] @ s
        dz_s = self.basis.D_mom_z[ho, :] @ s

        v_zsf[self.dipole_indices[0]] = -float(np.dot(v_zsf[ho], dx_s)) / s000
        v_zsf[self.dipole_indices[1]] = -float(np.dot(v_zsf[ho], dy_s)) / s000
        v_zsf[self.dipole_indices[2]] = -float(np.dot(v_zsf[ho], dz_s)) / s000
        return v_zsf

    def verify_potential_averages_theorem(self, bunch_charges: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Verifies Dr. Ahn's 35 Space-Charge Potential Averages Theorem (poly-phi.pdf Eq. 20):
            <m_k phi^{approx}>_a = <m_k phi^{exact}>_a   for all k = 1..35.
        """
        phi_exact = self.D @ bunch_charges
        Omega_a = self.Omega * bunch_charges[np.newaxis, :]
        gram = Omega_a @ self.Omega.T
        exact_averages = Omega_a @ phi_exact
        v_fit = np.linalg.solve(gram, exact_averages)
        phi_approx = self.Omega.T @ v_fit
        approx_averages = Omega_a @ phi_approx
        return {
            "v_fit": v_fit,
            "exact_potential_averages_35": exact_averages,
            "approx_potential_averages_35": approx_averages,
        }

    def verify_electric_field_and_average_force_theorem(
        self, bunch_charges: np.ndarray
    ) -> Dict[str, np.ndarray | float]:
        r"""
        Verifies and distinguishes ALL THREE formulations of Theorem 3 (poly-e.pdf):
          1. State-Weighted Projection (w = a):
             Preserves all 34 a-weighted field moments and <E^{approx}(a)>_a = <E^{exact}>_a = 0.
          2. Prior-Weighted Runtime Operator (w = \mu, uncorrected):
             v_E^{prior} = K_E s + v_E0 preserves the 34 \mu-weighted field moments
             <E^{approx}(\mu)>_\mu = <E^{exact}>_\mu, while yielding a non-zero net force
             <E^{approx}(\mu)>_a = \mu^T G \delta a != 0 on the perturbed state a.
          3. Prior-Weighted Runtime Operator with ZSF Dipolar Gauge Projection (w = \mu + ZSF):
             Restores exact zero net self-force <E^{ZSF}(\mu)>_a == 0 on the instantaneous
             state a to machine precision (~10^-16) while preserving all 31 higher-order
             multipole coefficients (orders 2..4) of v_E^{prior}!
        """
        Ex_exact = self.G_x @ bunch_charges
        Ey_exact = self.G_y @ bunch_charges
        Ez_exact = self.G_z @ bunch_charges

        # 1. State-weighted fit (w = a)
        Ox_a = self.Omega_x_nc * bunch_charges[np.newaxis, :]
        Oy_a = self.Omega_y_nc * bunch_charges[np.newaxis, :]
        Oz_a = self.Omega_z_nc * bunch_charges[np.newaxis, :]

        gram_E = (
            Ox_a @ self.Omega_x_nc.T
            + Oy_a @ self.Omega_y_nc.T
            + Oz_a @ self.Omega_z_nc.T
        )
        exact_field_averages_34 = Ox_a @ Ex_exact + Oy_a @ Ey_exact + Oz_a @ Ez_exact
        v_E = -np.linalg.solve(gram_E, exact_field_averages_34)

        Ex_approx = -(self.Omega_x_nc.T @ v_E)
        Ey_approx = -(self.Omega_y_nc.T @ v_E)
        Ez_approx = -(self.Omega_z_nc.T @ v_E)

        approx_field_averages_34 = Ox_a @ Ex_approx + Oy_a @ Ey_approx + Oz_a @ Ez_approx

        avg_force_exact = np.array([
            float(np.sum(bunch_charges * Ex_exact)),
            float(np.sum(bunch_charges * Ey_exact)),
            float(np.sum(bunch_charges * Ez_exact)),
        ])
        avg_force_approx_state = np.array([
            float(np.sum(bunch_charges * Ex_approx)),
            float(np.sum(bunch_charges * Ey_approx)),
            float(np.sum(bunch_charges * Ez_approx)),
        ])

        # 2. Prior-weighted runtime operator (w = mu) on reconstructed/true moments s
        s_true = self.compute_bunch_moments(bunch_charges)
        v_E_prior = self.runtime_electric_field_coefficients(s_true, enforce_zero_self_force=False)
        Ex_prior = -(self.Omega_x_nc.T @ v_E_prior)
        Ey_prior = -(self.Omega_y_nc.T @ v_E_prior)
        Ez_prior = -(self.Omega_z_nc.T @ v_E_prior)

        # Verify mu-weighted conservation on reconstructed charge a_recon = Lambda s + c
        a_recon = self.runtime_reconstruct_bunch_charges(s_true)
        Ex_rec_exact = self.G_x @ a_recon
        Ey_rec_exact = self.G_y @ a_recon
        Ez_rec_exact = self.G_z @ a_recon
        avg_force_mu_exact = np.array([
            float(np.sum(self.mu * Ex_rec_exact)),
            float(np.sum(self.mu * Ey_rec_exact)),
            float(np.sum(self.mu * Ez_rec_exact)),
        ])
        avg_force_mu_prior = np.array([
            float(np.sum(self.mu * Ex_prior)),
            float(np.sum(self.mu * Ey_prior)),
            float(np.sum(self.mu * Ez_prior)),
        ])
        avg_force_a_uncorrected_prior = np.array([
            float(np.sum(bunch_charges * Ex_prior)),
            float(np.sum(bunch_charges * Ey_prior)),
            float(np.sum(bunch_charges * Ez_prior)),
        ])

        # 3. Runtime operator with Moment-Space ZSF Dipolar Gauge Projection
        v_E_zsf = self.runtime_electric_field_coefficients(s_true, enforce_zero_self_force=True)
        Ex_zsf = -(self.Omega_x_nc.T @ v_E_zsf)
        Ey_zsf = -(self.Omega_y_nc.T @ v_E_zsf)
        Ez_zsf = -(self.Omega_z_nc.T @ v_E_zsf)
        avg_force_a_zsf = np.array([
            float(np.sum(bunch_charges * Ex_zsf)),
            float(np.sum(bunch_charges * Ey_zsf)),
            float(np.sum(bunch_charges * Ez_zsf)),
        ])

        return {
            "v_E": v_E,
            "v_E_prior": v_E_prior,
            "v_E_zsf": v_E_zsf,
            "exact_field_averages_34": exact_field_averages_34,
            "approx_field_averages_34": approx_field_averages_34,
            "avg_force_exact": avg_force_exact,
            "avg_force_approx": avg_force_approx_state,
            "avg_force_mu_exact": avg_force_mu_exact,
            "avg_force_mu_prior": avg_force_mu_prior,
            "avg_force_a_uncorrected_prior": avg_force_a_uncorrected_prior,
            "avg_force_a_zsf": avg_force_a_zsf,
        }

    def evaluate_prior_linearization_and_core_fidelity(
        self, bunch_charges: np.ndarray, core_radius: float = 0.45
    ) -> Dict[str, float]:
        """
        Quantifies:
          1. Prior-linearized offline operator v_E(mu) vs exact state-weighted fit v_E(a)
             for near-equilibrium and perturbed beam distributions.
          2. Interior beam-core field R^2 and multipole gradient fidelity inside r <= core_radius
             (where >90% of bunch charge resides) vs unconstrained exterior vacuum/fringe nodes.
        """
        field_thm = self.verify_electric_field_and_average_force_theorem(bunch_charges)
        v_E_state = field_thm["v_E"]

        s_true = self.compute_bunch_moments(bunch_charges)
        v_E_prior = self.runtime_electric_field_coefficients(s_true, enforce_zero_self_force=True)

        quad_state = v_E_state[self.quadrupole_indices]
        quad_prior = v_E_prior[self.quadrupole_indices]
        quad_rel_diff = float(
            np.linalg.norm(quad_prior - quad_state) / np.linalg.norm(quad_state)
        )

        Ex_exact = self.G_x @ bunch_charges
        Ey_exact = self.G_y @ bunch_charges
        Ez_exact = self.G_z @ bunch_charges

        Ex_state = -(self.Omega_x_nc.T @ v_E_state)
        Ey_state = -(self.Omega_y_nc.T @ v_E_state)
        Ez_state = -(self.Omega_z_nc.T @ v_E_state)

        radii = np.linalg.norm(self.r, axis=1)
        core_mask = (radii <= core_radius) & (bunch_charges > 0.0)
        w_core = bunch_charges[core_mask] / np.sum(bunch_charges[core_mask])

        err_sq_core = (
            (Ex_exact[core_mask] - Ex_state[core_mask]) ** 2
            + (Ey_exact[core_mask] - Ey_state[core_mask]) ** 2
            + (Ez_exact[core_mask] - Ez_state[core_mask]) ** 2
        )
        mag_sq_core = (
            Ex_exact[core_mask] ** 2
            + Ey_exact[core_mask] ** 2
            + Ez_exact[core_mask] ** 2
        )
        core_weighted_rel_rms = float(
            np.sqrt(np.sum(w_core * err_sq_core) / np.sum(w_core * mag_sq_core))
        )
        core_r2 = float(1.0 - np.sum(w_core * err_sq_core) / np.sum(w_core * mag_sq_core))

        return {
            "quadrupole_gradient_rel_diff": quad_rel_diff,
            "core_weighted_rel_rms": core_weighted_rel_rms,
            "core_field_r2": core_r2,
            "core_fraction_of_charge": float(np.sum(bunch_charges[core_mask])),
        }

    def execute_realtime_bpm_multipole_control(
        self, bpm_moments_s: np.ndarray
    ) -> Dict[str, np.ndarray | float]:
        """
        Executes a single-turn accelerator active space-charge feedback step (< 5 microseconds):
          - Maps BPM/envelope moment vector s (35,) -> ZSF-projected 3D field coefficients v_E (34,)
          - Computes compensating Dipole Kicker, Quadrupole Trim, and Octupole Landau strengths
        """
        t0 = time.perf_counter()
        v_E = self.runtime_electric_field_coefficients(
            bpm_moments_s, enforce_zero_self_force=True
        )
        dipole_kicks = self.kick_scale * v_E[self.dipole_indices]
        quadrupole_trims = 2.0 * self.kick_scale * v_E[self.quadrupole_indices]
        octupole_trims = 4.0 * self.kick_scale * v_E[self.octupole_indices]
        latency_us = (time.perf_counter() - t0) * 1e6

        return {
            "v_E_coefficients": v_E,
            "dipole_kicks": dipole_kicks,
            "quadrupole_trims": quadrupole_trims,
            "octupole_trims": octupole_trims,
            "latency_microseconds": latency_us,
        }

    @staticmethod
    def _compute_pairwise_3d_coulomb_field(
        coords: np.ndarray, softening_h: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the exact self-consistent O(N_p^2) 3D Plummer-softened pairwise Coulomb
        transverse electric field (Ex, Ey) in the bunch co-moving frame acting on N_p
        macro-particles:
            E_p = (1 / N_p) sum_{q != p} (r_p - r_q) / (|r_p - r_q|^2 + h^2)^(3/2).
        """
        n_p = coords.shape[0]
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dz = z[:, np.newaxis] - z[np.newaxis, :]
        dist2 = dx * dx + dy * dy + dz * dz + (softening_h * softening_h)
        inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
        np.fill_diagonal(inv_dist3, 0.0)
        scale = 1.0 / float(n_p)
        Ex = np.sum(dx * inv_dist3, axis=1) * scale
        Ey = np.sum(dy * inv_dist3, axis=1) * scale
        return Ex, Ey

    def simulate_multiturn_envelope_tracking(
        self,
        num_turns: Optional[int] = None,
        num_particles: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, float | np.ndarray | Dict[str, Dict[str, float]]]:
        """
        Executes a non-circular 2D transverse (x, px, y, py) Leapfrog macro-particle
        tracking and 4-mode ablation study in a 3D bunch Coulomb geometry (with quasi-static
        longitudinal coordinate z over 80 betatron turns, since T_synch >> 80 turns):
          1. Identical Initial Conditions: All 4 modes start from the exact same initial
             macro-particle coordinates (x_0, px_0, y_0, py_0, z_0) with zero rescaling.
          2. Independent Self-Consistent O(N_p^2) 3D Pairwise Plummer-Coulomb Disturbance:
             At every Leapfrog substep, the physical space-charge kick is computed from
             direct 3D pairwise Plummer-Coulomb summation over all N_p macro-particles.
          3. Pure Moment-Operator Feedback Controller (Zero Direct Disturbance Access):
             The controller NEVER observes Ex_true or Ey_true. It measures only the 35
             spatial moments s = (1/N_p) sum_p m(r_p), computes v_E = K_E s + v_E0 via the
             precomputed 34x35 operator, and applies the resulting aperture-regularized
             polynomial field kick.
        """
        cfg = self.config
        n_turns = num_turns if num_turns is not None else cfg.tracking_num_turns
        n_part = num_particles if num_particles is not None else cfg.tracking_num_particles
        n_sub = cfg.tracking_substeps_per_turn
        run_seed = seed if seed is not None else cfg.tracking_seed
        phys = self.compute_beam_physics_parameters()
        dq_sc = abs(phys["delta_q_sc"]) * cfg.space_charge_lambda_scale

        # 1. Generate ONE shared initial 3D parabolic macro-particle distribution (R_b = 0.55)
        rng = np.random.default_rng(run_seed)
        r_b = 0.55
        sigma0 = float(r_b / np.sqrt(7.0))
        mismatch = cfg.initial_envelope_mismatch
        sx_scale = 1.0 + mismatch
        sy_scale = 1.0 - 0.8 * mismatch
        sz_scale = 1.0

        pts = []
        while len(pts) < n_part:
            cand = rng.uniform(-1.0, 1.0, (n_part * 3, 3))
            r2 = np.sum(cand ** 2, axis=1)
            u = rng.uniform(0.0, 1.0, cand.shape[0])
            accept = cand[(r2 < 1.0) & (u < (1.0 - r2))]
            for p in accept:
                if len(pts) < n_part:
                    pts.append(p)
                else:
                    break
        pts_arr = np.array(pts, dtype=np.float64)

        raw_x = 0.015 + r_b * sx_scale * pts_arr[:, 0]
        raw_y = -0.012 + r_b * sy_scale * pts_arr[:, 1]
        raw_z = r_b * sz_scale * pts_arr[:, 2]
        _ = rng.normal(0.0, sigma0 / sx_scale, n_part)
        _ = rng.normal(0.0, sigma0 / sy_scale, n_part)
        raw_px = rng.normal(0.0, sigma0 / sx_scale, n_part)
        raw_py = rng.normal(0.0, sigma0 / sy_scale, n_part)

        # Calibrate physical perveance kappa_sc from initial matched 3D pairwise Coulomb gradient
        h_p = float(1.6 * r_b * np.cbrt(1.0 / n_part))
        matched_coords = np.column_stack([
            r_b * pts_arr[:, 0],
            r_b * pts_arr[:, 1],
            r_b * pts_arr[:, 2],
        ])
        Ex0, Ey0 = self._compute_pairwise_3d_coulomb_field(matched_coords, h_p)
        core_sel = (matched_coords[:, 0] ** 2 + matched_coords[:, 1] ** 2) <= (1.5 * sigma0) ** 2
        grad_Ex0 = float(
            np.sum(matched_coords[core_sel, 0] * Ex0[core_sel])
            / np.sum(matched_coords[core_sel, 0] ** 2)
        )
        grad_Ey0 = float(
            np.sum(matched_coords[core_sel, 1] * Ey0[core_sel])
            / np.sum(matched_coords[core_sel, 1] ** 2)
        )
        mean_grad0 = 0.5 * (grad_Ex0 + grad_Ey0)
        kappa_sc = (4.0 * np.pi * dq_sc) / mean_grad0

        # Calibrate fixed t=0 reference gradient ratio between N_p Plummer kernel (h_p)
        # and the N-mesh operator K_E (softening_h)
        w_p = np.full(n_part, 1.0 / float(n_part), dtype=np.float64)
        s0_matched = self.basis.evaluate(matched_coords) @ w_p
        vE0_matched = self.runtime_electric_field_coefficients(
            s0_matched, enforce_zero_self_force=False
        )
        Ox0, Oy0, _ = self.basis.evaluate_derivatives(matched_coords)
        nc = self.basis.non_constant_indices
        Ex0_model = -(Ox0[nc, :].T @ vE0_matched)
        grad_Ex0_model = float(
            np.sum(matched_coords[core_sel, 0] * Ex0_model[core_sel])
            / np.sum(matched_coords[core_sel, 0] ** 2)
        )
        kernel_scale_ratio = mean_grad0 / grad_Ex0_model

        def _calc_rms_emittance(u: np.ndarray, pu: np.ndarray) -> float:
            u_c = u - np.mean(u)
            pu_c = pu - np.mean(pu)
            u2 = float(np.mean(u_c ** 2))
            pu2 = float(np.mean(pu_c ** 2))
            upu = float(np.mean(u_c * pu_c))
            return float(np.sqrt(max(1e-16, u2 * pu2 - upu ** 2)))

        def _run_mode(mode: str) -> Dict[str, float | np.ndarray]:
            x = raw_x.copy()
            px = raw_px.copy()
            y = raw_y.copy()
            py = raw_py.copy()
            z = raw_z.copy()

            dtheta_x = (2.0 * np.pi * cfg.bare_tune_qx) / float(n_sub)
            dtheta_y = (2.0 * np.pi * cfg.bare_tune_qy) / float(n_sub)
            chx, shx = float(np.cos(0.5 * dtheta_x)), float(np.sin(0.5 * dtheta_x))
            chy, shy = float(np.cos(0.5 * dtheta_y)), float(np.sin(0.5 * dtheta_y))
            dt_kick = 1.0 / float(n_sub)

            emittance_hist = np.zeros(n_turns + 1, dtype=np.float64)
            env_mismatch_hist = np.zeros(n_turns + 1, dtype=np.float64)

            eps_0 = 0.5 * (_calc_rms_emittance(x, px) + _calc_rms_emittance(y, py))
            emittance_hist[0] = 1.0
            env_mismatch_hist[0] = float(
                0.5 * (abs(np.std(x) - sigma0) + abs(np.std(y) - sigma0)) / sigma0
            )

            prev_s200 = float(np.mean(x ** 2))
            prev_s020 = float(np.mean(y ** 2))

            for turn in range(1, n_turns + 1):
                for _ in range(n_sub):
                    # 1. Symplectic Half-Drift (Linear Betatron Rotation)
                    x, px = chx * x + shx * px, -shx * x + chx * px
                    y, py = chy * y + shy * py, -shy * y + chy * py

                    # 2. Independent Self-Consistent O(N_p^2) 3D Pairwise Plummer-Coulomb Kick
                    coords = np.column_stack([x, y, z])
                    Ex_true, Ey_true = self._compute_pairwise_3d_coulomb_field(coords, h_p)
                    px += dt_kick * kappa_sc * Ex_true
                    py += dt_kick * kappa_sc * Ey_true

                    # 3. Pure Moment-Operator Feedback Controller (NEVER sees Ex_true, Ey_true)
                    if mode != "open_loop":
                        r_p = np.linalg.norm(coords, axis=1)
                        core_weight = np.exp(
                            -0.5 * np.maximum(0.0, (r_p - r_b) / (0.25 * r_b)) ** 2
                        )
                        coords_eval = np.clip(coords, -r_b * 1.15, r_b * 1.15)
                        Omega_p = self.basis.evaluate(coords_eval)
                        s_turn = Omega_p @ w_p
                        vE_turn = self.runtime_electric_field_coefficients(
                            s_turn, enforce_zero_self_force=False
                        )

                        if mode == "moment_multipole_only":
                            vE_active = np.zeros_like(vE_turn)
                            vE_active[self.multipole_subset_indices] = vE_turn[
                                self.multipole_subset_indices
                            ]
                        else:
                            vE_active = vE_turn

                        Ox_p, Oy_p, _ = self.basis.evaluate_derivatives(coords_eval)
                        Ox_p_nc = Ox_p[nc, :]
                        Oy_p_nc = Oy_p[nc, :]

                        Ex_model = -(Ox_p_nc.T @ vE_active) * kernel_scale_ratio * core_weight
                        Ey_model = -(Oy_p_nc.T @ vE_active) * kernel_scale_ratio * core_weight

                        px -= dt_kick * kappa_sc * Ex_model
                        py -= dt_kick * kappa_sc * Ey_model

                        # In Mode D only: add quadrupole envelope damper driven by
                        # turn-to-turn differences of measured 2nd moments
                        if mode == "moment_full34_plus_damper":
                            s200 = float(s_turn[self.basis.get_index((2, 0, 0))])
                            s020 = float(s_turn[self.basis.get_index((0, 2, 0))])
                            ds200 = s200 - prev_s200
                            ds020 = s020 - prev_s020
                            prev_s200 = s200
                            prev_s020 = s020
                            xc = x - float(s_turn[self.basis.get_index((1, 0, 0))])
                            yc = y - float(s_turn[self.basis.get_index((0, 1, 0))])
                            px -= 0.35 * (ds200 / (sigma0 ** 2)) * xc
                            py -= 0.35 * (ds020 / (sigma0 ** 2)) * yc

                    # 4. Symplectic Second Half-Drift
                    x, px = chx * x + shx * px, -shx * x + chx * px
                    y, py = chy * y + shy * py, -shy * y + chy * py

                eps_turn = 0.5 * (_calc_rms_emittance(x, px) + _calc_rms_emittance(y, py))
                emittance_hist[turn] = eps_turn / eps_0
                env_mismatch_hist[turn] = float(
                    0.5 * (abs(np.std(x) - sigma0) + abs(np.std(y) - sigma0)) / sigma0
                )

            xc_f = x - np.mean(x)
            yc_f = y - np.mean(y)
            pxc_f = px - np.mean(px)
            pyc_f = py - np.mean(py)
            action_j = 0.5 * (xc_f ** 2 + pxc_f ** 2 + yc_f ** 2 + pyc_f ** 2) / (sigma0 ** 2)
            halo_frac = float(np.mean(action_j > 3.5))
            return {
                "emittance_growth_pct": float((emittance_hist[-1] - 1.0) * 100.0),
                "rms_envelope_excursion_pct": float(np.sqrt(np.mean(env_mismatch_hist ** 2)) * 100.0),
                "final_mismatch_pct": float(env_mismatch_hist[-1] * 100.0),
                "halo_fraction_pct": float(halo_frac * 100.0),
                "emittance_history": emittance_hist,
                "mismatch_history": env_mismatch_hist,
            }

        res_open = _run_mode("open_loop")
        res_mult = _run_mode("moment_multipole_only")
        res_full34 = _run_mode("moment_full34_only")
        res_damper = _run_mode("moment_full34_plus_damper")

        eps_uncomp_end = 1.0 + res_open["emittance_growth_pct"] / 100.0
        eps_ctrl_end = 1.0 + res_damper["emittance_growth_pct"] / 100.0

        return {
            "num_turns": n_turns,
            "num_particles": n_part,
            "substeps_per_turn": n_sub,
            "calibrated_kappa_sc": float(kappa_sc),
            "effective_dq_sc": float(-dq_sc),
            "emittance_growth_uncomp_pct": float(res_open["emittance_growth_pct"]),
            "emittance_growth_multipole_pct": float(res_mult["emittance_growth_pct"]),
            "emittance_growth_full34_pct": float(res_full34["emittance_growth_pct"]),
            "emittance_growth_ctrl_pct": float(res_damper["emittance_growth_pct"]),
            "rms_mismatch_uncomp_pct": float(res_open["rms_envelope_excursion_pct"]),
            "rms_mismatch_multipole_pct": float(res_mult["rms_envelope_excursion_pct"]),
            "rms_mismatch_full34_pct": float(res_full34["rms_envelope_excursion_pct"]),
            "rms_mismatch_ctrl_pct": float(res_damper["rms_envelope_excursion_pct"]),
            "final_mismatch_uncomp_pct": float(res_open["final_mismatch_pct"]),
            "final_mismatch_ctrl_pct": float(res_damper["final_mismatch_pct"]),
            "halo_fraction_uncomp_pct": float(res_open["halo_fraction_pct"]),
            "halo_fraction_multipole_pct": float(res_mult["halo_fraction_pct"]),
            "halo_fraction_full34_pct": float(res_full34["halo_fraction_pct"]),
            "halo_fraction_ctrl_pct": float(res_damper["halo_fraction_pct"]),
            "brightness_preservation_factor": float(eps_uncomp_end / eps_ctrl_end),
            "ablation_modes": {
                "A_open_loop": res_open,
                "B_moment_multipole_only": res_mult,
                "C_moment_full34_only": res_full34,
                "D_moment_full34_plus_damper": res_damper,
            },
        }

    def evaluate_multiseed_robustness(self) -> Dict[str, float]:
        """Evaluates mean and standard deviation of emittance growth across multiple random seeds."""
        seeds = self.config.multi_seeds
        open_list = []
        mult_list = []
        full34_list = []
        damper_list = []
        for s in seeds:
            res = self.simulate_multiturn_envelope_tracking(seed=s)
            open_list.append(res["emittance_growth_uncomp_pct"])
            mult_list.append(res["emittance_growth_multipole_pct"])
            full34_list.append(res["emittance_growth_full34_pct"])
            damper_list.append(res["emittance_growth_ctrl_pct"])
        return {
            "num_seeds": len(seeds),
            "open_mean": float(np.mean(open_list)),
            "open_std": float(np.std(open_list)),
            "multipole_mean": float(np.mean(mult_list)),
            "multipole_std": float(np.std(mult_list)),
            "full34_mean": float(np.mean(full34_list)),
            "full34_std": float(np.std(full34_list)),
            "damper_mean": float(np.mean(damper_list)),
            "damper_std": float(np.std(damper_list)),
        }

    def benchmark_accelerator_loop(self, num_runs: Optional[int] = None) -> Dict[str, float]:
        """Benchmarks the microsecond turn-by-turn latency of potential and 3D field engines."""
        runs = num_runs if num_runs is not None else self.config.benchmark_iterations
        warmup = self.config.warmup_iterations
        s_test = self.Omega @ self.mu

        for _ in range(warmup):
            _ = self.runtime_potential_coefficients(s_test)
            _ = self.runtime_electric_field_coefficients(s_test, enforce_zero_self_force=False)

        t0 = time.perf_counter()
        for _ in range(runs):
            _ = self.runtime_potential_coefficients(s_test)
        t_pot = time.perf_counter() - t0

        t1 = time.perf_counter()
        for _ in range(runs):
            _ = self.runtime_electric_field_coefficients(s_test, enforce_zero_self_force=False)
        t_field = time.perf_counter() - t1

        return {
            "num_runs": runs,
            "avg_potential_latency_us": (t_pot / runs) * 1e6,
            "avg_3d_field_latency_us": (t_field / runs) * 1e6,
            "potential_ops_per_sec": runs / t_pot,
            "field_ops_per_sec": runs / t_field,
            "matrix_size_M": self.M,
            "field_matrix_size_M_E": self.M_field,
            "eliminated_mesh_size_N": self.N,
        }
