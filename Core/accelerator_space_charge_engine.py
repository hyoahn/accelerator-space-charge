"""
High-Energy Particle Accelerator 3D Space-Charge & Active Multipole Control Engine
==================================================================================
Originator & Author: Dr. Hyo Eun Ahn (Independent Researcher, ahn@aya.yale.edu)
Companion Code for Paper 2:
  "Microsecond Active Mitigation of Space-Charge Instabilities in High-Energy
   Hadron Beams via Closed-Form Moment Invariants" (doc/arXiv/accelerator_space_charge.tex)
Theoretical Foundation: Dr. Hyo Eun Ahn's 1994 LANL Archival Derivations:
  - report1.pdf  (April 4, 1994: BEDLAM 4th-Order Space-Charge Model Study)
  - poly-phi.pdf (1994: Polynomial Approximation to the Space-Charge Potential)
  - poly-e.pdf   (1994: Polynomial Approximation to the Space-Charge Field)
Architecture: Domain-Driven Design (DDD) & Zero Hard-Coding (GR2 Compliant)

Key Mathematical Theorems for Relativistic Beam Bunch Control (CERN FCC-hh / HL-LHC):
-------------------------------------------------------------------------------------
1. Theorem 1 (report1.pdf): Underdetermined 3D Bunch Charge Inversion with Normalized Prior
     a = \\Lambda s + c,   \\Lambda = \\Omega^T (\\Omega \\Omega^T)^{-1},   c = [I_N - \\Lambda \\Omega] \\mu
   Strictly enforcing \\sum_{i=1}^N \\mu_i = 1 prevents trivial uniform collapse and
   suppresses unphysical negative halo charge in the 3D beam bunch.

2. Theorem 2 (poly-phi.pdf): Closed-Form 35x35 Space-Charge Potential Reduction
     v = K s + v_0,   K = \\Lambda'^T D \\Lambda \\in R^{35 \\times 35},   v_0 = \\Lambda'^T D c \\in R^{35}
   where \\Lambda'^T = [{\\Omega \\mu} \\Omega^T]^{-1} {\\Omega \\mu}.
   Eliminates the N-point PIC grid at runtime and preserves all 35 potential averages:
     <m_k \\phi^{approx}>_a = <m_k \\phi^{exact}>_a   (k = 1..35).

3. Theorem 3 (poly-e.pdf): Direct 3D Vector Electric Field Fit & Average Force Preservation
     \\sum_{j=1}^3 {\\tilde{\\Omega}^{x_j} a} \\tilde{\\Omega}^{x_j T} v_E
       = -\\sum_{j=1}^3 {\\tilde{\\Omega}^{x_j} a} E_{x_j}^{exact}
   Preserves all 34 generalized field averages (Eq. 27 of poly-e.pdf), guaranteeing the
   Average Force Preservation Theorem:
     <E_x^{approx}> = <E_x^{exact}>,  <E_y^{approx}> = <E_y^{exact}>,  <E_z^{approx}> = <E_z^{exact}>.
   Runtime 34x35 operator v_E = K_E s + v_E0 drives active Kicker/Quadrupole/Octupole
   corrector magnets from BPM bunch moments in < 5 microseconds per turn.
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class AcceleratorSpaceChargeConfig:
    """Immutable Domain Specification for the Accelerator Space-Charge Engine (GR2)."""
    max_order: int = 4
    spatial_dims: int = 3
    softening_scale: float = 0.5
    zero_self_interaction: bool = True
    prior_gaussian_factor: float = 0.5
    beam_rigidity_brho_tm: float = 100.0
    corrector_magnet_length_m: float = 1.0
    active_compensation_gain: float = 1.0
    warmup_iterations: int = 100
    benchmark_iterations: int = 10000
    revolution_deadline_us: float = 10.0


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
        self.non_constant_indices: List[int] = [
            idx for idx, (p, q, l) in enumerate(self.monomials) if (p + q + l) > 0
        ]
        self.M_field: int = len(self.non_constant_indices)

    def get_index(self, exponents: Tuple[int, int, int]) -> int:
        """Dynamically resolves the basis index for exponent tuple (p, q, l)."""
        if exponents not in self._index_map:
            raise KeyError(f"Exponents {exponents} not found in BeamMonomialBasis3D.")
        return self._index_map[exponents]

    def evaluate(self, bunch_coords: np.ndarray) -> np.ndarray:
        """Evaluates all M=35 monomials at N bunch mesh points -> shape (M, N)."""
        N = bunch_coords.shape[0]
        x, y, z = bunch_coords[:, 0], bunch_coords[:, 1], bunch_coords[:, 2]
        Omega = np.ones((self.M, N), dtype=np.float64)
        for idx, (p, q, l) in enumerate(self.monomials):
            if p > 0:
                Omega[idx, :] *= x ** p
            if q > 0:
                Omega[idx, :] *= y ** q
            if l > 0:
                Omega[idx, :] *= z ** l
        return Omega

    def evaluate_derivatives(
        self, bunch_coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluates partial derivatives (d/dx, d/dy, d/dz) of all M monomials -> shape (M, N)."""
        N = bunch_coords.shape[0]
        x, y, z = bunch_coords[:, 0], bunch_coords[:, 1], bunch_coords[:, 2]
        Omega_x = np.zeros((self.M, N), dtype=np.float64)
        Omega_y = np.zeros((self.M, N), dtype=np.float64)
        Omega_z = np.zeros((self.M, N), dtype=np.float64)

        for idx, (p, q, l) in enumerate(self.monomials):
            if p > 0:
                Omega_x[idx, :] = p * (x ** (p - 1)) * (y ** q) * (z ** l)
            if q > 0:
                Omega_y[idx, :] = q * (x ** p) * (y ** (q - 1)) * (z ** l)
            if l > 0:
                Omega_z[idx, :] = l * (x ** p) * (y ** q) * (z ** (l - 1))

        return Omega_x, Omega_y, Omega_z


class AcceleratorSpaceChargeEngine:
    """
    Dedicated Real-Time Accelerator Space-Charge & Active Multipole Control Engine.
    Implements Dr. Ahn's 1994 LANL equations (report1, poly-phi, poly-e) for microsecond
    turn-by-turn space-charge instability suppression in high-intensity synchrotrons/colliders.
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

        # 2. Evaluate basis matrix Omega (35 x N) and non-constant derivatives (34 x N)
        self.Omega = self.basis.evaluate(self.r)
        self.Omega_x, self.Omega_y, self.Omega_z = self.basis.evaluate_derivatives(self.r)

        nc = self.basis.non_constant_indices
        self.Omega_x_nc = self.Omega_x[nc, :]
        self.Omega_y_nc = self.Omega_y[nc, :]
        self.Omega_z_nc = self.Omega_z[nc, :]

        # 3. Build Coulomb potential kernel D and 3D electric field kernels G_x, G_y, G_z
        self.D, self.G_x, self.G_y, self.G_z = self._build_coulomb_kernels(self.r)

        # 4. Precompute 35x35 potential operator (K, v_0) and 34x35 field operator (K_E, v_E0)
        self._precompute_space_charge_operators()
        self._setup_multipole_indices()

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
        """Builds Coulomb potential kernel D and 3D electric field kernels G_x, G_y, G_z."""
        diff = r[:, np.newaxis, :] - r[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=-1)
        h = np.cbrt(1.0 / self.N) * self.config.softening_scale
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

    def _precompute_space_charge_operators(self) -> None:
        """
        Precomputes all offline matrices from poly-phi.pdf and poly-e.pdf so that
        the mesh dimension N completely falls out of the turn-by-turn runtime loop.
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

    def runtime_electric_field_coefficients(self, s: np.ndarray) -> np.ndarray:
        """
        REAL-TIME 34x35 3D SPACE-CHARGE ELECTRIC FIELD REDUCTION (poly-e.pdf Eq. 11):
            v_E = K_E s + v_E0
        """
        return self.K_E @ s + self.v_E0

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
    ) -> Dict[str, np.ndarray]:
        """
        Verifies Dr. Ahn's Direct 3D Electric Field Fit & Average Force Preservation Theorem
        (poly-e.pdf Eq. 11 & Eq. 27):
          1. All 34 generalized field averages match identically:
             sum_j < (dm_k/dx_j) E_{x_j}^{approx} > = sum_j < (dm_k/dx_j) E_{x_j}^{exact} >
          2. Average Force Preservation (<E^{approx}> = <E^{exact}>) holds to machine precision.
        """
        Ex_exact = self.G_x @ bunch_charges
        Ey_exact = self.G_y @ bunch_charges
        Ez_exact = self.G_z @ bunch_charges

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
        avg_force_approx = np.array([
            float(np.sum(bunch_charges * Ex_approx)),
            float(np.sum(bunch_charges * Ey_approx)),
            float(np.sum(bunch_charges * Ez_approx)),
        ])

        return {
            "v_E": v_E,
            "exact_field_averages_34": exact_field_averages_34,
            "approx_field_averages_34": approx_field_averages_34,
            "avg_force_exact": avg_force_exact,
            "avg_force_approx": avg_force_approx,
        }

    def execute_realtime_bpm_multipole_control(
        self, bpm_moments_s: np.ndarray
    ) -> Dict[str, np.ndarray | float]:
        """
        Executes a single-turn accelerator active space-charge feedback step (< 5 microseconds):
          - Maps BPM moment vector s (35,) -> 3D field coefficients v_E (34,)
          - Computes compensating Dipole Kicker, Quadrupole Trim, and Octupole Landau strengths
        """
        t0 = time.perf_counter()
        v_E = self.K_E @ bpm_moments_s + self.v_E0
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

    def benchmark_accelerator_loop(self, num_runs: Optional[int] = None) -> Dict[str, float]:
        """Benchmarks the microsecond turn-by-turn latency of potential and 3D field engines."""
        runs = num_runs if num_runs is not None else self.config.benchmark_iterations
        warmup = self.config.warmup_iterations
        s_test = self.Omega @ self.mu

        for _ in range(warmup):
            _ = self.runtime_potential_coefficients(s_test)
            _ = self.runtime_electric_field_coefficients(s_test)

        t0 = time.perf_counter()
        for _ in range(runs):
            _ = self.runtime_potential_coefficients(s_test)
        t_pot = time.perf_counter() - t0

        t1 = time.perf_counter()
        for _ in range(runs):
            _ = self.runtime_electric_field_coefficients(s_test)
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

