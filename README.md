# High-Intensity Accelerator 3D Space-Charge & Fast Multipole Feedback Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official open-source Python reference engine and verification suite for the manuscript:

> **Closed-Form Moment Reduction of 3D Space-Charge Fields for Fast Feedback in High-Intensity Hadron Rings**  
> **Author:** Dr. Hyo Eun Ahn (`hyo.eun.ahn@proton.me`)  
> **Theoretical Origin:** 1994 Los Alamos National Laboratory (LANL) Space-Charge Moment Formulations (`report1.pdf`, `poly-phi.pdf`, `poly-e.pdf`) + 2026 Moment-Space Zero-Self-Force (ZSF) Dipolar Offset Projection & De-Confounded Pairwise-Coulomb Tracking Ablation

---

## 1. Physical Regime & Mathematical Highlights

In low-to-medium energy high-intensity hadron accumulator rings, rapid-cycling synchrotrons, and injector boosters—such as the **LANL Proton Storage Ring (PSR)** ($E_k = 800\,\mathrm{MeV}, T_{\mathrm{rev}} = 0.358\,\mu\mathrm{s}$), **ORNL Spallation Neutron Source (SNS)** ($E_k = 1.0\,\mathrm{GeV}, T_{\mathrm{rev}} = 0.945\,\mu\mathrm{s}, \Delta Q_{\mathrm{sc}} = -0.2271$), **CERN Proton Synchrotron Booster (PSB)** ($T_{\mathrm{rev}} = 0.57\text{--}1.01\,\mu\mathrm{s}$), and **FAIR SIS18**—the incoherent Laslett space-charge tune shift reaches $|\Delta Q_{\mathrm{sc}}| \approx 0.1\text{--}0.4$, driving envelope mismatch oscillations and transverse emittance growth over a few turns ($5$-turn control window $= 4.73\,\mu\mathrm{s} \le 5.0\,\mu\mathrm{s}$).

This repository implements the three 1994 Los Alamos theorems, the exact $O(M_E)$ Moment-Space Zero-Self-Force (ZSF) dipolar offset projection, three-operator core field fidelity analysis, and a de-confounded non-circular 2D-in-3D pairwise Plummer-Coulomb multi-turn Leapfrog tracking and 10-mode ablation suite:

1. **Theorem 1 — Affine Charge Inversion with Normalized Physical Prior (`report1.pdf`)**:
   $$\mathbf{a} = \mathbf{\Lambda}\mathbf{s} + \mathbf{c}, \qquad \mathbf{\Lambda} = \mathbf{\Omega}^T(\mathbf{\Omega}\mathbf{\Omega}^T)^{-1}, \qquad \mathbf{c} = (\mathbf{I}_N - \mathbf{\Lambda}\mathbf{\Omega})\boldsymbol{\mu}, \qquad \operatorname{rank}(\mathbf{\Omega}) = 35$$
   - Proves that Paul Channell's uniform prior $\mu_i = \mathrm{const}$ is algebraically identical to the zero prior ($\|\mathbf{a}(\mu=\mathrm{const}) - \mathbf{a}(\mu=0)\|_\infty = 8.87 \times 10^{-17}$).
   - Enforcing $\sum_{i=1}^N \mu_i = 1, \mu_i > 0$ guarantees strict pointwise positivity ($\mathbf{a} \equiv \boldsymbol{\mu} > \mathbf{0}$) at equilibrium, maintains $a_i > 0$ if and only if $[\mathbf{\Lambda}(\mathbf{s} - \mathbf{\Omega}\boldsymbol{\mu})]_i > -\mu_i$, and suppresses fringe negative-charge overshoot by **$3.04\times$ ($0.2669 \to 0.0876$)** and $L_2$ profile error by **$2.00\times$ ($0.0326 \to 0.0163$)** on the 3D anisotropic bunch benchmark.

2. **Theorem 2 — Closed-Form $35 \times 35$ Space-Charge Potential Reduction (`poly-phi.pdf`)**:
   $$\mathbf{v} = \mathbf{K}\mathbf{s} + \mathbf{v}_0, \qquad \mathbf{K} = \mathbf{\Lambda}^{\prime T}(\boldsymbol{\mu})\mathbf{D}\mathbf{\Lambda} \in \mathbb{R}^{35 \times 35}$$
   - Eliminates the $N$-point spatial mesh offline ($1.14 \times 10^{-13}$ agreement with full $N$-grid prior-weighted projection) and preserves all $M=35$ weighted potential moment averages to machine precision ($4.44 \times 10^{-16}$).

3. **Theorem 3 & Corollary 4 — Direct 3D Vector Electric Field Reduction (`poly-e.pdf`) & Exact Moment-Space Zero-Self-Force (ZSF) Dipolar Offset Projection**:
   $$\mathbf{v}_E^{\mathrm{prior}} = \mathbf{K}_E\mathbf{s} + \mathbf{v}_{E,0}, \qquad \mathbf{K}_E \in \mathbb{R}^{34 \times 35}$$
   - **Three-Operator Core Fidelity ($r \le 0.45$, $79.9\%$ of bunch charge)**:
     - **State-Weighted Fit ($\mathbf{w} = \mathbf{a}$)**: Preserves all $M_E = 34$ generalized field averages ($4.17 \times 10^{-16}$), zero self-force ($\|\langle\mathbf{E}_{\mathrm{approx}}(\mathbf{a})\rangle_{\mathbf{a}}\|_\infty = 4.58 \times 10^{-16}$), $R^2 = 0.8962$ ($\epsilon_E = 32.22\%$).
     - **Uncorrected Prior Runtime Operator ($\mathbf{w} = \boldsymbol{\mu}$)**: Conserves $\boldsymbol{\mu}$-weighted field averages ($2.03 \times 10^{-15}$), $R^2 = 0.8868$ ($\epsilon_E = 33.64\%$), quadrupole gradient agreement within $11.74\%$, but leaves a non-zero net force $\|\langle\mathbf{E}_{\mathrm{approx}}^{\mathrm{prior}}\rangle_{\mathbf{a}}\|_\infty = 0.1123$.
     - **ZSF Dipolar-Projected Runtime Operator ($\mathbf{w} = \boldsymbol{\mu} + \mathrm{ZSF}$)**: Restores **exact zero net self-force on the perturbed distribution $\mathbf{a}$ ($\|\langle\mathbf{E}_{\mathrm{approx}}^{\mathrm{ZSF}}\rangle_{\mathbf{a}}\|_\infty = 4.02 \times 10^{-16}$)** in $57$ scalar MACs using only $\mathbf{s} \in \mathbb{R}^{35}$ without touching the $N$-point grid or altering higher-order multipole gradients, achieving $R^2 = 0.8878$ ($\epsilon_E = 33.50\%$).

4. **Idealized Multi-Turn 2D-in-3D Pairwise-Coulomb Leapfrog Tracking & De-Confounded 10-Mode Ablation (`simulate_multiturn_envelope_tracking`)**:
   - Tracks $N_p = 500$ macro-particles over $N_{\mathrm{turns}} = 80$ turns ($K_{\mathrm{sub}} = 4$ Leapfrog substeps/turn) from **strictly identical initial conditions** under **exact self-consistent $O(N_p^2)$ 3D pairwise Plummer-Coulomb forces** ($\Delta Q_{\mathrm{sc,eff}} = -0.1817$).
   - Separates **Laboratory-Frame vs. Centroid-Frame alignment** from **Corollary 4 ZSF dipolar projection**, and uses the windowed moment vector $\mathbf{s}_{\mathrm{win}}$ ($s_{\mathrm{win},000}=1$) so that both polynomial and applied net particle kicks vanish to $\sim 10^{-16}$ **purely from the 35 moments with zero particle-level `np.mean` subtraction**:

| Ablation Mode (Identical $t=0$ State, Seed 42) | Frame & ZSF | Max Poly Self-Force | Max Applied Self-Force | Centroid Orbit Err $\epsilon_{\mathrm{cm}}$ | RMS Emittance Growth | RMS Env. Mismatch | Halo ($J_\perp > 3.5\sigma_0^2$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mode A**: Open-Loop | Uncompensated | $0.00$ | $0.00$ | $0.00$ | $+40.65\%$ | $16.60\%$ | $29.60\%$ |
| **Mode B-Lab**: 11-Term Multipole | Lab, No ZSF | $0.4170$ | $0.3931$ | $3.39 \times 10^{-2}$ | $+27.51\%$ | $8.04\%$ | $16.40\%$ |
| **Mode B-CF**: 11-Term Multipole | Centroid, No ZSF | $0.3592$ | $0.3549$ | $4.87 \times 10^{-2}$ | $+17.29\%$ | $7.06\%$ | $16.40\%$ |
| **Mode B-CF-ZSF**: 11-Term Multipole | **Centroid + ZSF** | **$1.49 \times 10^{-16}$** | **$1.56 \times 10^{-16}$** | **$4.85 \times 10^{-17}$** | **$+17.29\%$** | **$7.06\%$** | **$16.40\%$** |
| **Mode C-Lab**: Full 34-Basis | Lab, No ZSF | $0.1833$ | $0.1731$ | $3.66 \times 10^{-2}$ | $+16.44\%$ | $3.74\%$ | $17.60\%$ |
| **Mode C-CF**: Full 34-Basis | Centroid, No ZSF | $0.1680$ | $0.1522$ | $6.00 \times 10^{-2}$ | $+13.45\%$ | $3.98\%$ | $17.00\%$ |
| **Mode C-CF-ZSF**: Full 34-Basis | **Centroid + ZSF** | **$3.84 \times 10^{-16}$** | **$2.13 \times 10^{-16}$** | **$4.17 \times 10^{-17}$** | **$+13.45\%$** | **$3.98\%$** | **$17.00\%$** |
| **Mode D-Lab**: Full 34 + Damper | Lab, No ZSF | $0.1848$ | $0.1658$ | $3.38 \times 10^{-2}$ | $+7.00\%$ | $4.11\%$ | $12.20\%$ |
| **Mode D-CF**: Full 34 + Damper | Centroid, No ZSF | $0.1506$ | $0.1474$ | $8.12 \times 10^{-2}$ | $+6.40\%$ | $4.22\%$ | $13.40\%$ |
| **Mode D-CF-ZSF**: Full 34 + Damper | **Centroid + ZSF** | **$3.98 \times 10^{-16}$** | **$2.42 \times 10^{-16}$** | **$5.66 \times 10^{-17}$** | **$+6.40\%$ ($6.4\times$ lower)** | **$4.22\%$ ($3.9\times$ lower)** | **$13.40\%$ ($2.2\times$ lower)** |

- **4-Seed Statistical Robustness ($\mathrm{seed} \in \{42, 1, 7, 99\}$, Mean $\pm$ Sample SD $\mathrm{ddof}=1$)**:
  - **Mode A**: $\{+40.65\%, +53.20\%, +60.85\%, +46.37\%\} \implies +50.27\% \pm 8.72\%$ ($\mathrm{pop.\ SD} = 7.55\%$)
  - **Mode B-Lab**: $\{+27.51\%, +19.68\%, +21.32\%, +19.03\%\} \implies +21.88\% \pm 3.87\%$ ($\mathrm{pop.\ SD} = 3.35\%$)
  - **Modes B-CF & B-CF-ZSF**: $\{+17.29\%, +16.39\%, +14.86\%, +17.54\%\} \implies +16.52\% \pm 1.21\%$ ($\mathrm{pop.\ SD} = 1.05\%$)
  - **Mode C-Lab**: $\{+16.44\%, +17.53\%, +22.03\%, +14.26\%\} \implies +17.56\% \pm 3.27\%$ ($\mathrm{pop.\ SD} = 2.83\%$)
  - **Modes C-CF & C-CF-ZSF**: $\{+13.45\%, +22.34\%, +16.46\%, +16.29\%\} \implies +17.13\% \pm 3.73\%$ ($\mathrm{pop.\ SD} = 3.23\%$)
  - **Mode D-Lab**: $\{+7.00\%, +8.76\%, +10.86\%, +3.16\%\} \implies +7.44\% \pm 3.26\%$ ($\mathrm{pop.\ SD} = 2.83\%$)
  - **Modes D-CF & D-CF-ZSF**: $\{+6.40\%, +10.11\%, +5.05\%, +7.60\%\} \implies +7.29\% \pm 2.15\%$ ($\mathrm{pop.\ SD} = 1.86\%$)
- **Macro-Particle Resolution Convergence ($N_p \in \{500, 1000, 2000\}$, Seed 42)**:
  - **Mode A (Open-Loop)**: $\{+40.65\%, +38.04\%, +48.55\%\}$ vs. **Mode D-CF-ZSF**: $\{+6.40\%, +7.05\%, +8.99\%\}$ ($5.4\text{--}6.4\times$ reduction across all $N_p$).

---

## 2. Quick Start & Verification (Supports Both Repository Tree & Flat-Folder Execution)

```bash
python3 Test/benchmark_accelerator_space_charge.py
# Or when both .py files are placed in the same flat directory:
python3 benchmark_accelerator_space_charge.py
```
