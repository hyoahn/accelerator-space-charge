# High-Intensity Accelerator 3D Space-Charge & Active Multipole Control Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official open-source Python reference engine and verification suite for the manuscript:

> **Microsecond Active Mitigation of Space-Charge Instabilities in High-Intensity Hadron Rings via Closed-Form Moment Invariants**  
> **Author:** Dr. Hyo Eun Ahn (`ahn@aya.yale.edu`)  
> **Theoretical Origin:** 1994 Los Alamos National Laboratory (LANL) Space-Charge Moment Study (`report1.pdf`, `poly-phi.pdf`, `poly-e.pdf`)

---

## 1. Physical Regime & Mathematical Highlights

In low-to-medium energy high-intensity hadron accumulator rings, rapid-cycling synchrotrons, and collider injector boosters—such as the **LANL Proton Storage Ring (PSR)** ($E_k = 800\,\mathrm{MeV}, T_{\mathrm{rev}} = 0.358\,\mu\mathrm{s}$), **ORNL Spallation Neutron Source (SNS)** ($E_k = 1.0\,\mathrm{GeV}, T_{\mathrm{rev}} = 0.945\,\mu\mathrm{s}, \Delta Q_{\mathrm{sc}} = -0.2271$), **CERN Proton Synchrotron Booster (PSB)** ($T_{\mathrm{rev}} = 0.57\text{--}1.01\,\mu\mathrm{s}$), and **FAIR SIS18**—the incoherent Laslett space-charge tune shift reaches $|\Delta Q_{\mathrm{sc}}| \approx 0.1\text{--}0.4$, driving envelope mismatch resonances and transverse emittance growth over a few turns ($5$-turn control window $= 4.73\,\mu\mathrm{s} \le 5.0\,\mu\mathrm{s}$).

This repository implements the three foundational 1994 Los Alamos theorems and a non-circular, self-consistent 3D pairwise Plummer-Coulomb multi-turn symplectic tracking and 4-mode ablation suite:

1. **Theorem 1 — Affine Charge Inversion with Normalized Physical Prior (`report1.pdf`)**:
   $$\mathbf{a} = \mathbf{\Lambda}\mathbf{s} + \mathbf{c}, \qquad \mathbf{\Lambda} = \mathbf{\Omega}^T(\mathbf{\Omega}\mathbf{\Omega}^T)^{-1}, \qquad \mathbf{c} = (\mathbf{I}_N - \mathbf{\Lambda}\mathbf{\Omega})\boldsymbol{\mu}$$
   - Proves that Paul Channell's uniform prior $\mu_i = \mathrm{const}$ is algebraically identical to the zero prior ($\|\mathbf{a}(\mu=\mathrm{const}) - \mathbf{a}(\mu=0)\|_\infty = 8.87 \times 10^{-17}$).
   - Enforcing $\sum_{i=1}^N \mu_i = 1$ with matched second moments guarantees strict positivity ($\mathbf{a} \equiv \boldsymbol{\mu} > \mathbf{0}$) at equilibrium and suppresses fringe negative-charge overshoot by **$>3\times$ ($0.2669 \to 0.0876$)** and $L_2$ profile error by **$2\times$ ($0.0326 \to 0.0163$)** under finite moment perturbations.

2. **Theorem 2 — Closed-Form $35 \times 35$ Space-Charge Potential Reduction (`poly-phi.pdf`)**:
   $$\mathbf{v} = \mathbf{K}\mathbf{s} + \mathbf{v}_0, \qquad \mathbf{K} = \mathbf{\Lambda}^{\prime T}(\boldsymbol{\mu})\mathbf{D}\mathbf{\Lambda} \in \mathbb{R}^{35 \times 35}$$
   - Eliminates the $N$-point spatial mesh offline and preserves all $M=35$ weighted potential moment averages to machine precision ($4.44 \times 10^{-16}$).

3. **Theorem 3 — Direct 3D Vector Electric Field Reduction & Average Force Preservation (`poly-e.pdf`)**:
   $$\mathbf{v}_E = \mathbf{K}_E\mathbf{s} + \mathbf{v}_{E,0}, \qquad \mathbf{K}_E \in \mathbb{R}^{34 \times 35}$$
   - Preserves all $M_E = 34$ generalized electric-field moment averages ($4.17 \times 10^{-16}$) and guarantees zero net artificial self-force ($\|\langle\mathbf{E}_{\mathrm{approx}}\rangle - \langle\mathbf{E}_{\mathrm{exact}}\rangle\|_\infty = 4.58 \times 10^{-16}$).
   - Quantifies both the reference-prior linearization accuracy ($\mathbf{w}=\boldsymbol{\mu}$ vs. $\mathbf{w}=\mathbf{a}$, $11.74\%$ quadrupole gradient difference) and interior beam-core multipole fit fidelity ($R^2 = 0.8962$, pointwise RMS relative error $32.22\%$ inside $r \le 0.45$).

4. **Non-Circular Multi-Turn 3D Symplectic Pairwise-Coulomb Tracking & 4-Mode Ablation (`simulate_multiturn_envelope_tracking`)**:
   - Tracks $N_p = 500$ macro-particles over $N_{\mathrm{turns}} = 80$ turns ($K_{\mathrm{sub}} = 4$ Leapfrog substeps/turn) from **strictly identical initial conditions** (zero initial rescaling) under **exact self-consistent $O(N_p^2)$ 3D pairwise Plummer-Coulomb space-charge forces** ($\Delta Q_{\mathrm{sc,eff}} = -0.1817$).
   - The feedback controller **never observes the pairwise Coulomb kicks**; it takes only the 35 spatial moments $\mathbf{s}$ and evaluates $\mathbf{v}_E = \mathbf{K}_E\mathbf{s} + \mathbf{v}_{E,0}$:

| Ablation Mode (Identical $t=0$ State) | Controller Input | RMS Emittance Growth | RMS Envelope Mismatch | High-Action Halo ($J_\perp > 3.5\sigma_0^2$) |
| :--- | :--- | :---: | :---: | :---: |
| **Mode A**: Open-Loop Uncompensated | None ($\mathbf{v}_E = \mathbf{0}$) | $+40.65\%$ | $16.60\%$ | $29.60\%$ |
| **Mode B**: Pure Moment Dipole+Quad+Oct Only | $11$ Multipoles of $\mathbf{K}_E\mathbf{s}+\mathbf{v}_{E,0}$ | $+27.51\%$ | $8.04\%$ | $16.40\%$ |
| **Mode C**: Pure Moment Full 34-Basis Field Only | All $34$ Terms of $\mathbf{K}_E\mathbf{s}+\mathbf{v}_{E,0}$ | $+16.44\%$ | $3.74\%$ | $17.60\%$ |
| **Mode D**: Full 34-Basis Moment + $\dot{s}_{200}$ Damper | $\mathbf{K}_E\mathbf{s}+\mathbf{v}_{E,0}$ + $\Delta s_{200},\Delta s_{020}$ | **$+7.00\%$ ($5.8\times$ lower)** | **$4.11\%$ ($4.0\times$ lower)** | **$12.20\%$ ($2.4\times$ lower)** |

---

## 2. Quick Start & Verification

```bash
python3 Test/benchmark_accelerator_space_charge.py
```
