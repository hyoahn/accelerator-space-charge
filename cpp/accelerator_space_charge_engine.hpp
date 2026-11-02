// Copyright (c) 2026 Dr. Hyo E. Ahn (ahn@aya.yale.edu)
// C++20 Deterministic 35x35 Potential & 34x35 3D Electric Field Space-Charge Engine
#pragma once
#include <array>
#include <cstddef>

namespace lanl::accelerator {

constexpr std::size_t M_POTENTIAL = 35;
constexpr std::size_t M_EFIELD = 34;

struct SpaceChargeMomentOperator {
    std::array<std::array<double, M_POTENTIAL>, M_POTENTIAL> K_pot{};
    std::array<double, M_POTENTIAL> v0_pot{};
    std::array<std::array<double, M_POTENTIAL>, M_EFIELD> K_field{};
    std::array<double, M_EFIELD> v0_field{};

    [[nodiscard]] constexpr std::array<double, M_POTENTIAL>
    reconstruct_potential(const std::array<double, M_POTENTIAL>& s) const noexcept {
        std::array<double, M_POTENTIAL> v = v0_pot;
        for (std::size_t i = 0; i < M_POTENTIAL; ++i) {
            double acc = v[i];
            for (std::size_t j = 0; j < M_POTENTIAL; ++j) {
                acc += K_pot[i][j] * s[j];
            }
            v[i] = acc;
        }
        return v;
    }

    [[nodiscard]] constexpr std::array<double, M_EFIELD>
    reconstruct_electric_field(const std::array<double, M_POTENTIAL>& s) const noexcept {
        std::array<double, M_EFIELD> v_e = v0_field;
        for (std::size_t i = 0; i < M_EFIELD; ++i) {
            double acc = v_e[i];
            for (std::size_t j = 0; j < M_POTENTIAL; ++j) {
                acc += K_field[i][j] * s[j];
            }
            v_e[i] = acc;
        }
        return v_e;
    }
};

} // namespace lanl::accelerator
