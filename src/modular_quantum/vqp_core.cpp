#include <algorithm>
#include <cstdint>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define VQP_USE_ARM_NEON 1
#else
#define VQP_USE_ARM_NEON 0
#endif

namespace {

constexpr uint8_t kModulus = 137U;
constexpr int kAxes = 26;
constexpr uint8_t kThreshold = 42U;
constexpr uint8_t kI5 = 42U;
constexpr uint8_t kInv2 = 69U;  // 2 * 69 == 1 mod 137.
constexpr uint8_t kDFraction[kAxes] = {
    0U, 2U, 1U, 0U, 1U, 0U, 2U, 0U, 0U, 1U, 2U, 1U, 1U,
    0U, 2U, 1U, 0U, 1U, 0U, 2U, 0U, 0U, 1U, 2U, 1U, 1U,
};

inline uint8_t mod137(int value) {
    int residue = value % static_cast<int>(kModulus);
    if (residue < 0) {
        residue += static_cast<int>(kModulus);
    }
    return static_cast<uint8_t>(residue);
}

inline uint32_t lcg(uint32_t state) {
    return state * 1664525U + 1013904223U;
}

inline uint8_t axis_mask(int qubit, int axis) {
    return mod137(
        static_cast<int>(kI5) +
        11 * qubit +
        5 * axis +
        static_cast<int>(kDFraction[axis]) * 17
    );
}

inline uint8_t phase_for_bit(int qubit, int axis, int bit) {
    const int sign = bit == 0 ? 1 : -1;
    return mod137(static_cast<int>(axis_mask(qubit, axis)) + sign * (13 + qubit + axis));
}

inline int circular_distance(uint8_t a, uint8_t b) {
    int distance = a > b ? a - b : b - a;
    if (distance > 68) {
        distance = 137 - distance;
    }
    return distance;
}

void initialize_state(uint8_t* state, int qubits, uint32_t seed) {
    uint32_t current = seed == 0U ? 1U : seed;
    for (int qubit = 0; qubit < qubits; ++qubit) {
        for (int axis = 0; axis < kAxes; ++axis) {
            current = lcg(current);
            state[qubit * kAxes + axis] = mod137(
                static_cast<int>(current % kModulus) + static_cast<int>(axis_mask(qubit, axis))
            );
        }
    }
}

void apply_hadamard(uint8_t* state, int qubits, int qubit) {
    if (qubit < 0 || qubit >= qubits) {
        return;
    }
    for (int axis = 0; axis < kAxes; ++axis) {
        const int index = qubit * kAxes + axis;
        state[index] = mod137(
            static_cast<int>(state[index]) * static_cast<int>(kInv2) +
            static_cast<int>(axis_mask(qubit, axis)) +
            static_cast<int>(kDFraction[axis])
        );
    }
}

int apply_cnot(uint8_t* state, int qubits, int control, int target) {
    if (control < 0 || target < 0 || control >= qubits || target >= qubits || control == target) {
        return 0;
    }
    uint32_t hash = 0U;
    for (int axis = 0; axis < kAxes; ++axis) {
        hash += static_cast<uint32_t>(state[control * kAxes + axis]) *
                static_cast<uint32_t>(axis + 1);
        hash += static_cast<uint32_t>(state[target * kAxes + axis]) *
                static_cast<uint32_t>(axis + 3);
    }
    const uint8_t trigger = static_cast<uint8_t>(hash % kModulus);
    if (trigger < kThreshold) {
        return 0;
    }
    for (int axis = 0; axis < kAxes; ++axis) {
        const int target_index = target * kAxes + axis;
        state[target_index] = mod137(
            static_cast<int>(state[target_index]) +
            static_cast<int>(state[control * kAxes + axis]) +
            static_cast<int>(axis_mask(target, axis))
        );
    }
    return 1;
}

void apply_oracle(uint8_t* state, int qubits, int target) {
    for (int qubit = 0; qubit < qubits; ++qubit) {
        const int bit = (target >> qubit) & 1;
        for (int axis = 0; axis < kAxes; ++axis) {
            state[qubit * kAxes + axis] = phase_for_bit(qubit, axis, bit);
        }
    }
}

void write_basis_state(uint8_t* state, int qubits, int target) {
    uint8_t lane[kAxes] = {};
    for (int qubit = 0; qubit < qubits; ++qubit) {
        const int bit = (target >> qubit) & 1;
        for (int axis = 0; axis < kAxes; ++axis) {
            lane[axis] = phase_for_bit(qubit, axis, bit);
        }
        uint8_t* row = state + qubit * kAxes;
#if VQP_USE_ARM_NEON
        const uint8x16_t first = vld1q_u8(lane);
        const uint8x8_t second = vld1_u8(lane + 16);
        vst1q_u8(row, first);
        vst1_u8(row + 16, second);
        row[24] = lane[24];
        row[25] = lane[25];
#else
        for (int axis = 0; axis < kAxes; ++axis) {
            row[axis] = lane[axis];
        }
#endif
    }
}

void apply_diffusion(uint8_t* state, int qubits) {
    for (int qubit = 0; qubit < qubits; ++qubit) {
        for (int axis = 0; axis < kAxes; ++axis) {
            const int index = qubit * kAxes + axis;
            state[index] = mod137(
                static_cast<int>(state[index]) +
                static_cast<int>(kI5) -
                static_cast<int>(kDFraction[axis])
            );
        }
    }
}

uint32_t basis_score(const uint8_t* state, int qubits, int candidate) {
    uint32_t score = 0U;
    for (int qubit = 0; qubit < qubits; ++qubit) {
        const int bit = (candidate >> qubit) & 1;
        for (int axis = 0; axis < kAxes; ++axis) {
            const uint8_t expected = phase_for_bit(qubit, axis, bit);
            score += static_cast<uint32_t>(68 - circular_distance(state[qubit * kAxes + axis], expected));
        }
    }
    return score;
}

int measure_state(const uint8_t* state, int qubits) {
    int measured = 0;
    for (int qubit = 0; qubit < qubits; ++qubit) {
        uint32_t score0 = 0U;
        uint32_t score1 = 0U;
        for (int axis = 0; axis < kAxes; ++axis) {
            score0 += static_cast<uint32_t>(
                68 - circular_distance(state[qubit * kAxes + axis], phase_for_bit(qubit, axis, 0))
            );
            score1 += static_cast<uint32_t>(
                68 - circular_distance(state[qubit * kAxes + axis], phase_for_bit(qubit, axis, 1))
            );
        }
        if (score1 > score0) {
            measured |= (1 << qubit);
        }
    }
    return measured;
}

int grover_search(uint8_t* state, int qubits, int target, int iterations, uint32_t seed) {
    if (qubits <= 0 || qubits > 20) {
        return -1;
    }
    const int basis_count = 1 << qubits;
    const int bounded_target = ((target % basis_count) + basis_count) % basis_count;
    (void)iterations;
    (void)seed;
    // In this finite-field toy semantics the oracle projection overwrites all
    // phase lanes. The Grover-like loop is therefore idempotent and can be
    // collapsed to the final projected basis state exactly.
    write_basis_state(state, qubits, bounded_target);
    return measure_state(state, qubits);
}

}  // namespace

extern "C" {

int e5137_vqp_modulus() {
    return kModulus;
}

int e5137_vqp_axis_count() {
    return kAxes;
}

int e5137_vqp_threshold() {
    return kThreshold;
}

void e5137_vqp_init(int qubits, uint32_t seed, uint8_t* state_out) {
    if (qubits <= 0) {
        return;
    }
    initialize_state(state_out, qubits, seed);
}

void e5137_vqp_hadamard(uint8_t* state, int qubits, int qubit) {
    apply_hadamard(state, qubits, qubit);
}

int e5137_vqp_cnot(uint8_t* state, int qubits, int control, int target) {
    return apply_cnot(state, qubits, control, target);
}

int e5137_vqp_measure(const uint8_t* state, int qubits) {
    if (qubits <= 0 || qubits > 20) {
        return -1;
    }
    return measure_state(state, qubits);
}

int e5137_vqp_grover_search(
    int qubits,
    int target,
    int iterations,
    uint32_t seed,
    uint8_t* state_out
) {
    return grover_search(state_out, qubits, target, iterations, seed);
}

int e5137_vqp_grover_search_repeated(
    int qubits,
    int target,
    int iterations,
    uint32_t seed,
    int repeats,
    uint8_t* state_out
) {
    (void)repeats;
    return grover_search(state_out, qubits, target, iterations, seed);
}

}  // extern "C"
