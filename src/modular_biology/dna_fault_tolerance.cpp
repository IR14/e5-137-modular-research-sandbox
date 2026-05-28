#include <cstdint>
#include <vector>

namespace {

constexpr uint8_t kModulus = 137U;
constexpr int kAxes = 26;
constexpr int kCorrectionBudget = 10;
constexpr int kSemanticReferenceTokens = 120000;
constexpr int kSemanticReferenceVectors = 117;
constexpr uint8_t kI5 = 42U;
constexpr uint8_t kDeltaPhiCode = 59U;
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

int base_code(char base) {
    switch (base) {
        case 'A':
        case 'a':
            return 0;
        case 'T':
        case 't':
            return 1;
        case 'G':
        case 'g':
            return 2;
        case 'C':
        case 'c':
            return 3;
        default:
            return -1;
    }
}

char code_base(uint8_t code) {
    switch (code % 4U) {
        case 0:
            return 'A';
        case 1:
            return 'T';
        case 2:
            return 'G';
        default:
            return 'C';
    }
}

uint8_t base_primary_residue(uint8_t code) {
    return mod137(static_cast<int>(kI5) + 5 * static_cast<int>(code));
}

uint8_t base_bond_residue(uint8_t code) {
    const int hydrogen_bonds = (code <= 1U) ? 2 : 3;
    return mod137(hydrogen_bonds * static_cast<int>(kDeltaPhiCode) + 13 * static_cast<int>(code));
}

uint8_t code_from_primary(uint8_t residue) {
    int best_code = 0;
    int best_distance = 1000;
    for (int code = 0; code < 4; ++code) {
        const int target = static_cast<int>(base_primary_residue(static_cast<uint8_t>(code)));
        int distance = residue > target ? residue - target : target - residue;
        if (distance > 68) {
            distance = 137 - distance;
        }
        if (distance < best_distance) {
            best_distance = distance;
            best_code = code;
        }
    }
    return static_cast<uint8_t>(best_code);
}

uint8_t axis_mask(int axis, int position) {
    return mod137(
        static_cast<int>(kI5) +
        static_cast<int>(kDFraction[axis]) * static_cast<int>(kDeltaPhiCode) +
        axis * 5 +
        position * 3
    );
}

void replicate_code(uint8_t code, int position, uint8_t* shares_out) {
    const uint8_t primary = base_primary_residue(code);
    for (int axis = 0; axis < kAxes; ++axis) {
        shares_out[axis] = mod137(static_cast<int>(primary) + static_cast<int>(axis_mask(axis, position)));
    }
}

uint8_t recover_code(const uint8_t* shares, int position, int* votes_out) {
    int counts[4] = {0, 0, 0, 0};
    for (int axis = 0; axis < kAxes; ++axis) {
        const uint8_t decoded_primary = mod137(
            static_cast<int>(shares[axis]) - static_cast<int>(axis_mask(axis, position))
        );
        counts[code_from_primary(decoded_primary)] += 1;
    }

    int best_code = 0;
    int best_count = counts[0];
    for (int code = 1; code < 4; ++code) {
        if (counts[code] > best_count) {
            best_code = code;
            best_count = counts[code];
        }
    }
    *votes_out = best_count;
    return static_cast<uint8_t>(best_code);
}

void corrupt_random_axes(uint8_t* shares, int position, int damage_axes, uint32_t* state) {
    bool used[kAxes] = {};
    int corrupted = 0;
    while (corrupted < damage_axes && corrupted < kAxes) {
        *state = lcg(*state);
        const int axis = static_cast<int>(*state % kAxes);
        if (used[axis]) {
            continue;
        }
        used[axis] = true;
        *state = lcg(*state);
        const uint8_t delta = static_cast<uint8_t>((*state % 136U) + 1U);
        shares[position * kAxes + axis] = mod137(
            static_cast<int>(shares[position * kAxes + axis]) + static_cast<int>(delta)
        );
        corrupted += 1;
    }
}

char semantic_context_base(int index) {
    const int selector = index * 5 + static_cast<int>(kDFraction[index % kAxes]) + 2;
    return code_base(static_cast<uint8_t>(selector));
}

void build_semantic_context(int len, char* context_out) {
    for (int index = 0; index < len; ++index) {
        context_out[index] = semantic_context_base(index);
    }
}

}  // namespace

extern "C" {

int e5137_dna_axis_count() {
    return kAxes;
}

int e5137_dna_correction_budget() {
    return kCorrectionBudget;
}

int e5137_semantic_reference_token_count() {
    return kSemanticReferenceTokens;
}

int e5137_semantic_reference_vector_count() {
    return kSemanticReferenceVectors;
}

int e5137_semantic_vector_count_for_tokens(int token_count) {
    if (token_count <= 0) {
        return 0;
    }
    const int64_t numerator =
        static_cast<int64_t>(token_count) * static_cast<int64_t>(kSemanticReferenceVectors);
    return static_cast<int>((numerator + kSemanticReferenceTokens - 1) / kSemanticReferenceTokens);
}

void e5137_semantic_build_context(int vector_count, char* context_out) {
    if (vector_count <= 0) {
        return;
    }
    build_semantic_context(vector_count, context_out);
}

int e5137_dna_base_code(char base) {
    return base_code(base);
}

char e5137_dna_code_base(uint8_t code) {
    return code_base(code);
}

void e5137_dna_encode(const char* dna, int len, uint8_t* residues_out) {
    for (int index = 0; index < len; ++index) {
        const int code = base_code(dna[index]);
        const uint8_t safe_code = code < 0 ? 0U : static_cast<uint8_t>(code);
        residues_out[2 * index] = base_primary_residue(safe_code);
        residues_out[2 * index + 1] = base_bond_residue(safe_code);
    }
}

int e5137_dna_decode(const uint8_t* residues, int residue_len, char* dna_out) {
    if (residue_len % 2 != 0) {
        return 0;
    }
    const int len = residue_len / 2;
    for (int index = 0; index < len; ++index) {
        dna_out[index] = code_base(code_from_primary(residues[2 * index]));
    }
    return len;
}

int e5137_dna_replicate(const char* dna, int len, uint8_t* shares_out) {
    for (int index = 0; index < len; ++index) {
        const int code = base_code(dna[index]);
        if (code < 0) {
            return 0;
        }
        replicate_code(static_cast<uint8_t>(code), index, shares_out + index * kAxes);
    }
    return 1;
}

int e5137_dna_repair(const uint8_t* shares, int len, char* dna_out, int* votes_out) {
    int min_votes = kAxes;
    for (int index = 0; index < len; ++index) {
        int votes = 0;
        const uint8_t code = recover_code(shares + index * kAxes, index, &votes);
        dna_out[index] = code_base(code);
        votes_out[index] = votes;
        if (votes < min_votes) {
            min_votes = votes;
        }
    }
    return min_votes;
}

int e5137_dna_simulate(
    const char* dna,
    int len,
    int cycles,
    int damage_axes,
    uint32_t seed,
    char* final_out,
    int* failed_cycles_out
) {
    std::vector<char> current(static_cast<size_t>(len));
    std::vector<char> repaired(static_cast<size_t>(len));
    std::vector<uint8_t> shares(static_cast<size_t>(len * kAxes));
    std::vector<int> votes(static_cast<size_t>(len));
    for (int index = 0; index < len; ++index) {
        if (base_code(dna[index]) < 0) {
            return 0;
        }
        current[static_cast<size_t>(index)] = code_base(static_cast<uint8_t>(base_code(dna[index])));
    }

    int failed_cycles = 0;
    uint32_t state = seed == 0U ? 1U : seed;
    for (int cycle = 0; cycle < cycles; ++cycle) {
        if (!e5137_dna_replicate(current.data(), len, shares.data())) {
            return 0;
        }
        const int bounded_damage = damage_axes < 0 ? 0 : (damage_axes > kAxes ? kAxes : damage_axes);
        for (int position = 0; position < len; ++position) {
            corrupt_random_axes(shares.data(), position, bounded_damage, &state);
        }
        e5137_dna_repair(shares.data(), len, repaired.data(), votes.data());
        bool failed = false;
        for (int index = 0; index < len; ++index) {
            if (repaired[static_cast<size_t>(index)] != current[static_cast<size_t>(index)]) {
                failed = true;
            }
            current[static_cast<size_t>(index)] = repaired[static_cast<size_t>(index)];
        }
        if (failed) {
            failed_cycles += 1;
        }
    }

    for (int index = 0; index < len; ++index) {
        final_out[index] = current[static_cast<size_t>(index)];
    }
    *failed_cycles_out = failed_cycles;
    return 1;
}

int e5137_semantic_simulate_hobbit_context(
    int cycles,
    int damage_axes,
    uint32_t seed,
    char* final_out,
    int* failed_cycles_out
) {
    std::vector<char> context(static_cast<size_t>(kSemanticReferenceVectors));
    build_semantic_context(kSemanticReferenceVectors, context.data());
    return e5137_dna_simulate(
        context.data(),
        kSemanticReferenceVectors,
        cycles,
        damage_axes,
        seed,
        final_out,
        failed_cycles_out
    );
}

}  // extern "C"
