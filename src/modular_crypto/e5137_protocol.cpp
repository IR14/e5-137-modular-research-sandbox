#include <cstdint>

namespace {

constexpr uint8_t kModulus = 137U;
constexpr int kAxes = 26;
constexpr int kCorrectableAxes = 10;
constexpr uint8_t kI5 = 42U;
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

uint8_t inverse_mod137(uint8_t value) {
    const int a = static_cast<int>(value % kModulus);
    if (a == 0) {
        return 0U;
    }
    for (int candidate = 1; candidate < kModulus; ++candidate) {
        if ((a * candidate) % kModulus == 1) {
            return static_cast<uint8_t>(candidate);
        }
    }
    return 0U;
}

uint8_t nonzero_residue(int value) {
    return static_cast<uint8_t>((value % 136 + 136) % 136 + 1);
}

uint8_t stream_mask(int index, int lane, const uint8_t* key26) {
    const int axis = (index + 7 * lane + kDFraction[index % kAxes]) % kAxes;
    const uint8_t inv = inverse_mod137(nonzero_residue((index + 1) * (axis + 1) + lane * kI5));
    return mod137(
        static_cast<int>(key26[axis]) +
        static_cast<int>(kDFraction[axis]) * static_cast<int>(kI5) +
        static_cast<int>(inv) +
        lane * 5 +
        axis
    );
}

uint8_t axis_mask(int axis, const uint8_t* key26) {
    const uint8_t inv = inverse_mod137(nonzero_residue((axis + 1) * kI5 + 5));
    return mod137(
        static_cast<int>(key26[axis]) +
        static_cast<int>(kDFraction[axis]) * static_cast<int>(kI5) +
        static_cast<int>(inv) +
        axis
    );
}

}  // namespace

extern "C" {

int e5137_axis_count() {
    return kAxes;
}

int e5137_correctable_axes() {
    return kCorrectableAxes;
}

uint8_t e5137_mod_inverse(uint8_t value) {
    return inverse_mod137(value);
}

void e5137_generate_key(const uint8_t* seed, int seed_len, uint8_t* key_out) {
    for (int axis = 0; axis < kAxes; ++axis) {
        int acc = static_cast<int>(kI5) + axis * 5 + static_cast<int>(kDFraction[axis]);
        for (int offset = 0; offset < seed_len; ++offset) {
            const uint8_t inv = inverse_mod137(nonzero_residue((offset + 1) * (axis + 1)));
            acc += static_cast<int>(seed[offset]) * static_cast<int>(inv);
            acc += static_cast<int>(kDFraction[(axis + offset) % kAxes]) * (offset + 1);
        }
        key_out[axis] = mod137(acc);
    }
}

void e5137_replicate_symbol(uint8_t symbol, const uint8_t* key26, uint8_t* shares_out) {
    const uint8_t reduced = static_cast<uint8_t>(symbol % kModulus);
    for (int axis = 0; axis < kAxes; ++axis) {
        shares_out[axis] = mod137(static_cast<int>(reduced) + static_cast<int>(axis_mask(axis, key26)));
    }
}

int e5137_recover_symbol(const uint8_t* shares26, const uint8_t* key26, uint8_t* symbol_out) {
    int counts[kModulus] = {0};
    for (int axis = 0; axis < kAxes; ++axis) {
        const uint8_t decoded = mod137(
            static_cast<int>(shares26[axis]) - static_cast<int>(axis_mask(axis, key26))
        );
        counts[decoded] += 1;
    }

    int best_symbol = 0;
    int best_count = counts[0];
    for (int symbol = 1; symbol < kModulus; ++symbol) {
        if (counts[symbol] > best_count) {
            best_symbol = symbol;
            best_count = counts[symbol];
        }
    }
    *symbol_out = static_cast<uint8_t>(best_symbol);
    return best_count;
}

void e5137_encrypt(const uint8_t* plaintext, int plaintext_len, const uint8_t* key26, uint8_t* cipher_out) {
    for (int index = 0; index < plaintext_len; ++index) {
        const uint8_t value = plaintext[index];
        const uint8_t low = static_cast<uint8_t>(value % kModulus);
        const uint8_t high = static_cast<uint8_t>(value / kModulus);
        cipher_out[2 * index] = mod137(
            static_cast<int>(low) + static_cast<int>(stream_mask(index, 0, key26))
        );
        cipher_out[2 * index + 1] = mod137(
            static_cast<int>(high) + static_cast<int>(stream_mask(index, 1, key26))
        );
    }
}

void e5137_decrypt(const uint8_t* cipher, int cipher_len, const uint8_t* key26, uint8_t* plaintext_out) {
    const int plaintext_len = cipher_len / 2;
    for (int index = 0; index < plaintext_len; ++index) {
        const uint8_t low = mod137(
            static_cast<int>(cipher[2 * index]) - static_cast<int>(stream_mask(index, 0, key26))
        );
        const uint8_t high = mod137(
            static_cast<int>(cipher[2 * index + 1]) - static_cast<int>(stream_mask(index, 1, key26))
        );
        plaintext_out[index] = static_cast<uint8_t>(low + high * kModulus);
    }
}

}  // extern "C"
