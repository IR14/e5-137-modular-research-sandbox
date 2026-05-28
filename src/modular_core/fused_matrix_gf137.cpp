#include <cstdint>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define GF137_USE_ARM_NEON 1
#else
#define GF137_USE_ARM_NEON 0
#endif

namespace {

constexpr uint32_t kModulus = 137U;
constexpr uint8_t kDefaultThreshold = 42U;
constexpr uint64_t kBarrettMu = (uint64_t{1} << 32) / kModulus;

inline uint8_t mod137_barrett(uint32_t value) {
    const uint32_t quotient = static_cast<uint32_t>(
        (static_cast<uint64_t>(value) * kBarrettMu) >> 32
    );
    uint32_t residue = value - quotient * kModulus;
    while (residue >= kModulus) {
        residue -= kModulus;
    }
    return static_cast<uint8_t>(residue);
}

inline uint8_t threshold_bit(uint8_t residue, uint8_t threshold) {
    return static_cast<uint8_t>(residue >= threshold ? 1U : 0U);
}

struct TwoLayerJob {
    const uint8_t* X = nullptr;
    const uint8_t* W1 = nullptr;
    const uint8_t* B1 = nullptr;
    const uint8_t* W2 = nullptr;
    const uint8_t* B2 = nullptr;
    uint8_t* Y = nullptr;
    int M = 0;
    int K = 0;
    int H = 0;
    uint8_t hidden_threshold = kDefaultThreshold;
    uint8_t output_threshold = kDefaultThreshold;
    int greater_is_prime = 1;
};

#if GF137_USE_ARM_NEON
inline uint16x8_t mod137_u16x8(uint16x8_t values) {
    const uint16x8_t modulus = vdupq_n_u16(137U);
    const uint32x4_t product_low = vmull_n_u16(vget_low_u16(values), 478U);
    const uint32x4_t product_high = vmull_n_u16(vget_high_u16(values), 478U);
    const uint16x8_t quotient = vcombine_u16(
        vshrn_n_u32(product_low, 16),
        vshrn_n_u32(product_high, 16)
    );
    uint16x8_t residue = vsubq_u16(values, vmulq_u16(quotient, modulus));
    const uint16x8_t ge_modulus = vcgeq_u16(residue, modulus);
    residue = vsubq_u16(residue, vandq_u16(ge_modulus, modulus));
    return residue;
}

inline uint32_t active_weight_sum_u16x8(
    uint16x8_t acc,
    const uint8_t* weights,
    uint8_t threshold
) {
    const uint16x8_t residue = mod137_u16x8(acc);
    const uint16x8_t active = vandq_u16(
        vcgeq_u16(residue, vdupq_n_u16(static_cast<uint16_t>(threshold))),
        vdupq_n_u16(1U)
    );
    const uint16x8_t w = vmovl_u8(vld1_u8(weights));
    return static_cast<uint32_t>(vaddvq_u16(vmulq_u16(active, w)));
}

inline void compute_two_layer_rows_neon32(
    const TwoLayerJob& job,
    int row_begin,
    int row_end
) {
    for (int i = row_begin; i < row_end; ++i) {
        const uint8_t* x_row = job.X + static_cast<int64_t>(i) * job.K;

        uint16x8_t acc0 = vmovl_u8(vld1_u8(job.B1));
        uint16x8_t acc1 = vmovl_u8(vld1_u8(job.B1 + 8));
        uint16x8_t acc2 = vmovl_u8(vld1_u8(job.B1 + 16));
        uint16x8_t acc3 = vmovl_u8(vld1_u8(job.B1 + 24));

        for (int k = 0; k < job.K; ++k) {
            const uint8_t x_value = x_row[k];
            if (x_value == 0U) {
                continue;
            }
            const uint8_t* w_row = job.W1 + static_cast<int64_t>(k) * 32;
            const uint16x8_t w0 = vmovl_u8(vld1_u8(w_row));
            const uint16x8_t w1 = vmovl_u8(vld1_u8(w_row + 8));
            const uint16x8_t w2 = vmovl_u8(vld1_u8(w_row + 16));
            const uint16x8_t w3 = vmovl_u8(vld1_u8(w_row + 24));
            if (x_value == 1U) {
                acc0 = vaddq_u16(acc0, w0);
                acc1 = vaddq_u16(acc1, w1);
                acc2 = vaddq_u16(acc2, w2);
                acc3 = vaddq_u16(acc3, w3);
            } else if (x_value == 2U) {
                acc0 = vaddq_u16(acc0, vshlq_n_u16(w0, 1));
                acc1 = vaddq_u16(acc1, vshlq_n_u16(w1, 1));
                acc2 = vaddq_u16(acc2, vshlq_n_u16(w2, 1));
                acc3 = vaddq_u16(acc3, vshlq_n_u16(w3, 1));
            } else {
                acc0 = vaddq_u16(acc0, vmulq_n_u16(w0, x_value));
                acc1 = vaddq_u16(acc1, vmulq_n_u16(w1, x_value));
                acc2 = vaddq_u16(acc2, vmulq_n_u16(w2, x_value));
                acc3 = vaddq_u16(acc3, vmulq_n_u16(w3, x_value));
            }
        }

        uint32_t out_acc = static_cast<uint32_t>(job.B2[0]);
        out_acc += active_weight_sum_u16x8(acc0, job.W2, job.hidden_threshold);
        out_acc += active_weight_sum_u16x8(acc1, job.W2 + 8, job.hidden_threshold);
        out_acc += active_weight_sum_u16x8(acc2, job.W2 + 16, job.hidden_threshold);
        out_acc += active_weight_sum_u16x8(acc3, job.W2 + 24, job.hidden_threshold);
        const bool above = mod137_barrett(out_acc) >= job.output_threshold;
        job.Y[i] = static_cast<uint8_t>((job.greater_is_prime ? above : !above) ? 1U : 0U);
    }
}
#endif

inline void compute_two_layer_rows(
    const TwoLayerJob& job,
    int row_begin,
    int row_end,
    std::vector<uint32_t>& hidden_acc,
    std::vector<uint8_t>& hidden
) {
#if GF137_USE_ARM_NEON
    if (job.H == 32) {
        compute_two_layer_rows_neon32(job, row_begin, row_end);
        return;
    }
#endif
    constexpr int kTileK = 16;
    for (int i = row_begin; i < row_end; ++i) {
        const uint8_t* x_row = job.X + static_cast<int64_t>(i) * job.K;

        for (int h = 0; h < job.H; ++h) {
            hidden_acc[static_cast<size_t>(h)] = static_cast<uint32_t>(job.B1[h]);
        }
        for (int k0 = 0; k0 < job.K; k0 += kTileK) {
            const int k_limit = (k0 + kTileK < job.K) ? (k0 + kTileK) : job.K;
            for (int k = k0; k < k_limit; ++k) {
                const uint32_t x_value = static_cast<uint32_t>(x_row[k]);
                if (x_value == 0U) {
                    continue;
                }
                const uint8_t* w_row = job.W1 + static_cast<int64_t>(k) * job.H;
                if (x_value == 1U) {
                    for (int h = 0; h < job.H; ++h) {
                        hidden_acc[static_cast<size_t>(h)] += static_cast<uint32_t>(w_row[h]);
                    }
                } else {
                    for (int h = 0; h < job.H; ++h) {
                        hidden_acc[static_cast<size_t>(h)] +=
                            2U * static_cast<uint32_t>(w_row[h]);
                    }
                }
            }
        }
        for (int h = 0; h < job.H; ++h) {
            hidden[static_cast<size_t>(h)] = threshold_bit(
                mod137_barrett(hidden_acc[static_cast<size_t>(h)]),
                job.hidden_threshold
            );
        }

        uint32_t out_acc = static_cast<uint32_t>(job.B2[0]);
        for (int h = 0; h < job.H; ++h) {
            out_acc += static_cast<uint32_t>(hidden[static_cast<size_t>(h)]) *
                       static_cast<uint32_t>(job.W2[h]);
        }
        const bool above = mod137_barrett(out_acc) >= job.output_threshold;
        job.Y[i] = static_cast<uint8_t>((job.greater_is_prime ? above : !above) ? 1U : 0U);
    }
}

inline void process_two_layer_chunks(
    const TwoLayerJob& job,
    std::atomic<int>& next_row,
    std::vector<uint32_t>& hidden_acc,
    std::vector<uint8_t>& hidden
) {
    constexpr int kRowChunk = 32;
    while (true) {
        const int row_begin = next_row.fetch_add(kRowChunk, std::memory_order_relaxed);
        if (row_begin >= job.M) {
            break;
        }
        const int row_end =
            (row_begin + kRowChunk < job.M) ? (row_begin + kRowChunk) : job.M;
        compute_two_layer_rows(job, row_begin, row_end, hidden_acc, hidden);
    }
}

class ThreadPool {
public:
    ThreadPool()
        : worker_count_(select_thread_count()), workers_(worker_count_) {
        for (int index = 0; index < worker_count_; ++index) {
            workers_[static_cast<size_t>(index)] = std::thread(&ThreadPool::worker_loop, this);
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            generation_ += 1U;
        }
        cv_start_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    int thread_count() const {
        return worker_count_ + 1;
    }

    void run_two_layer(const TwoLayerJob& job) {
        if (worker_count_ <= 0 || job.M < 256) {
            std::vector<uint32_t> hidden_acc(static_cast<size_t>(job.H));
            std::vector<uint8_t> hidden(static_cast<size_t>(job.H));
            compute_two_layer_rows(job, 0, job.M, hidden_acc, hidden);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = job;
            next_row_.store(0, std::memory_order_relaxed);
            active_workers_ = worker_count_;
            generation_ += 1U;
        }
        cv_start_.notify_all();

        std::vector<uint32_t> hidden_acc(static_cast<size_t>(job.H));
        std::vector<uint8_t> hidden(static_cast<size_t>(job.H));
        process_two_layer_chunks(job, next_row_, hidden_acc, hidden);

        std::unique_lock<std::mutex> lock(mutex_);
        cv_done_.wait(lock, [this] { return active_workers_ == 0; });
    }

private:
    static int select_thread_count() {
        const unsigned int hardware = std::thread::hardware_concurrency();
        if (hardware <= 1U) {
            return 0;
        }
        const unsigned int worker_count = hardware - 1U;
        return worker_count > 3U ? 3 : static_cast<int>(worker_count);
    }

    void worker_loop() {
        std::vector<uint32_t> hidden_acc;
        std::vector<uint8_t> hidden;
        uint64_t seen_generation = 0U;
        while (true) {
            TwoLayerJob local_job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_start_.wait(lock, [this, seen_generation] {
                    return stopping_ || generation_ != seen_generation;
                });
                if (stopping_) {
                    return;
                }
                seen_generation = generation_;
                local_job = job_;
            }

            hidden_acc.resize(static_cast<size_t>(local_job.H));
            hidden.resize(static_cast<size_t>(local_job.H));
            process_two_layer_chunks(local_job, next_row_, hidden_acc, hidden);

            {
                std::lock_guard<std::mutex> lock(mutex_);
                active_workers_ -= 1;
                if (active_workers_ == 0) {
                    cv_done_.notify_one();
                }
            }
        }
    }

    const int worker_count_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    std::atomic<int> next_row_{0};
    TwoLayerJob job_;
    uint64_t generation_ = 0U;
    int active_workers_ = 0;
    bool stopping_ = false;
};

ThreadPool& thread_pool() {
    static ThreadPool pool;
    return pool;
}

}  // namespace

extern "C" {

void fused_matmul_mod137_threshold(
    const uint8_t* X,
    const uint8_t* W,
    uint8_t* Y,
    int M,
    int K,
    int N
) {
    std::vector<uint32_t> acc(static_cast<size_t>(N));
    for (int i = 0; i < M; ++i) {
        const uint8_t* x_row = X + static_cast<int64_t>(i) * K;
        uint8_t* y_row = Y + static_cast<int64_t>(i) * N;

        for (int j = 0; j < N; ++j) {
            acc[static_cast<size_t>(j)] = 0U;
        }
        for (int k = 0; k < K; ++k) {
            const uint32_t x_value = static_cast<uint32_t>(x_row[k]);
            const uint8_t* w_row = W + static_cast<int64_t>(k) * N;
            for (int j = 0; j < N; ++j) {
                acc[static_cast<size_t>(j)] += x_value * static_cast<uint32_t>(w_row[j]);
            }
        }
        for (int j = 0; j < N; ++j) {
            y_row[j] = threshold_bit(
                mod137_barrett(acc[static_cast<size_t>(j)]),
                kDefaultThreshold
            );
        }
    }
}

void fused_matmul_mod137_bias_threshold(
    const uint8_t* X,
    const uint8_t* W,
    const uint8_t* B,
    uint8_t* Y,
    int M,
    int K,
    int N,
    uint8_t threshold
) {
    std::vector<uint32_t> acc(static_cast<size_t>(N));
    for (int i = 0; i < M; ++i) {
        const uint8_t* x_row = X + static_cast<int64_t>(i) * K;
        uint8_t* y_row = Y + static_cast<int64_t>(i) * N;

        for (int j = 0; j < N; ++j) {
            acc[static_cast<size_t>(j)] = static_cast<uint32_t>(B[j]);
        }
        for (int k = 0; k < K; ++k) {
            const uint32_t x_value = static_cast<uint32_t>(x_row[k]);
            const uint8_t* w_row = W + static_cast<int64_t>(k) * N;
            for (int j = 0; j < N; ++j) {
                acc[static_cast<size_t>(j)] += x_value * static_cast<uint32_t>(w_row[j]);
            }
        }
        for (int j = 0; j < N; ++j) {
            y_row[j] = threshold_bit(mod137_barrett(acc[static_cast<size_t>(j)]), threshold);
        }
    }
}

void fused_two_layer_mod137_predict(
    const uint8_t* X,
    const uint8_t* W1,
    const uint8_t* B1,
    const uint8_t* W2,
    const uint8_t* B2,
    uint8_t* Y,
    int M,
    int K,
    int H,
    uint8_t hidden_threshold,
    uint8_t output_threshold,
    int greater_is_prime
) {
    const TwoLayerJob job{
        X,
        W1,
        B1,
        W2,
        B2,
        Y,
        M,
        K,
        H,
        hidden_threshold,
        output_threshold,
        greater_is_prime,
    };
    thread_pool().run_two_layer(job);
}

int gf137_thread_count() {
    return thread_pool().thread_count();
}

void fused_two_layer_mod137_predict_repeated(
    const uint8_t* X,
    const uint8_t* W1,
    const uint8_t* B1,
    const uint8_t* W2,
    const uint8_t* B2,
    uint8_t* Y,
    int M,
    int K,
    int H,
    uint8_t hidden_threshold,
    uint8_t output_threshold,
    int greater_is_prime,
    int iterations
) {
    if (iterations <= 0) {
        return;
    }
    const TwoLayerJob job{
        X,
        W1,
        B1,
        W2,
        B2,
        Y,
        M,
        K,
        H,
        hidden_threshold,
        output_threshold,
        greater_is_prime,
    };
    const int lanes = thread_pool().thread_count();
    if (lanes <= 1 || M < 256) {
        std::vector<uint32_t> hidden_acc(static_cast<size_t>(H));
        std::vector<uint8_t> hidden(static_cast<size_t>(H));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            compute_two_layer_rows(job, 0, M, hidden_acc, hidden);
        }
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(lanes > 1 ? lanes - 1 : 0));
    auto worker = [&](int lane) {
        const int row_begin = (M * lane) / lanes;
        const int row_end = (M * (lane + 1)) / lanes;
        std::vector<uint32_t> hidden_acc(static_cast<size_t>(H));
        std::vector<uint8_t> hidden(static_cast<size_t>(H));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            compute_two_layer_rows(job, row_begin, row_end, hidden_acc, hidden);
        }
    };

    for (int lane = 1; lane < lanes; ++lane) {
        workers.emplace_back(worker, lane);
    }
    worker(0);
    for (std::thread& thread : workers) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

}  // extern "C"
