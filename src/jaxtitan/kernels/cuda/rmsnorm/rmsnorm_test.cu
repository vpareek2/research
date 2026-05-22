#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using bf16 = __nv_bfloat16;

static constexpr size_t kHidden = 1024;

void dispatch_rmsnorm(
    bf16 *x,
    bf16 *weight,
    bf16 *out,
    float eps,
    size_t rows
);

static void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(
            stderr,
            "CUDA error at %s:%d: %s failed: %s\n",
            file,
            line,
            expr,
            cudaGetErrorString(status)
        );
        std::exit(1);
    }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

static float make_x(size_t row, size_t col) {
    const int raw = static_cast<int>((row * 37 + col * 17) % 251) - 125;
    return static_cast<float>(raw) / 31.0f;
}

static float make_weight(size_t col) {
    const int raw = static_cast<int>((col * 13) % 97) - 48;
    return 1.0f + static_cast<float>(raw) / 257.0f;
}

static std::vector<bf16> rmsnorm_reference(
    const std::vector<bf16> &x,
    const std::vector<bf16> &weight,
    float eps,
    size_t rows
) {
    std::vector<bf16> ref(rows * kHidden);
    for (size_t row = 0; row < rows; ++row) {
        float sum_squares = 0.0f;
        for (size_t col = 0; col < kHidden; ++col) {
            const float value = __bfloat162float(x[row * kHidden + col]);
            sum_squares += value * value;
        }

        const float inv_rms = rsqrtf(sum_squares / static_cast<float>(kHidden) + eps);
        for (size_t col = 0; col < kHidden; ++col) {
            const float x_value = __bfloat162float(x[row * kHidden + col]);
            const float weight_value = __bfloat162float(weight[col]);
            ref[row * kHidden + col] = __float2bfloat16(x_value * inv_rms * weight_value);
        }
    }
    return ref;
}

static float run_case(size_t rows) {
    constexpr float eps = 1.0e-6f;
    const size_t elements = rows * kHidden;

    std::vector<float> x(elements);
    std::vector<float> weight(kHidden);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < kHidden; ++col) {
            x[row * kHidden + col] = make_x(row, col);
        }
    }
    for (size_t col = 0; col < kHidden; ++col) {
        weight[col] = make_weight(col);
    }

    std::vector<bf16> x_bf(elements);
    std::vector<bf16> weight_bf(kHidden);
    std::vector<bf16> out_bf(elements);
    for (size_t i = 0; i < elements; ++i) {
        x_bf[i] = __float2bfloat16(x[i]);
    }
    for (size_t i = 0; i < kHidden; ++i) {
        weight_bf[i] = __float2bfloat16(weight[i]);
    }

    bf16 *d_x = nullptr;
    bf16 *d_weight = nullptr;
    bf16 *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, elements * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&d_weight, kHidden * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&d_out, elements * sizeof(bf16)));
    CHECK_CUDA(cudaMemcpy(d_x, x_bf.data(), elements * sizeof(bf16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, weight_bf.data(), kHidden * sizeof(bf16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0, elements * sizeof(bf16)));

    dispatch_rmsnorm(d_x, d_weight, d_out, eps, rows);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(out_bf.data(), d_out, elements * sizeof(bf16), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_out));

    const std::vector<bf16> ref = rmsnorm_reference(x_bf, weight_bf, eps, rows);
    float max_abs_error = 0.0f;
    for (size_t i = 0; i < elements; ++i) {
        const float got = __bfloat162float(out_bf[i]);
        const float expected = __bfloat162float(ref[i]);
        max_abs_error = std::max(max_abs_error, std::abs(got - expected));
    }
    return max_abs_error;
}

int main() {
    constexpr float tolerance = 1.0e-2f;
    const size_t cases[] = {1, 4, 17};

    bool passed = true;
    for (size_t rows : cases) {
        const float max_abs_error = run_case(rows);
        std::printf(
            "case rows=%zu hidden=%zu max_abs_error=%.8f tolerance=%.8f\n",
            rows,
            kHidden,
            max_abs_error,
            tolerance
        );
        passed = passed && max_abs_error <= tolerance;
    }

    if (!passed) {
        std::fprintf(stderr, "FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
