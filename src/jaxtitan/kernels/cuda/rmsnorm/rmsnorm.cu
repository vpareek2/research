#include "kittens.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

using namespace kittens;

static constexpr size_t kHidden = 1024;
static constexpr int kThreads = kittens::WARP_THREADS;

struct RmsNormGlobals {
    // TK global descriptors need the compile-time row tile type; this POC
    // intentionally supports one hidden size so shape mistakes fail early.
    using row_vec = sv_bf<kHidden>;
    using x_gl = gl<bf16, -1, -1, -1, -1, row_vec>;
    using weight_gl = gl<bf16, -1, -1, -1, -1, row_vec>;
    using out_gl = gl<bf16, -1, -1, -1, -1, row_vec>;

    x_gl x;
    weight_gl weight;
    out_gl out;
    float eps;
};

__global__ __launch_bounds__(kThreads, 1)
void rmsnorm_kernel(const __grid_constant__ RmsNormGlobals g) {
    const int row = blockIdx.x;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator allocator(reinterpret_cast<int *>(&__shm[0]));

    RmsNormGlobals::row_vec &x_s = allocator.allocate<RmsNormGlobals::row_vec>();
    RmsNormGlobals::row_vec &weight_s = allocator.allocate<RmsNormGlobals::row_vec>();

    warp::load(x_s, g.x, {row, 0, 0, 0});
    warp::load(weight_s, g.weight, {0, 0, 0, 0});

    // One CUDA block owns one logical row. Lanes cooperatively reduce the
    // squared norm in fp32, then reuse the same shared vector for output.
    float sum_squares = 0.0f;
    for (int col = laneid(); col < static_cast<int>(kHidden); col += WARP_THREADS) {
        const float value = __bfloat162float(x_s[col]);
        sum_squares += value * value;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_squares += __shfl_down_sync(0xffffffff, sum_squares, offset);
    }
    sum_squares = __shfl_sync(0xffffffff, sum_squares, 0);

    const float inv_rms = rsqrtf(sum_squares / static_cast<float>(kHidden) + g.eps);

    for (int col = laneid(); col < static_cast<int>(kHidden); col += WARP_THREADS) {
        const float x_value = __bfloat162float(x_s[col]);
        const float w_value = __bfloat162float(weight_s[col]);
        x_s[col] = __float2bfloat16(x_value * inv_rms * w_value);
    }

    warp::store(g.out, x_s, {row, 0, 0, 0});
}

void dispatch_rmsnorm(
    bf16 *x,
    bf16 *weight,
    bf16 *out,
    float eps,
    size_t rows
) {
    using row_vec = RmsNormGlobals::row_vec;
    using x_gl = RmsNormGlobals::x_gl;
    using weight_gl = RmsNormGlobals::weight_gl;
    using out_gl = RmsNormGlobals::out_gl;

    // Treat the input as [rows, hidden] by mapping rows onto the first TK
    // descriptor axis. The future JAX wrapper should guard this layout.
    x_gl x_arg{x, rows, 1, 1, kHidden};
    weight_gl weight_arg{weight, 1, 1, 1, kHidden};
    out_gl out_arg{out, rows, 1, 1, kHidden};

    RmsNormGlobals globals{x_arg, weight_arg, out_arg, eps};

    const unsigned long shared_bytes = 2 * sizeof(row_vec);
    dim3 grid(static_cast<unsigned int>(rows), 1, 1);
    dim3 block(kThreads, 1, 1);

    rmsnorm_kernel<<<grid, block, shared_bytes>>>(globals);
}
