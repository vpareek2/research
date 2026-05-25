#include "xla/ffi/api/ffi.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ffi = xla::ffi;

using bf16 = __nv_bfloat16;

static constexpr size_t kHidden = 1024;

void dispatch_rmsnorm_stream(
    bf16 *x,
    bf16 *weight,
    bf16 *out,
    float eps,
    size_t rows,
    cudaStream_t stream
);

static ffi::Error rmsnorm_bf16_h1024(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> x,
    ffi::Buffer<ffi::BF16> weight,
    ffi::ResultBuffer<ffi::BF16> out,
    double eps
) {
    const auto x_dims = x.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto out_dims = out->dimensions();

    if (x_dims.size() < 2) {
        return ffi::Error::InvalidArgument("rmsnorm input rank must be at least 2");
    }
    if (x_dims.back() != static_cast<int64_t>(kHidden)) {
        return ffi::Error::InvalidArgument("rmsnorm hidden dimension must be 1024");
    }
    if (weight_dims.size() != 1 || weight_dims[0] != static_cast<int64_t>(kHidden)) {
        return ffi::Error::InvalidArgument("rmsnorm weight shape must be [1024]");
    }
    if (out_dims.size() != x_dims.size()) {
        return ffi::Error::InvalidArgument("rmsnorm output rank must match input rank");
    }
    for (size_t i = 0; i < x_dims.size(); ++i) {
        if (out_dims[i] != x_dims[i]) {
            return ffi::Error::InvalidArgument("rmsnorm output shape must match input shape");
        }
    }

    const size_t elements = x.element_count();
    if (elements % kHidden != 0) {
        return ffi::Error::InvalidArgument("rmsnorm element count must be divisible by 1024");
    }
    const size_t rows = elements / kHidden;

    // XLA owns allocation and scheduling. The handler only launches into the
    // stream supplied by the runtime, so JAX can order this with surrounding ops.
    dispatch_rmsnorm_stream(
        reinterpret_cast<bf16 *>(x.typed_data()),
        reinterpret_cast<bf16 *>(weight.typed_data()),
        reinterpret_cast<bf16 *>(out->typed_data()),
        static_cast<float>(eps),
        rows,
        stream
    );
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return ffi::Error::Internal(cudaGetErrorString(status));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JaxtitanRmsNormBf16H1024,
    rmsnorm_bf16_h1024,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Ret<ffi::Buffer<ffi::BF16>>()
        .Attr<double>("eps")
);
