// burnin device kernels.
//
// This file is compiled at runtime with NVRTC, so it must not include any
// headers. All kernels use grid-stride loops and can be launched with any grid.
//
// CUDA's 16- and 8-bit float types come from headers, so values of those types
// are stored as raw bits and converted here. One compiled module serves every
// GPU in a run, so the conversions only use instructions that all GPUs have:
// inline PTX for FP16, and integer arithmetic for BF16 and FP8.

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

// Raw bits of an IEEE half-precision, bfloat16 or FP8 E4M3 value.
struct f16_bits { u16 bits; };
struct bf16_bits { u16 bits; };
struct e4m3_bits { u8 bits; };

// SplitMix64 finaliser: a cheap, well-mixed hash.
__device__ __forceinline__ u64 mix64(u64 x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// Maps the top 53 bits of a hash to a uniform value in [-1, 1).
__device__ __forceinline__ double to_unit(u64 h) {
    return (double)(h >> 11) * (1.0 / 4503599627370496.0) - 1.0;
}

// Converts a fill value to an element type. Every conversion rounds toward
// zero, so values stay in [-1, 1).
template <typename T> __device__ T from_double(double v);

template <> __device__ __forceinline__ double from_double<double>(double v) { return v; }

template <> __device__ __forceinline__ float from_double<float>(double v) {
    return __double2float_rz(v);
}

template <> __device__ __forceinline__ f16_bits from_double<f16_bits>(double v) {
    f16_bits out;
    asm("cvt.rz.f16.f32 %0, %1;" : "=h"(out.bits) : "f"(__double2float_rz(v)));
    return out;
}

// Bfloat16 is the top half of a float.
template <> __device__ __forceinline__ bf16_bits from_double<bf16_bits>(double v) {
    bf16_bits out;
    out.bits = (u16)(__float_as_uint(__double2float_rz(v)) >> 16);
    return out;
}

// E4M3 has a sign bit, 4 exponent bits with a bias of 7 and 3 mantissa bits.
// This is only correct for magnitudes below 448, its largest value.
template <> __device__ __forceinline__ e4m3_bits from_double<e4m3_bits>(double v) {
    const u32 bits = __float_as_uint(__double2float_rz(v));
    const u32 sign = (bits >> 24) & 0x80u;
    const u32 magnitude = bits & 0x7fffffffu;
    e4m3_bits out;
    if (magnitude < 0x3c800000u) {
        // Below 2^-6, the smallest normal value: subnormals are multiples of 2^-9.
        out.bits = (u8)(sign | (u32)(__uint_as_float(magnitude) * 512.0f));
    } else {
        // Keep the top 3 mantissa bits and rebias the exponent from 127 to 7.
        out.bits = (u8)(sign | ((magnitude >> 20) - ((127u - 7u) << 3)));
    }
    return out;
}

__device__ __forceinline__ u32 to_bits(float v) { return __float_as_uint(v); }
__device__ __forceinline__ u64 to_bits(double v) { return (u64)__double_as_longlong(v); }
__device__ __forceinline__ u16 to_bits(f16_bits v) { return v.bits; }
__device__ __forceinline__ u16 to_bits(bf16_bits v) { return v.bits; }

// Widens a result exactly, to measure how far apart two results are.
__device__ __forceinline__ double to_double(float v) { return v; }
__device__ __forceinline__ double to_double(double v) { return v; }

__device__ __forceinline__ double to_double(f16_bits v) {
    float out;
    asm("cvt.f32.f16 %0, %1;" : "=f"(out) : "h"(v.bits));
    return out;
}

__device__ __forceinline__ double to_double(bf16_bits v) {
    return __uint_as_float((u32)v.bits << 16);
}

template <typename T>
__device__ void fill(T *out, u64 n, u64 seed) {
    const u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        out[i] = from_double<T>(to_unit(mix64(seed ^ mix64(i))));
}

// Counts the elements of `count` consecutive candidate matrices that differ
// from `ref`. A value matches when its bits are identical, or when it is
// within `tol` of the reference (which also treats -0 and +0 as equal).
template <typename T>
__device__ void verify(const T *ref, const T *cand, u64 elems, u32 count,
                       double tol, u64 *mismatches) {
    u64 local = 0;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < elems; i += stride) {
        const T r = ref[i];
        const auto r_bits = to_bits(r);
        for (u32 s = 0; s < count; ++s) {
            const T v = cand[(u64)s * elems + i];
            if (to_bits(v) != r_bits && !(fabs(to_double(v) - to_double(r)) <= tol))
                ++local;
        }
    }

    // Sum across the warp so only one lane per warp touches global memory.
    for (int offset = 16; offset > 0; offset >>= 1)
        local += __shfl_down_sync(0xffffffffu, local, offset);
    if ((threadIdx.x & 31u) == 0 && local != 0)
        atomicAdd(mismatches, local);
}

extern "C" __global__ void burnin_fill_f32(float *out, u64 n, u64 seed) {
    fill(out, n, seed);
}

extern "C" __global__ void burnin_fill_f64(double *out, u64 n, u64 seed) {
    fill(out, n, seed);
}

extern "C" __global__ void burnin_fill_f16(f16_bits *out, u64 n, u64 seed) {
    fill(out, n, seed);
}

extern "C" __global__ void burnin_fill_bf16(bf16_bits *out, u64 n, u64 seed) {
    fill(out, n, seed);
}

extern "C" __global__ void burnin_fill_e4m3(e4m3_bits *out, u64 n, u64 seed) {
    fill(out, n, seed);
}

extern "C" __global__ void burnin_verify_f32(const float *ref, const float *cand, u64 elems,
                                             u32 count, double tol, u64 *mismatches) {
    verify(ref, cand, elems, count, tol, mismatches);
}

extern "C" __global__ void burnin_verify_f64(const double *ref, const double *cand, u64 elems,
                                             u32 count, double tol, u64 *mismatches) {
    verify(ref, cand, elems, count, tol, mismatches);
}

extern "C" __global__ void burnin_verify_f16(const f16_bits *ref, const f16_bits *cand,
                                             u64 elems, u32 count, double tol,
                                             u64 *mismatches) {
    verify(ref, cand, elems, count, tol, mismatches);
}

extern "C" __global__ void burnin_verify_bf16(const bf16_bits *ref, const bf16_bits *cand,
                                              u64 elems, u32 count, double tol,
                                              u64 *mismatches) {
    verify(ref, cand, elems, count, tol, mismatches);
}
