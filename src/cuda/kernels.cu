// burnin device kernels.
//
// This file is compiled at runtime with NVRTC, so it must not include any
// headers. All kernels use grid-stride loops and can be launched with any grid.

typedef unsigned long long u64;
typedef unsigned int u32;

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

__device__ __forceinline__ u32 to_bits(float v) { return __float_as_uint(v); }
__device__ __forceinline__ u64 to_bits(double v) { return (u64)__double_as_longlong(v); }

template <typename T>
__device__ void fill(T *out, u64 n, u64 seed) {
    const u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        out[i] = (T)to_unit(mix64(seed ^ mix64(i)));
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
            if (to_bits(v) != r_bits && !(fabs((double)v - (double)r) <= tol))
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

extern "C" __global__ void burnin_verify_f32(const float *ref, const float *cand, u64 elems,
                                             u32 count, double tol, u64 *mismatches) {
    verify(ref, cand, elems, count, tol, mismatches);
}

extern "C" __global__ void burnin_verify_f64(const double *ref, const double *cand, u64 elems,
                                             u32 count, double tol, u64 *mismatches) {
    verify(ref, cand, elems, count, tol, mismatches);
}
