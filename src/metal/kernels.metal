// burnin GPU kernels for Apple GPUs.
//
// This file is compiled at runtime by Metal, with fast math off so that NaN and
// infinity compare as IEEE 754 says. Every kernel uses a grid-stride loop and
// can be dispatched with any number of threadgroups of THREADGROUP_SIZE threads.
//
// Metal has no 64-bit atomic add, so the verify kernel counts without atomics.
// Each thread counts its mismatches in 64 bits, each threadgroup adds up its
// threads' counts and writes the total to its own element of `partials`, and
// the host adds up the partials. No count can overflow, because a buffer holds
// far fewer than 2^64 values.

#include <metal_stdlib>
using namespace metal;

#define THREADGROUP_SIZE 256

// SplitMix64 finaliser: a cheap, well-mixed hash.
static inline ulong mix64(ulong x) {
    x += 0x9e3779b97f4a7c15ul;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ul;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebul;
    return x ^ (x >> 31);
}

// Maps the top 24 bits of a hash to a uniform value in [-1, 1). Every such
// value is exact in fp32.
static inline float to_unit(ulong h) {
    return float(h >> 40) * (1.0f / 8388608.0f) - 1.0f;
}

kernel void burnin_fill_f32(device float *out [[buffer(0)]],
                            constant ulong &n [[buffer(1)]],
                            constant ulong &seed [[buffer(2)]],
                            uint thread_index [[thread_position_in_grid]],
                            uint threads [[threads_per_grid]]) {
    for (ulong i = thread_index; i < n; i += threads)
        out[i] = to_unit(mix64(seed ^ mix64(i)));
}

// Counts the elements of `cand` that differ from `ref`, writing one count per
// threadgroup. A value matches when its bits are identical, or when it is
// within `tol` of the reference (which also treats -0 and +0 as equal).
[[max_total_threads_per_threadgroup(THREADGROUP_SIZE)]]
kernel void burnin_verify_f32(device const float *ref [[buffer(0)]],
                              device const float *cand [[buffer(1)]],
                              constant ulong &elems [[buffer(2)]],
                              constant float &tol [[buffer(3)]],
                              device ulong *partials [[buffer(4)]],
                              uint thread_index [[thread_position_in_grid]],
                              uint threads [[threads_per_grid]],
                              uint lane [[thread_position_in_threadgroup]],
                              uint group [[threadgroup_position_in_grid]]) {
    ulong local = 0;
    for (ulong i = thread_index; i < elems; i += threads) {
        const float r = ref[i];
        const float v = cand[i];
        if (as_type<uint>(v) != as_type<uint>(r) && !(fabs(v - r) <= tol))
            ++local;
    }

    // Sum the threadgroup's counts in threadgroup memory.
    threadgroup ulong counts[THREADGROUP_SIZE];
    counts[lane] = local;
    for (uint stride = THREADGROUP_SIZE / 2; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < stride)
            counts[lane] += counts[lane + stride];
    }
    if (lane == 0)
        partials[group] = counts[0];
}
