//! The stress loop for one Apple GPU.
//!
//! Memory is filled with result matrices that should all be identical: each
//! one is the product of the same two inputs. Every pass recomputes a
//! reference result, then recomputes the rest in chunks and checks each chunk
//! against the reference on the GPU. Any difference means the hardware got a
//! calculation wrong. Chunks are sized to take about `--chunk-secs`, so
//! progress, stopping and error reports stay prompt on slow and fast GPUs alike.
//!
//! macOS stops GPU work that holds up the rest of the system for too long, so
//! each GEMM is split into bands of rows that take at most about
//! [`COMMAND_SECS`] each. Every band and every check gets a command buffer of
//! its own. They are committed in order on one queue, and the worker only
//! waits for them at the end of each chunk.

use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use objc2::AllocAnyThread;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferError,
    MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary, MTLOrigin,
    MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication, MPSSupportsMTLDevice,
};

use super::Device;
use crate::RunArgs;
use crate::mem;
use crate::supervisor::{Ready, Reporter};
use crate::units::format_bytes;

const KERNELS: &str = include_str!("kernels.metal");
/// Threads per threadgroup; must match `THREADGROUP_SIZE` in kernels.metal.
const THREADGROUP_SIZE: usize = 256;
/// Most threadgroups in one dispatch, which is plenty to fill the largest Apple GPU.
const MAX_THREADGROUPS: usize = 1024;
/// Longest GPU time to aim for in one command buffer. Short command buffers
/// keep macOS from stopping the work for holding up the display, and let other
/// apps' GPU work in quickly.
const COMMAND_SECS: f64 = 0.005;
/// Throughput assumed until the GPU has been timed. It is below that of any
/// Apple GPU, so the first command buffers are short as well.
const ASSUMED_FLOPS: f64 = 1e12;
/// What matrix sizes and bands of rows must be a multiple of. Metal
/// Performance Shaders can leave part of an fp32 result unwritten when a
/// dimension is not a multiple of 64 and the result crosses a 2 GiB boundary in
/// the GPU's address space, which would look like a hardware fault.
pub const ALIGNMENT: usize = 64;
const SEED_A: u64 = 0x6275_726e_696e_0001;
const SEED_B: u64 = 0x6275_726e_696e_0002;
/// Written to every byte of one value by `--inject-fault`, which makes it
/// 3.4e38: a value no real result can equal. Inputs lie in [-1, 1), so results
/// are bounded by the matrix size.
const POISON_BYTE: u8 = 0x7f;

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type CommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Tests one GPU until `stop` is set. `run` has already checked that the
/// precision is fp32, the only one Apple GPUs support.
pub fn worker(
    device: &Device,
    args: &RunArgs,
    reporter: &Reporter,
    stop: &AtomicBool,
) -> Result<()> {
    // Metal hands back many objects autoreleased. Without a pool on this
    // thread they would pile up until it exits.
    autoreleasepool(|_| burn(device, args, reporter, stop))
}

fn burn(device: &Device, args: &RunArgs, reporter: &Reporter, stop: &AtomicBool) -> Result<()> {
    // SAFETY: `device` is a valid Metal device.
    if !unsafe { MPSSupportsMTLDevice(Some(device)) } {
        bail!("Metal Performance Shaders does not support this GPU");
    }
    let kernels = Kernels::load(device)?;
    let mut queue = Queue::new(device)?;

    let n = args.matrix_size;
    let max_buffer = device.maxBufferLength();
    let matrix_bytes = n
        .checked_mul(n)
        .and_then(|elems| elems.checked_mul(size_of::<f32>()));
    let matrix_bytes = match matrix_bytes {
        Some(bytes) if bytes <= max_buffer => bytes,
        _ => bail!(
            "a {n}x{n} matrix does not fit in the largest buffer this GPU allows ({}); \
             use a smaller --matrix-size",
            format_bytes(max_buffer as u64)
        ),
    };
    let elems = n * n;

    let budget = super::budget(args.mem, device);
    let wanted = mem::result_slots(budget.bytes, matrix_bytes as u64, matrix_bytes as u64) as usize;
    if wanted < 2 {
        bail!(
            "a {} budget cannot hold two {n}x{n} inputs and two results ({} each); \
             use more memory or a smaller --matrix-size",
            format_bytes(budget.bytes),
            format_bytes(matrix_bytes as u64)
        );
    }

    let a = alloc_private(device, matrix_bytes).context("could not allocate memory for inputs")?;
    let b = alloc_private(device, matrix_bytes).context("could not allocate memory for inputs")?;
    queue.submit("filling the inputs", |commands| {
        kernels.fill(commands, &a, elems, SEED_A)?;
        kernels.fill(commands, &b, elems, SEED_B)
    })?;
    let results = alloc_results(device, matrix_bytes, wanted)?;
    let slots = results.len();
    queue.finish()?;

    // Warm up once so MPS has picked its kernels, then time two GEMMs to size
    // the bands and the chunks.
    let flops_per_gemm = 2.0 * (n as f64).powi(3);
    let assumed_secs = flops_per_gemm / ASSUMED_FLOPS;
    let gemm = Gemm::new(device, n, band_height(n, assumed_secs), &a, &b, &results);
    gemm.submit(&mut queue, 0)?;
    queue.finish()?;
    let timer = Instant::now();
    for _ in 0..2 {
        gemm.submit(&mut queue, 0)?;
    }
    queue.finish()?;
    let secs_per_gemm = timer.elapsed().as_secs_f64() / 2.0;
    let gemm = Gemm::new(device, n, band_height(n, secs_per_gemm), &a, &b, &results);
    let chunk = ((args.chunk_secs / secs_per_gemm).round() as usize).clamp(1, slots - 1);
    let checker = Checker::new(device, &kernels, elems, args.tolerance as f32, chunk)?;

    let reduced = if slots < wanted {
        format!(", reduced from {wanted} when memory ran out")
    } else {
        String::new()
    };
    reporter.ready(Ready {
        detail: format!(
            "{slots} fp32 results of {n}x{n}, {} ({}{reduced}); {chunk} GEMMs per chunk \
             at {:.2} TFLOP/s, {} command buffers per GEMM",
            format_bytes((matrix_bytes * slots) as u64),
            mem::describe(args.mem, &budget),
            flops_per_gemm / secs_per_gemm / 1e12,
            gemm.bands.len()
        ),
        flops_per_gemm,
        chunk_secs: chunk as f64 * secs_per_gemm,
    });

    let mut pass = 0;
    let mut gemms = 0;
    let mut inject = args.inject_fault;
    while !stop.load(Ordering::SeqCst) {
        pass += 1;
        // A fresh reference for this pass, so a fault in an earlier pass
        // cannot hide one in this pass.
        gemm.submit(&mut queue, 0)?;
        gemms += 1;

        let mut first = 1;
        while first < slots {
            let count = chunk.min(slots - first);
            for slot in first..first + count {
                gemm.submit(&mut queue, slot)?;
            }
            if inject {
                queue.submit("injecting the fault", |commands| {
                    inject_fault(commands, &results[first], elems)
                })?;
                inject = false;
            }
            for (index, candidate) in results[first..first + count].iter().enumerate() {
                queue.submit("checking a result", |commands| {
                    checker.encode(commands, &results[0], candidate, index)
                })?;
            }
            queue.finish()?;
            let found = checker.mismatches(count);
            gemms += count as u64;
            if found > 0 {
                reporter.mismatch(pass, first, first + count - 1, found);
            }
            reporter.progress(pass, gemms);
            if stop.load(Ordering::SeqCst) {
                break;
            }
            first += count;
        }
    }
    Ok(())
}

/// Allocates a buffer that only the GPU can access.
fn alloc_private(device: &Device, bytes: usize) -> Option<Buffer> {
    device.newBufferWithLength_options(bytes, MTLResourceOptions::StorageModePrivate)
}

/// Allocates up to `wanted` result matrices, one buffer each. If memory runs
/// out first, it gives back a tenth of what it got, to leave some headroom.
fn alloc_results(device: &Device, matrix_bytes: usize, wanted: usize) -> Result<Vec<Buffer>> {
    let mut results = Vec::with_capacity(wanted);
    while results.len() < wanted {
        match alloc_private(device, matrix_bytes) {
            Some(buffer) => results.push(buffer),
            None if results.len() >= 2 => {
                results.truncate((results.len() * 9 / 10).max(2));
                break;
            }
            None => bail!("could not allocate memory for results"),
        }
    }
    Ok(results)
}

/// Overwrites the value in the middle of `result` with a poison value.
fn inject_fault(commands: &CommandBuffer, result: &Buffer, elems: usize) -> Result<()> {
    let blit = commands
        .blitCommandEncoder()
        .context("could not create a Metal blit encoder")?;
    let offset = elems / 2 * size_of::<f32>();
    blit.fillBuffer_range_value(result, NSRange::new(offset, size_of::<f32>()), POISON_BYTE);
    blit.endEncoding();
    Ok(())
}

/// Commits command buffers in order on one queue, and checks each one once
/// it has finished.
struct Queue {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Command buffers committed but not yet checked, oldest first, with what
    /// each one does.
    pending: Vec<(Retained<CommandBuffer>, &'static str)>,
}

impl Queue {
    fn new(device: &Device) -> Result<Self> {
        let queue = device
            .newCommandQueue()
            .context("could not create a Metal command queue")?;
        Ok(Self {
            queue,
            pending: Vec::new(),
        })
    }

    /// Encodes a new command buffer with `encode` and commits it.
    fn submit(
        &mut self,
        task: &'static str,
        encode: impl FnOnce(&CommandBuffer) -> Result<()>,
    ) -> Result<()> {
        // Metal hands back command buffers and encoders autoreleased, so each
        // submission drains its own pool.
        autoreleasepool(|_| {
            let commands = self
                .queue
                .commandBuffer()
                .context("could not create a Metal command buffer")?;
            encode(&commands)?;
            commands.commit();
            self.pending.push((commands, task));
            Ok(())
        })
    }

    /// Waits for every committed command buffer, and fails if the GPU reported
    /// an error in any of them.
    fn finish(&mut self) -> Result<()> {
        for (commands, task) in self.pending.drain(..) {
            commands.waitUntilCompleted();
            if commands.status() != MTLCommandBufferStatus::Completed {
                bail!("{task} failed on the GPU: {}", command_error(&commands));
            }
        }
        Ok(())
    }
}

/// Explains why a command buffer failed.
fn command_error(commands: &CommandBuffer) -> String {
    let Some(error) = commands.error() else {
        return format!("it ended with status {:?}", commands.status());
    };
    let hint = if error.code() == MTLCommandBufferError::Timeout.0 as isize {
        "; macOS stops GPU work that runs too long, so try a smaller --matrix-size"
    } else {
        ""
    };
    format!("{}{hint}", error.localizedDescription())
}

/// Rows per band for a GEMM that takes `gemm_secs` as a whole, so that each
/// band takes at most about [`COMMAND_SECS`]. It is a multiple of
/// [`ALIGNMENT`], as `n` is, and at most `n`.
fn band_height(n: usize, gemm_secs: f64) -> usize {
    let rows = (n as f64 * COMMAND_SECS / gemm_secs) as usize;
    (rows / ALIGNMENT * ALIGNMENT).clamp(ALIGNMENT, n)
}

/// The rows of each band of an `n`-row matrix split `height` rows at a time.
fn band_rows(n: usize, height: usize) -> impl Iterator<Item = Range<usize>> {
    (0..n)
        .step_by(height)
        .map(move |start| start..(start + height).min(n))
}

/// `c = a * b` for the two inputs and each result, all square row-major fp32
/// matrices, split into bands of rows of `c`.
struct Gemm {
    /// One kernel per band.
    bands: Vec<Retained<MPSMatrixMultiplication>>,
    a: Retained<MPSMatrix>,
    b: Retained<MPSMatrix>,
    results: Vec<Retained<MPSMatrix>>,
}

impl Gemm {
    /// Splits the GEMM into bands of `height` rows.
    fn new(
        device: &Device,
        n: usize,
        height: usize,
        a: &Buffer,
        b: &Buffer,
        results: &[Buffer],
    ) -> Self {
        // SAFETY: the rows of an n x n fp32 matrix are n * 4 bytes apart.
        let descriptor = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                n,
                n,
                n * size_of::<f32>(),
                MPSDataType::Float32,
            )
        };
        let matrix = |buffer: &Buffer| {
            // SAFETY: every buffer holds exactly one matrix of this shape.
            unsafe { MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), buffer, &descriptor) }
        };
        Self {
            bands: band_rows(n, height)
                .map(|rows| band_kernel(device, n, rows))
                .collect(),
            a: matrix(a),
            b: matrix(b),
            results: results.iter().map(matrix).collect(),
        }
    }

    /// Commits the GEMM that writes result `slot`, one command buffer per band.
    fn submit(&self, queue: &mut Queue, slot: usize) -> Result<()> {
        for kernel in &self.bands {
            queue.submit("a GEMM", |commands| {
                // SAFETY: each band lies within the matrices, which all have
                // the kernel's shape, and the command buffer keeps their
                // buffers alive until it has finished.
                unsafe {
                    kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                        commands,
                        &self.a,
                        &self.b,
                        &self.results[slot],
                    )
                };
                Ok(())
            })?;
        }
        Ok(())
    }
}

/// A kernel for the rows `rows` of `c = a * b`, for n x n matrices.
fn band_kernel(device: &Device, n: usize, rows: Range<usize>) -> Retained<MPSMatrixMultiplication> {
    // SAFETY: plain C = A * B, for `rows.len()` rows of C and A.
    let kernel = unsafe {
        MPSMatrixMultiplication::initWithDevice_resultRows_resultColumns_interiorColumns(
            MPSMatrixMultiplication::alloc(),
            device,
            rows.len(),
            n,
            n,
        )
    };
    // MPS takes an origin's x as the row.
    let origin = MTLOrigin {
        x: rows.start,
        y: 0,
        z: 0,
    };
    // SAFETY: the rows lie within A and C.
    unsafe {
        kernel.setLeftMatrixOrigin(origin);
        kernel.setResultMatrixOrigin(origin);
    }
    kernel
}

/// Threadgroups in a dispatch over `elems` values.
fn threadgroups(elems: usize) -> MTLSize {
    MTLSize {
        width: elems.div_ceil(THREADGROUP_SIZE).clamp(1, MAX_THREADGROUPS),
        height: 1,
        depth: 1,
    }
}

const THREADS: MTLSize = MTLSize {
    width: THREADGROUP_SIZE,
    height: 1,
    depth: 1,
};

struct Kernels {
    fill: Pipeline,
    verify: Pipeline,
}

impl Kernels {
    /// Compiles the kernels in kernels.metal for `device`.
    fn load(device: &Device) -> Result<Self> {
        let options = MTLCompileOptions::new();
        // Fast math lets the compiler assume values are never NaN or infinite,
        // which could hide the very results the verify kernel looks for.
        // `mathMode` replaces this setting from macOS 15, but this one works on
        // every version.
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(KERNELS), Some(&options))
            .map_err(|err| {
                anyhow!(
                    "Metal could not compile the GPU kernels: {}",
                    err.localizedDescription()
                )
            })?;
        let pipeline = |name: &str| -> Result<Pipeline> {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .with_context(|| format!("the GPU kernels have no function {name}"))?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|err| {
                    anyhow!(
                        "Metal could not prepare {name}: {}",
                        err.localizedDescription()
                    )
                })?;
            if pipeline.maxTotalThreadsPerThreadgroup() < THREADGROUP_SIZE {
                bail!("{name} cannot run {THREADGROUP_SIZE} threads per threadgroup on this GPU");
            }
            Ok(pipeline)
        };
        Ok(Self {
            fill: pipeline("burnin_fill_f32")?,
            verify: pipeline("burnin_verify_f32")?,
        })
    }

    /// Fills `buffer` with `len` deterministic pseudo-random values in [-1, 1).
    fn fill(&self, commands: &CommandBuffer, buffer: &Buffer, len: usize, seed: u64) -> Result<()> {
        let encoder = commands
            .computeCommandEncoder()
            .context("could not create a Metal compute encoder")?;
        encoder.setComputePipelineState(&self.fill);
        let len_arg = len as u64;
        // SAFETY: the kernel writes `len` floats to `buffer`, which holds that
        // many, and reads the two u64 arguments.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buffer), 0, 0);
            encoder.setBytes_length_atIndex(NonNull::from(&len_arg).cast(), size_of::<u64>(), 1);
            encoder.setBytes_length_atIndex(NonNull::from(&seed).cast(), size_of::<u64>(), 2);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups(len), THREADS);
        encoder.endEncoding();
        Ok(())
    }
}

/// Compares results against the reference with the verify kernel.
struct Checker {
    pipeline: Pipeline,
    elems: usize,
    tolerance: f32,
    /// What the verify kernel writes: one mismatch count per threadgroup, for
    /// each result in a chunk.
    counts: Buffer,
    groups: MTLSize,
    /// Most results in one chunk.
    capacity: usize,
}

impl Checker {
    fn new(
        device: &Device,
        kernels: &Kernels,
        elems: usize,
        tolerance: f32,
        capacity: usize,
    ) -> Result<Self> {
        let groups = threadgroups(elems);
        let counts = device
            .newBufferWithLength_options(
                capacity * groups.width * size_of::<u64>(),
                MTLResourceOptions::StorageModeShared,
            )
            .context("could not allocate memory for mismatch counts")?;
        Ok(Self {
            pipeline: kernels.verify.clone(),
            elems,
            tolerance,
            counts,
            groups,
            capacity,
        })
    }

    /// Compares `candidate`, the `index`th result of its chunk, against `reference`.
    fn encode(
        &self,
        commands: &CommandBuffer,
        reference: &Buffer,
        candidate: &Buffer,
        index: usize,
    ) -> Result<()> {
        assert!(index < self.capacity);
        let encoder = commands
            .computeCommandEncoder()
            .context("could not create a Metal compute encoder")?;
        encoder.setComputePipelineState(&self.pipeline);
        let elems_arg = self.elems as u64;
        let offset = index * self.groups.width * size_of::<u64>();
        // SAFETY: the kernel reads `elems` floats from the reference and the
        // candidate, which both hold that many, and writes one count per
        // threadgroup at `offset`, where `counts` has room for them.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(reference), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(candidate), 0, 1);
            encoder.setBytes_length_atIndex(NonNull::from(&elems_arg).cast(), size_of::<u64>(), 2);
            encoder.setBytes_length_atIndex(
                NonNull::from(&self.tolerance).cast(),
                size_of::<f32>(),
                3,
            );
            encoder.setBuffer_offset_atIndex(Some(&self.counts), offset, 4);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(self.groups, THREADS);
        encoder.endEncoding();
        Ok(())
    }

    /// Adds up the mismatches in the first `results` results of the chunk
    /// just checked. Call it only once those checks have finished.
    fn mismatches(&self, results: usize) -> u64 {
        assert!(results <= self.capacity);
        // SAFETY: the buffer is shared with the CPU and holds `capacity`
        // results' counts, and the GPU has finished writing them.
        let counts = unsafe {
            std::slice::from_raw_parts(
                self.counts.contents().cast::<u64>().as_ptr(),
                results * self.groups.width,
            )
        };
        counts.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_take_at_most_about_the_target_time() {
        // 8192 * 0.005 / 0.1 is 409.6 rows, rounded down to a multiple of 64.
        assert_eq!(band_height(8192, 0.1), 384);
        // A fast GPU runs a whole GEMM in one command buffer.
        assert_eq!(band_height(8192, 0.001), 8192);
        // A band is never thinner than the alignment.
        assert_eq!(band_height(8192, 10.0), ALIGNMENT);
    }

    #[test]
    fn bands_cover_every_row_once() {
        for (n, height) in [(8192, 8192), (8192, 384), (8192, 64), (320, 128)] {
            let mut next = 0;
            for rows in band_rows(n, height) {
                assert_eq!(rows.start, next);
                assert_eq!(rows.start % ALIGNMENT, 0);
                assert_eq!(rows.len() % ALIGNMENT, 0);
                assert!(rows.len() <= height);
                next = rows.end;
            }
            assert_eq!(next, n);
        }
        assert_eq!(band_rows(8192, 384).count(), 22);
    }
}
