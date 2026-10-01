//! The stress loop for one GPU.
//!
//! Memory is filled with result matrices that should all be identical: each
//! one is the product of the same two inputs. Every pass recomputes a
//! reference result, then recomputes the rest in chunks and checks each chunk
//! against the reference on the GPU. Any difference means the hardware got a
//! calculation wrong. Chunks are sized to take about `--chunk-secs`, so
//! progress, stopping and error reports stay prompt on slow and fast GPUs alike.
//!
//! Each precision pairs an input type with a result type and a way of
//! multiplying: cuBLAS for most, cuBLASLt for FP8.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use cudarc::cublas::sys::{cublasMath_t, cublasOperation_t, cublasSetMathMode};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaViewMut, DeviceRepr, LaunchConfig,
    PushKernelArg, ValidAsZeroBits,
};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use half::{bf16, f16};

use super::DeviceInfo;
use super::fp8::{E4m3, Fp8Gemm};
use crate::mem::{self, HostMemory};
use crate::supervisor::{Ready, Reporter};
use crate::units::format_bytes;
use crate::{Precision, RunArgs};

const KERNELS: &str = include_str!("kernels.cu");
const BLOCK_SIZE: u32 = 256;
const SEED_A: u64 = 0x6275_726e_696e_0001;
const SEED_B: u64 = 0x6275_726e_696e_0002;

/// Element types the input matrices can hold.
trait Element: DeviceRepr + ValidAsZeroBits + Copy + Unpin + 'static {
    const FILL_KERNEL: &'static str;
}

/// Element types the results can hold.
trait ResultElement: Element {
    const VERIFY_KERNEL: &'static str;
    const ONE: Self;
    const ZERO: Self;
    /// A value no real result can equal. Inputs lie in [-1, 1), so results
    /// are bounded by the matrix size.
    const POISON: Self;
}

impl Element for f32 {
    const FILL_KERNEL: &'static str = "burnin_fill_f32";
}

impl ResultElement for f32 {
    const VERIFY_KERNEL: &'static str = "burnin_verify_f32";
    const ONE: Self = 1.0;
    const ZERO: Self = 0.0;
    const POISON: Self = 1.0e30;
}

impl Element for f64 {
    const FILL_KERNEL: &'static str = "burnin_fill_f64";
}

impl ResultElement for f64 {
    const VERIFY_KERNEL: &'static str = "burnin_verify_f64";
    const ONE: Self = 1.0;
    const ZERO: Self = 0.0;
    const POISON: Self = 1.0e300;
}

impl Element for f16 {
    const FILL_KERNEL: &'static str = "burnin_fill_f16";
}

impl ResultElement for f16 {
    const VERIFY_KERNEL: &'static str = "burnin_verify_f16";
    const ONE: Self = f16::ONE;
    const ZERO: Self = f16::ZERO;
    /// FP16 tops out at 65504, below the largest possible result, so the
    /// poison is NaN, which a GEMM of finite inputs never produces.
    const POISON: Self = f16::NAN;
}

impl Element for bf16 {
    const FILL_KERNEL: &'static str = "burnin_fill_bf16";
}

impl ResultElement for bf16 {
    const VERIFY_KERNEL: &'static str = "burnin_verify_bf16";
    const ONE: Self = bf16::ONE;
    const ZERO: Self = bf16::ZERO;
    const POISON: Self = bf16::from_f32_const(1.0e30);
}

impl Element for E4m3 {
    const FILL_KERNEL: &'static str = "burnin_fill_e4m3";
}

/// Multiplies two square input matrices of type `I` into a result of type `R`.
trait Multiply<I, R> {
    fn gemm(&self, a: &CudaSlice<I>, b: &CudaSlice<I>, c: &mut CudaViewMut<'_, R>) -> Result<()>;
}

/// cuBLAS, for precisions whose inputs and results share a type.
struct Cublas {
    blas: CudaBlas,
    n: i32,
}

impl Cublas {
    /// A handle for `n`x`n` GEMMs on `stream`, using the given math mode.
    fn new(stream: &Arc<CudaStream>, n: usize, math: cublasMath_t) -> Result<Self> {
        let blas = CudaBlas::new(stream.clone()).context("could not create a cuBLAS handle")?;
        // SAFETY: the handle is live for as long as `blas` is.
        unsafe { cublasSetMathMode(*blas.handle(), math) }
            .result()
            .context("could not set the cuBLAS math mode")?;
        Ok(Self { blas, n: n as i32 })
    }
}

impl<T: ResultElement> Multiply<T, T> for Cublas
where
    CudaBlas: Gemm<T>,
{
    /// `c = a * b` for square column-major matrices.
    fn gemm(&self, a: &CudaSlice<T>, b: &CudaSlice<T>, c: &mut CudaViewMut<'_, T>) -> Result<()> {
        let n = self.n;
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n,
            n,
            k: n,
            alpha: T::ONE,
            lda: n,
            ldb: n,
            beta: T::ZERO,
            ldc: n,
        };
        // SAFETY: a, b and c each hold n*n elements.
        unsafe { self.blas.gemm(cfg, a, b, c) }.context("cuBLAS GEMM failed")
    }
}

impl Multiply<E4m3, f32> for Fp8Gemm {
    fn gemm(
        &self,
        a: &CudaSlice<E4m3>,
        b: &CudaSlice<E4m3>,
        c: &mut CudaViewMut<'_, f32>,
    ) -> Result<()> {
        Fp8Gemm::gemm(self, a, b, c)
    }
}

/// Compiles the device kernels once; every worker loads the result.
pub fn compile_kernels() -> Result<Ptx> {
    let options = CompileOptions {
        name: Some("burnin_kernels.cu".into()),
        ..Default::default()
    };
    compile_ptx_with_opts(KERNELS, options)
        .map_err(|err| anyhow::anyhow!("NVRTC could not compile the device kernels: {err:?}"))
}

/// Everything one worker needs.
struct Job<'a> {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    info: &'a DeviceInfo,
    args: &'a RunArgs,
    /// Number of unified-memory GPUs sharing host memory in this run.
    host_sharers: u64,
    reporter: &'a Reporter,
    stop: &'a AtomicBool,
}

/// Tests one GPU until `stop` is set.
pub fn worker(
    info: &DeviceInfo,
    ptx: Ptx,
    args: &RunArgs,
    host_sharers: u64,
    reporter: &Reporter,
    stop: &AtomicBool,
) -> Result<()> {
    args.precision
        .check_support(info.compute_capability)
        .map_err(anyhow::Error::msg)?;
    let ctx = CudaContext::new(info.ordinal).context("could not create a CUDA context")?;
    // Everything runs on one stream, so cudarc's cross-stream event tracking
    // would only add overhead.
    // SAFETY: called before any allocation or launch on this context.
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.default_stream();
    let job = Job {
        ctx,
        stream,
        info,
        args,
        host_sharers,
        reporter,
        stop,
    };

    let n = args.matrix_size;
    // TF32 is FP32 with cuBLAS allowed to use TF32 tensor cores.
    let math = if args.precision == Precision::Tf32 {
        cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH
    } else {
        cublasMath_t::CUBLAS_DEFAULT_MATH
    };
    let cublas = || Cublas::new(&job.stream, n, math);
    match args.precision {
        Precision::Fp32 | Precision::Tf32 => burn::<f32, f32>(&job, ptx, &cublas()?),
        Precision::Fp16 => burn::<f16, f16>(&job, ptx, &cublas()?),
        Precision::Bf16 => burn::<bf16, bf16>(&job, ptx, &cublas()?),
        Precision::Fp64 => burn::<f64, f64>(&job, ptx, &cublas()?),
        Precision::Fp8 => {
            let lt = Fp8Gemm::new(&job.stream, n).context("could not set up cuBLASLt for fp8")?;
            burn::<E4m3, f32>(&job, ptx, &lt)
        }
    }
}

fn burn<I: Element, R: ResultElement>(
    job: &Job,
    ptx: Ptx,
    blas: &impl Multiply<I, R>,
) -> Result<()> {
    let Job {
        ctx,
        stream,
        info,
        args,
        ..
    } = job;
    let kernels = Kernels::load::<I, R>(ctx, ptx, info.sm_count)?;

    let n = args.matrix_size;
    let elems = n * n;
    let input_bytes = (elems * size_of::<I>()) as u64;
    let result_bytes = (elems * size_of::<R>()) as u64;

    let (device_free, _) = ctx.mem_get_info()?;
    let host = if info.integrated {
        HostMemory::read().ok()
    } else {
        None
    };
    let budget = mem::budget(
        args.mem,
        device_free as u64,
        info.integrated,
        host,
        job.host_sharers,
    );
    let wanted = mem::result_slots(budget.bytes, input_bytes, result_bytes) as usize;
    if wanted < 2 {
        bail!(
            "a {} budget cannot hold two {n}x{n} inputs of {} and two results of {}; \
             use more memory or a smaller --matrix-size",
            format_bytes(budget.bytes),
            format_bytes(input_bytes),
            format_bytes(result_bytes)
        );
    }

    // SAFETY: both inputs are fully written by the fill kernel before use.
    let mut a = unsafe { stream.alloc::<I>(elems) }?;
    let mut b = unsafe { stream.alloc::<I>(elems) }?;
    kernels.fill(stream, &mut a, SEED_A)?;
    kernels.fill(stream, &mut b, SEED_B)?;
    let (mut results, slots) = alloc_results::<R>(stream, elems, wanted)?;
    let mut mismatch_counter = stream.alloc_zeros::<u64>(1)?;

    // Warm up once so the BLAS library has picked its kernels, then time two
    // GEMMs to size the chunks.
    blas.gemm(&a, &b, &mut results.slice_mut(0..elems))?;
    stream.synchronize()?;
    let timer = Instant::now();
    for _ in 0..2 {
        blas.gemm(&a, &b, &mut results.slice_mut(0..elems))?;
    }
    stream.synchronize()?;
    let secs_per_gemm = timer.elapsed().as_secs_f64() / 2.0;
    let flops_per_gemm = 2.0 * (n as f64).powi(3);
    let chunk = ((args.chunk_secs / secs_per_gemm).round() as usize).clamp(1, slots - 1);

    let reduced = if slots < wanted {
        format!(", reduced from {wanted} after allocation failures")
    } else {
        String::new()
    };
    job.reporter.ready(Ready {
        detail: format!(
            "{slots} {} results of {n}x{n}, {} ({}{reduced}); {chunk} GEMMs per chunk at {:.2} TFLOP/s",
            args.precision.name(),
            format_bytes(result_bytes * slots as u64),
            mem::describe(args.mem, &budget),
            flops_per_gemm / secs_per_gemm / 1e12
        ),
        flops_per_gemm,
        chunk_secs: chunk as f64 * secs_per_gemm,
    });

    let mut pass = 0;
    let mut gemms = 0;
    let mut inject = args.inject_fault;
    while !job.stop.load(Ordering::SeqCst) {
        pass += 1;
        // A fresh reference for this pass, so a fault in an earlier pass
        // cannot hide one in this pass.
        blas.gemm(&a, &b, &mut results.slice_mut(0..elems))?;
        gemms += 1;

        let mut first = 1;
        while first < slots {
            let count = chunk.min(slots - first);
            for slot in first..first + count {
                blas.gemm(
                    &a,
                    &b,
                    &mut results.slice_mut(slot * elems..(slot + 1) * elems),
                )?;
            }
            if inject {
                let index = first * elems + elems / 2;
                stream.memcpy_htod(&[R::POISON], &mut results.slice_mut(index..index + 1))?;
                inject = false;
            }
            let found = kernels.verify(
                stream,
                &results,
                elems,
                first,
                count,
                args.tolerance,
                &mut mismatch_counter,
            )?;
            gemms += count as u64;
            if found > 0 {
                job.reporter.mismatch(pass, first, first + count - 1, found);
            }
            job.reporter.progress(pass, gemms);
            if job.stop.load(Ordering::SeqCst) {
                break;
            }
            first += count;
        }
    }
    Ok(())
}

/// Allocates up to `wanted` result matrices, asking for 10% fewer each time
/// the allocation fails.
fn alloc_results<T: ResultElement>(
    stream: &Arc<CudaStream>,
    elems: usize,
    wanted: usize,
) -> Result<(CudaSlice<T>, usize)> {
    let mut slots = wanted;
    loop {
        // SAFETY: every result is written by a GEMM before it is read.
        match unsafe { stream.alloc::<T>(slots * elems) } {
            Ok(buffer) => return Ok((buffer, slots)),
            Err(_) if slots > 2 => slots = (slots * 9 / 10).clamp(2, slots - 1),
            Err(err) => return Err(err).context("could not allocate memory for results"),
        }
    }
}

struct Kernels {
    fill: CudaFunction,
    verify: CudaFunction,
    max_blocks: usize,
}

impl Kernels {
    /// Loads the fill kernel for inputs of type `I` and the verify kernel for
    /// results of type `R`.
    fn load<I: Element, R: ResultElement>(
        ctx: &Arc<CudaContext>,
        ptx: Ptx,
        sm_count: u32,
    ) -> Result<Self> {
        let module = ctx
            .load_module(ptx)
            .context("could not load the device kernels")?;
        Ok(Self {
            fill: module.load_function(I::FILL_KERNEL)?,
            verify: module.load_function(R::VERIFY_KERNEL)?,
            max_blocks: sm_count as usize * 16,
        })
    }

    fn grid(&self, elems: usize) -> LaunchConfig {
        let blocks = elems
            .div_ceil(BLOCK_SIZE as usize)
            .clamp(1, self.max_blocks);
        LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Fills `buffer` with deterministic pseudo-random values in [-1, 1).
    fn fill<T: Element>(
        &self,
        stream: &Arc<CudaStream>,
        buffer: &mut CudaSlice<T>,
        seed: u64,
    ) -> Result<()> {
        let config = self.grid(buffer.len());
        let len = buffer.len() as u64;
        let mut launch = stream.launch_builder(&self.fill);
        launch.arg(buffer).arg(&len).arg(&seed);
        // SAFETY: the kernel writes exactly `len` elements of `buffer`.
        unsafe { launch.launch(config) }.context("fill kernel failed to launch")?;
        Ok(())
    }

    /// Compares results `first..first + count` against result 0 and returns
    /// how many values differ.
    #[allow(clippy::too_many_arguments)]
    fn verify<T: ResultElement>(
        &self,
        stream: &Arc<CudaStream>,
        results: &CudaSlice<T>,
        elems: usize,
        first: usize,
        count: usize,
        tolerance: f64,
        counter: &mut CudaSlice<u64>,
    ) -> Result<u64> {
        stream.memset_zeros(counter)?;
        let reference = results.slice(0..elems);
        let candidates = results.slice(first * elems..(first + count) * elems);
        let elems_arg = elems as u64;
        let count_arg = count as u32;
        {
            let mut launch = stream.launch_builder(&self.verify);
            launch
                .arg(&reference)
                .arg(&candidates)
                .arg(&elems_arg)
                .arg(&count_arg)
                .arg(&tolerance)
                .arg(&mut *counter);
            // SAFETY: the kernel reads `elems` values of the reference and
            // `count * elems` of the candidates, and writes one u64.
            unsafe { launch.launch(self.grid(elems)) }.context("verify kernel failed to launch")?;
        }
        let found = stream.clone_dtoh(&*counter)?;
        stream.synchronize()?;
        Ok(found[0])
    }
}
