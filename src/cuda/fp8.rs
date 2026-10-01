//! FP8 matrix multiplies through cuBLASLt.
//!
//! cudarc's safe cuBLASLt API needs the inputs and the result to share one
//! type, so FP8 inputs with FP32 results go through its lower-level bindings.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublaslt::result::{self, CublasError};
use cudarc::cublaslt::sys::{
    self, cublasComputeType_t, cublasLtMatmulDescAttributes_t as DescAttr,
    cublasLtMatmulPreferenceAttributes_t as PrefAttr, cudaDataType_t,
};
use cudarc::driver::{CudaSlice, CudaStream, CudaViewMut, DevicePtr, DevicePtrMut};

/// FP8 E4M3 values, stored as raw bits.
pub type E4m3 = u8;

/// NVIDIA's recommended cuBLASLt workspace for Hopper and newer, which is
/// also plenty for Ada.
const WORKSPACE_BYTES: usize = 32 << 20;

/// cuBLASLt set up for square GEMMs of one size, with FP8 E4M3 inputs and
/// FP32 results.
pub struct Fp8Gemm {
    stream: Arc<CudaStream>,
    workspace: CudaSlice<u8>,
    handle: sys::cublasLtHandle_t,
    desc: sys::cublasLtMatmulDesc_t,
    /// Shared by both inputs, which have the same shape and type.
    input_layout: sys::cublasLtMatrixLayout_t,
    /// Shared by C and D, which are the same matrix.
    result_layout: sys::cublasLtMatrixLayout_t,
    algo: sys::cublasLtMatmulAlgo_t,
}

impl Fp8Gemm {
    /// Sets up `n`x`n` GEMMs on `stream` and picks cuBLASLt's preferred
    /// algorithm for them.
    pub fn new(stream: &Arc<CudaStream>, n: usize) -> Result<Self> {
        // SAFETY: the workspace is scratch memory that cuBLASLt writes before reading.
        let workspace = unsafe { stream.alloc::<u8>(WORKSPACE_BYTES) }
            .context("could not allocate the cuBLASLt workspace")?;
        // Every handle starts out null, so if a step below fails, dropping
        // `gemm` frees only what was created.
        let mut gemm = Self {
            stream: stream.clone(),
            workspace,
            handle: ptr::null_mut(),
            desc: ptr::null_mut(),
            input_layout: ptr::null_mut(),
            result_layout: ptr::null_mut(),
            algo: sys::cublasLtMatmulAlgo_t { data: [0; 8] },
        };
        gemm.handle = result::create_handle().context("could not create a cuBLASLt handle")?;

        // FP8 needs FP32 compute and scale types, and the "TN" form: A
        // transposed and B not, which is B's default. The inputs have no
        // scale factors set, so cuBLASLt scales them by 1.
        gemm.desc = result::create_matmul_desc(
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cudaDataType_t::CUDA_R_32F,
        )?;
        let transpose = cublasOperation_t::CUBLAS_OP_T;
        // SAFETY: `desc` is live, and TRANSA takes a cublasOperation_t.
        unsafe {
            result::set_matmul_desc_attribute(
                gemm.desc,
                DescAttr::CUBLASLT_MATMUL_DESC_TRANSA,
                ptr::from_ref(&transpose).cast(),
                size_of_val(&transpose),
            )
        }?;

        let (rows, ld) = (n as u64, n as i64);
        gemm.input_layout =
            result::create_matrix_layout(cudaDataType_t::CUDA_R_8F_E4M3, rows, rows, ld)?;
        gemm.result_layout =
            result::create_matrix_layout(cudaDataType_t::CUDA_R_32F, rows, rows, ld)?;
        gemm.algo = gemm
            .pick_algo()
            .context("cuBLASLt has no FP8 GEMM for this GPU and matrix size")?;
        Ok(gemm)
    }

    /// Asks cuBLASLt for its fastest algorithm that fits in the workspace.
    fn pick_algo(&self) -> Result<sys::cublasLtMatmulAlgo_t, CublasError> {
        let pref = result::create_matmul_pref()?;
        let workspace_bytes = WORKSPACE_BYTES as u64;
        // SAFETY: every descriptor is live, and the workspace limit is the
        // u64 that cuBLASLt expects.
        let heuristic = unsafe {
            result::set_matmul_pref_attribute(
                pref,
                PrefAttr::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                ptr::from_ref(&workspace_bytes).cast(),
                size_of_val(&workspace_bytes),
            )
            .and_then(|()| {
                result::get_matmul_algo_heuristic(
                    self.handle,
                    self.desc,
                    self.input_layout,
                    self.input_layout,
                    self.result_layout,
                    self.result_layout,
                    pref,
                )
            })
        };
        // SAFETY: `pref` is live and not used again. As in `drop`, an error
        // here is ignored.
        let _ = unsafe { result::destroy_matmul_pref(pref) };
        Ok(heuristic?.algo)
    }

    /// `c = aᵀ * b` for square column-major matrices of the size this was
    /// set up for. cuBLASLt only multiplies FP8 with A transposed, which is
    /// just as much work as `a * b`.
    pub fn gemm(
        &self,
        a: &CudaSlice<E4m3>,
        b: &CudaSlice<E4m3>,
        c: &mut CudaViewMut<'_, f32>,
    ) -> Result<()> {
        let (alpha, beta) = (1.0f32, 0.0f32);
        let stream = &self.stream;
        let (a, _record_a) = a.device_ptr(stream);
        let (b, _record_b) = b.device_ptr(stream);
        let (c, _record_c) = c.device_ptr_mut(stream);
        let (workspace, _record_workspace) = self.workspace.device_ptr(stream);
        // SAFETY: the descriptors are live and describe the buffers: a and b
        // hold n*n E4M3 values and c holds n*n f32. C and D are the same
        // matrix with the same layout, and beta = 0, so C is never read.
        unsafe {
            result::matmul(
                self.handle,
                self.desc,
                ptr::from_ref(&alpha).cast(),
                ptr::from_ref(&beta).cast(),
                a as *const c_void,
                self.input_layout,
                b as *const c_void,
                self.input_layout,
                c as *const c_void,
                self.result_layout,
                c as *mut c_void,
                self.result_layout,
                &self.algo,
                workspace as *mut c_void,
                WORKSPACE_BYTES,
                stream.cu_stream() as sys::cudaStream_t,
            )
        }
        .context("cuBLASLt FP8 GEMM failed")
    }
}

impl Drop for Fp8Gemm {
    fn drop(&mut self) {
        // SAFETY: each handle is either null or live, and is not used again.
        // Errors are ignored, since nothing useful can be done about them here.
        unsafe {
            if !self.result_layout.is_null() {
                let _ = result::destroy_matrix_layout(self.result_layout);
            }
            if !self.input_layout.is_null() {
                let _ = result::destroy_matrix_layout(self.input_layout);
            }
            if !self.desc.is_null() {
                let _ = result::destroy_matmul_desc(self.desc);
            }
            if !self.handle.is_null() {
                let _ = result::destroy_handle(self.handle);
            }
        }
    }
}
