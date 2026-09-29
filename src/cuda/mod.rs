//! CUDA backend: device discovery, probing and the stress loop.

mod burn;

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use cudarc::driver::CudaContext;
use cudarc::driver::sys::CUdevice_attribute as Attr;

pub use burn::run;

use crate::mem::{self, HostMemory, MemSpec};
use crate::telemetry::Telemetry;
use crate::units::format_bytes;

/// Fails with a readable message when a CUDA library is missing, instead of
/// the panic cudarc raises on first use.
fn require_libraries(blas: bool, nvrtc: bool) -> Result<()> {
    // SAFETY: these only try to load the shared libraries.
    let (driver_found, blas_found, nvrtc_found) = unsafe {
        (
            cudarc::driver::sys::is_culib_present(),
            !blas || cudarc::cublas::sys::is_culib_present(),
            !nvrtc || cudarc::nvrtc::sys::is_culib_present(),
        )
    };
    if !driver_found {
        bail!(
            "the NVIDIA driver library (libcuda.so) was not found; is the NVIDIA driver installed?"
        );
    }
    if !blas_found {
        bail!(
            "cuBLAS (libcublas.so) was not found; install the CUDA libraries or add them to LD_LIBRARY_PATH"
        );
    }
    if !nvrtc_found {
        bail!(
            "NVRTC (libnvrtc.so) was not found; install the CUDA libraries or add them to LD_LIBRARY_PATH"
        );
    }
    Ok(())
}

fn device_count() -> Result<usize> {
    let count = CudaContext::device_count().context("could not initialise CUDA")?;
    Ok(count.max(0) as usize)
}

/// Opens a device, checking the ordinal first so the error is clear.
fn open(ordinal: usize) -> Result<Arc<CudaContext>> {
    let count = device_count()?;
    if count == 0 {
        bail!("no CUDA devices found");
    }
    if ordinal >= count {
        bail!("device {ordinal} does not exist; found {count} device(s)");
    }
    CudaContext::new(ordinal).with_context(|| format!("could not open device {ordinal}"))
}

/// Static facts about a device.
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub sm_count: u32,
    /// The GPU shares system RAM with the CPU instead of having its own memory.
    pub integrated: bool,
    pub total_mem: u64,
    /// PCI address in the format NVML expects, e.g. `00000000:01:00.0`.
    pub pci_bus_id: String,
}

impl DeviceInfo {
    fn query(ctx: &Arc<CudaContext>) -> Result<Self> {
        let domain = ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)?;
        let bus = ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)?;
        let device = ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)?;
        Ok(Self {
            ordinal: ctx.ordinal(),
            name: ctx.name()?,
            compute_capability: ctx.compute_capability()?,
            sm_count: ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32,
            integrated: ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_INTEGRATED)? != 0,
            total_mem: ctx.total_mem()? as u64,
            pci_bus_id: format!("{domain:08X}:{bus:02X}:{device:02X}.0"),
        })
    }

    pub fn arch(&self) -> String {
        let (major, minor) = self.compute_capability;
        format!("sm_{major}{minor}")
    }
}

pub fn list() -> Result<()> {
    require_libraries(false, false)?;
    let count = device_count()?;
    if count == 0 {
        println!("no CUDA devices found");
    }
    for ordinal in 0..count {
        let info = DeviceInfo::query(&CudaContext::new(ordinal)?)?;
        println!(
            "{:>2}  {:<32} {:<7} {:>10}  {}{}",
            info.ordinal,
            info.name,
            info.arch(),
            format_bytes(info.total_mem),
            info.pci_bus_id,
            if info.integrated {
                "  unified memory"
            } else {
                ""
            }
        );
    }
    Ok(())
}

pub fn probe(only: Option<usize>) -> Result<()> {
    // SAFETY: these only try to load the shared libraries.
    let libraries = unsafe {
        [
            ("libcuda", cudarc::driver::sys::is_culib_present()),
            ("libcublas", cudarc::cublas::sys::is_culib_present()),
            ("libnvrtc", cudarc::nvrtc::sys::is_culib_present()),
        ]
    };
    let telemetry = Telemetry::init();
    println!("libraries");
    for (name, found) in libraries {
        println!("  {name:<18}{}", if found { "found" } else { "missing" });
    }
    match &telemetry {
        Ok(_) => println!("  {:<18}found", "libnvidia-ml"),
        Err(err) => println!("  {:<18}unavailable ({err})", "libnvidia-ml"),
    }
    require_libraries(false, false)?;

    let host = HostMemory::read().ok();
    if let Some(host) = host {
        println!("host memory");
        println!("  {:<18}{}", "total", format_bytes(host.total));
        println!("  {:<18}{}", "free", format_bytes(host.free));
        println!("  {:<18}{}", "available", format_bytes(host.available));
    }
    if let Ok(telemetry) = &telemetry {
        if let Some(version) = telemetry.driver_version() {
            println!("driver {version}");
        }
    }

    let count = device_count()?;
    let ordinals: Vec<usize> = match only {
        Some(ordinal) if ordinal >= count => {
            bail!("device {ordinal} does not exist; found {count} device(s)")
        }
        Some(ordinal) => vec![ordinal],
        None => (0..count).collect(),
    };
    if ordinals.is_empty() {
        println!("no CUDA devices found");
    }

    for ordinal in ordinals {
        let ctx = CudaContext::new(ordinal)?;
        let info = DeviceInfo::query(&ctx)?;
        let (free, total) = ctx.mem_get_info()?;
        let yes_no = |value: i32| if value != 0 { "yes" } else { "no" };

        println!("device {ordinal}: {}", info.name);
        println!(
            "  {:<18}{} ({} SMs)",
            "architecture",
            info.arch(),
            info.sm_count
        );
        println!("  {:<18}{}", "pci bus id", info.pci_bus_id);
        println!(
            "  {:<18}{}",
            "unified memory",
            yes_no(info.integrated as i32)
        );
        println!(
            "  {:<18}{}",
            "pageable access",
            yes_no(ctx.attribute(
                Attr::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
            )?)
        );
        println!(
            "  {:<18}{}",
            "ecc",
            yes_no(ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_ECC_ENABLED)?)
        );
        println!(
            "  {:<18}{} free of {}",
            "cuda memory",
            format_bytes(free as u64),
            format_bytes(total as u64)
        );

        let spec = MemSpec::default();
        let budget = mem::budget(spec, free as u64, info.integrated, host);
        let square = (crate::DEFAULT_MATRIX_SIZE * crate::DEFAULT_MATRIX_SIZE) as u64;
        println!(
            "  {:<18}{} ({}): {} fp32 or {} fp64 result matrices",
            "default budget",
            format_bytes(budget.bytes),
            mem::describe(spec, &budget),
            mem::result_slots(budget.bytes, square * 4),
            mem::result_slots(budget.bytes, square * 8),
        );

        match &telemetry {
            Ok(telemetry) => match telemetry.device(&info.pci_bus_id) {
                Some(device) => crate::telemetry::print_probe(&device),
                None => println!("  {:<18}no NVML device at {}", "telemetry", info.pci_bus_id),
            },
            Err(_) => println!("  {:<18}NVML unavailable", "telemetry"),
        }
    }
    Ok(())
}
