//! CUDA backend: device discovery, probing and running the stress test.

mod burn;

use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use cudarc::driver::sys::{CUdevice, CUdevice_attribute as Attr};
use cudarc::driver::{CudaContext, result as driver};

use crate::mem::{self, HostMemory, MemSpec};
use crate::output::{Output, Record};
use crate::supervisor::{self, Config, Target, Work};
use crate::telemetry::{self, Telemetry};
use crate::units::format_bytes;
use crate::{Isolation, RunArgs, WorkerArgs, isolation};

/// Longest wait for every GPU to finish setting up before the clock starts anyway.
const MAX_SETUP: Duration = Duration::from_secs(120);
/// Shortest time without progress before a GPU is reported as stalled.
const MIN_STALL: Duration = Duration::from_secs(60);
/// Shortest wait for GPUs to finish their current chunk once the run ends.
const MIN_SHUTDOWN_GRACE: Duration = Duration::from_secs(30);
/// Longest wait for a GPU's process to exit after it has been given up on and killed.
const REAP_GRACE: Duration = Duration::from_secs(5);

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

/// Static facts about a device.
#[derive(Clone)]
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
    /// Reads a device's details without creating a context on it.
    fn query(ordinal: usize) -> Result<Self> {
        let dev = driver::device::get(ordinal as i32)
            .with_context(|| format!("could not find device {ordinal}"))?;
        let attribute = |attr: Attr| -> Result<i32> {
            // SAFETY: `dev` is a valid device handle from cuDeviceGet.
            Ok(unsafe { driver::device::get_attribute(dev, attr) }?)
        };
        let domain = attribute(Attr::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)?;
        let bus = attribute(Attr::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)?;
        let device = attribute(Attr::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)?;
        Ok(Self {
            ordinal,
            name: driver::device::get_name(dev)?,
            compute_capability: (
                attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
                attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
            ),
            sm_count: attribute(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32,
            integrated: attribute(Attr::CU_DEVICE_ATTRIBUTE_INTEGRATED)? != 0,
            // SAFETY: as above.
            total_mem: unsafe { driver::device::total_mem(dev) }? as u64,
            pci_bus_id: format!("{domain:08X}:{bus:02X}:{device:02X}.0"),
        })
    }

    pub fn arch(&self) -> String {
        let (major, minor) = self.compute_capability;
        format!("sm_{major}{minor}")
    }

    /// Architecture, size and PCI address, e.g. for a run's header.
    fn summary(&self) -> String {
        format!(
            "{}, {} SMs, {}, {}{}",
            self.arch(),
            self.sm_count,
            format_bytes(self.total_mem),
            self.pci_bus_id,
            if self.integrated {
                ", unified memory"
            } else {
                ""
            }
        )
    }
}

fn raw_device(ordinal: usize) -> Result<CUdevice> {
    Ok(driver::device::get(ordinal as i32)?)
}

pub fn list() -> Result<()> {
    require_libraries(false, false)?;
    let count = device_count()?;
    if count == 0 {
        println!("no CUDA devices found");
    }
    for ordinal in 0..count {
        let info = DeviceInfo::query(ordinal)?;
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
    let infos = (0..count)
        .map(DeviceInfo::query)
        .collect::<Result<Vec<_>>>()?;
    // A default run tests every GPU, so unified-memory GPUs share host memory.
    let host_sharers = infos.iter().filter(|info| info.integrated).count() as u64;
    let selected: Vec<&DeviceInfo> = match only {
        Some(ordinal) if ordinal >= count => {
            bail!("device {ordinal} does not exist; found {count} device(s)")
        }
        Some(ordinal) => vec![&infos[ordinal]],
        None => infos.iter().collect(),
    };
    if selected.is_empty() {
        println!("no CUDA devices found");
    }

    for info in selected {
        let ctx = CudaContext::new(info.ordinal)?;
        let (free, total) = ctx.mem_get_info()?;
        let dev = raw_device(info.ordinal)?;
        let flag = |attr: Attr| -> Result<&'static str> {
            // SAFETY: `dev` is a valid device handle from cuDeviceGet.
            let value = unsafe { driver::device::get_attribute(dev, attr) }?;
            Ok(if value != 0 { "yes" } else { "no" })
        };

        println!("device {}: {}", info.ordinal, info.name);
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
            if info.integrated { "yes" } else { "no" }
        );
        println!(
            "  {:<18}{}",
            "pageable access",
            flag(Attr::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES)?
        );
        println!(
            "  {:<18}{}",
            "ecc",
            flag(Attr::CU_DEVICE_ATTRIBUTE_ECC_ENABLED)?
        );
        println!(
            "  {:<18}{} free of {}",
            "cuda memory",
            format_bytes(free as u64),
            format_bytes(total as u64)
        );

        let spec = MemSpec::default();
        let budget = mem::budget(spec, free as u64, info.integrated, host, host_sharers);
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

/// Tests the selected GPUs in parallel and returns the process exit status.
pub fn run(args: &RunArgs) -> Result<u8> {
    require_libraries(true, true)?;
    if args.matrix_size == 0 || args.matrix_size > i32::MAX as usize {
        bail!("matrix size must be between 1 and {}", i32::MAX);
    }
    if args.chunk_secs.is_nan() || args.chunk_secs <= 0.0 {
        bail!("chunk length must be greater than zero");
    }
    if args.tolerance.is_nan() || args.tolerance < 0.0 {
        bail!("tolerance must not be negative");
    }

    let count = device_count()?;
    if count == 0 {
        bail!("no CUDA devices found");
    }
    let ordinals = supervisor::select_devices(&args.devices, count).map_err(anyhow::Error::msg)?;
    let infos = ordinals
        .into_iter()
        .map(DeviceInfo::query)
        .collect::<Result<Vec<_>>>()?;
    // Worker processes compile their own kernels; threads share one copy.
    let ptx = match args.isolation {
        Isolation::Thread => Some(burn::compile_kernels()?),
        Isolation::Process => None,
    };
    let host_sharers = infos.iter().filter(|info| info.integrated).count() as u64;

    // Telemetry is read in this process, matched to each GPU by PCI address.
    let telemetry = Telemetry::init().ok();
    let pci_bus_ids: Vec<String> = infos.iter().map(|info| info.pci_bus_id.clone()).collect();

    let stop = Arc::new(AtomicBool::new(false));
    install_stop_handler(stop.clone())?;

    let mut out = Output::new(args.format, std::io::stdout().lock());
    out.emit(Record::Start {
        version: env!("CARGO_PKG_VERSION"),
        precision: args.precision.name(),
        gpus: infos.len(),
    });
    let workers = infos
        .into_iter()
        .map(|info| {
            let target = Target {
                ordinal: info.ordinal,
                name: info.name.clone(),
                detail: info.summary(),
            };
            let work: Work = match &ptx {
                Some(ptx) => {
                    let (ptx, args, stop) = (ptx.clone(), args.clone(), stop.clone());
                    Box::new(move |reporter| {
                        burn::worker(&info, ptx, &args, host_sharers, reporter, &stop)
                    })
                }
                None => isolation::child_work(
                    args.worker_args(info.ordinal, host_sharers),
                    stop.clone(),
                ),
            };
            (target, work)
        })
        .collect();

    let config = Config {
        duration: args.duration,
        report_every: args.report_every,
        max_setup: MAX_SETUP,
        min_stall: MIN_STALL,
        min_hang: args.hang_timeout,
        min_shutdown_grace: MIN_SHUTDOWN_GRACE,
        reap_grace: REAP_GRACE,
    };
    let summary = telemetry::watch(telemetry.as_ref(), &pci_bus_ids, |monitor| {
        supervisor::supervise(workers, &config, stop, monitor, &mut out)
    });
    Ok(summary.exit_status())
}

/// Tests one GPU as a child process of `burnin run`; see [`crate::isolation`].
pub fn serve_worker(args: &WorkerArgs) -> ExitCode {
    let setup = || -> Result<(DeviceInfo, cudarc::nvrtc::Ptx)> {
        require_libraries(true, true)?;
        device_count()?;
        Ok((DeviceInfo::query(args.ordinal)?, burn::compile_kernels()?))
    };
    isolation::serve(|stop| -> Work {
        let (run, host_sharers) = (args.run.clone(), args.host_sharers);
        match setup() {
            Ok((info, ptx)) => Box::new(move |reporter| {
                burn::worker(&info, ptx, &run, host_sharers, reporter, &stop)
            }),
            Err(err) => Box::new(move |_| Err(err)),
        }
    })
}

/// The first Ctrl-C (or SIGTERM) lets every GPU finish its current chunk and
/// prints the summary; a second one exits immediately.
fn install_stop_handler(stop: Arc<AtomicBool>) -> Result<()> {
    let presses = AtomicUsize::new(0);
    ctrlc::set_handler(move || {
        if presses.fetch_add(1, Ordering::SeqCst) == 0 {
            eprintln!("\nstopping after the current chunk; press Ctrl-C again to quit now");
            stop.store(true, Ordering::SeqCst);
        } else {
            std::process::exit(130);
        }
    })
    .context("could not install the Ctrl-C handler")
}
