//! Metal backend for Apple GPUs: device discovery, probing and running the
//! stress test.

mod burn;

use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use anyhow::{Result, bail};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSProcessInfo;
use objc2_metal::{MTLCopyAllDevices, MTLDevice, MTLDeviceLocation, MTLGPUFamily};
use objc2_metal_performance_shaders::MPSSupportsMTLDevice;

use crate::mem::{self, Budget, HostMemory, MemSpec};
use crate::monitor::NoMonitor;
use crate::output::{Output, Record};
use crate::supervisor::{self, Config, Target, Work};
use crate::units::format_bytes;
use crate::{Isolation, Precision, RunArgs, WorkerArgs, isolation};

/// Longest wait for every GPU to finish setting up before the clock starts anyway.
const MAX_SETUP: Duration = Duration::from_secs(120);
/// Shortest time without progress before a GPU is reported as stalled.
const MIN_STALL: Duration = Duration::from_secs(60);
/// Shortest wait for GPUs to finish their current chunk once the run ends.
const MIN_SHUTDOWN_GRACE: Duration = Duration::from_secs(30);
/// Longest wait for a GPU's process to exit after it has been given up on and killed.
const REAP_GRACE: Duration = Duration::from_secs(5);

/// GPU families from newest to oldest, for naming the newest one a device supports.
const FAMILIES: [(MTLGPUFamily, &str); 5] = [
    (MTLGPUFamily::Apple10, "apple10"),
    (MTLGPUFamily::Apple9, "apple9"),
    (MTLGPUFamily::Apple8, "apple8"),
    (MTLGPUFamily::Apple7, "apple7"),
    (MTLGPUFamily::Mac2, "mac2"),
];

/// A Metal device. Metal devices can be used from any thread.
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;

/// Every Metal device in the system; a device's ordinal is its place in the list.
fn devices() -> Vec<Device> {
    MTLCopyAllDevices().to_vec()
}

/// Static facts about a device.
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    /// The newest GPU family the device supports, such as `apple9`.
    pub family: &'static str,
    /// The GPU shares system RAM with the CPU instead of having its own memory.
    pub unified: bool,
    /// The most memory Metal recommends the GPU use at once.
    pub working_set: u64,
    /// The largest single buffer the GPU can allocate.
    pub max_buffer: u64,
}

impl DeviceInfo {
    fn query(ordinal: usize, device: &Device) -> Self {
        let family = FAMILIES
            .iter()
            .find(|(family, _)| device.supportsFamily(*family))
            .map_or("unknown", |(_, name)| name);
        Self {
            ordinal,
            name: device.name().to_string(),
            family,
            unified: device.hasUnifiedMemory(),
            working_set: device.recommendedMaxWorkingSetSize(),
            max_buffer: device.maxBufferLength() as u64,
        }
    }

    /// Family and memory, e.g. for a run's header.
    fn summary(&self) -> String {
        format!(
            "{}, {} working set{}",
            self.family,
            format_bytes(self.working_set),
            if self.unified { ", unified memory" } else { "" }
        )
    }
}

/// How much memory a run may take on `device`.
fn budget(spec: MemSpec, device: &Device) -> Budget {
    let host = if device.hasUnifiedMemory() {
        HostMemory::read().ok()
    } else {
        None
    };
    mem::working_set_budget(
        spec,
        device.recommendedMaxWorkingSetSize(),
        device.currentAllocatedSize() as u64,
        host,
    )
}

fn location(device: &Device) -> &'static str {
    match device.location() {
        MTLDeviceLocation::BuiltIn => "built-in",
        MTLDeviceLocation::Slot => "slot",
        MTLDeviceLocation::External => "external",
        _ => "unspecified",
    }
}

pub fn list() -> Result<()> {
    let devices = devices();
    if devices.is_empty() {
        println!("no Metal devices found");
    }
    for (ordinal, device) in devices.iter().enumerate() {
        let info = DeviceInfo::query(ordinal, device);
        println!(
            "{:>2}  {:<32} {:<7} {:>10} working set{}",
            info.ordinal,
            info.name,
            info.family,
            format_bytes(info.working_set),
            if info.unified { "  unified memory" } else { "" }
        );
    }
    Ok(())
}

pub fn probe(only: Option<usize>) -> Result<()> {
    let process = NSProcessInfo::processInfo();
    println!("macOS {}", process.operatingSystemVersionString());
    if let Ok(host) = HostMemory::read() {
        println!("host memory");
        println!("  {:<18}{}", "total", format_bytes(host.total));
        println!("  {:<18}{}", "free", format_bytes(host.free));
        println!("  {:<18}{}", "available", format_bytes(host.available));
    }

    let devices = devices();
    let count = devices.len();
    let selected: Vec<usize> = match only {
        Some(ordinal) if ordinal >= count => {
            bail!("device {ordinal} does not exist; found {count} device(s)")
        }
        Some(ordinal) => vec![ordinal],
        None => (0..count).collect(),
    };
    if selected.is_empty() {
        println!("no Metal devices found");
    }

    for ordinal in selected {
        let device = &devices[ordinal];
        let info = DeviceInfo::query(ordinal, device);
        // SAFETY: `device` is a valid Metal device.
        let mps = unsafe { MPSSupportsMTLDevice(Some(device)) };

        println!("device {}: {}", info.ordinal, info.name);
        println!("  {:<18}{}", "gpu family", info.family);
        println!("  {:<18}{}", "location", location(device));
        println!(
            "  {:<18}{}",
            "unified memory",
            if info.unified { "yes" } else { "no" }
        );
        println!(
            "  {:<18}{} recommended",
            "working set",
            format_bytes(info.working_set)
        );
        println!("  {:<18}{}", "max buffer", format_bytes(info.max_buffer));
        println!(
            "  {:<18}{}",
            "mps",
            if mps { "supported" } else { "not supported" }
        );

        let spec = MemSpec::default();
        let budget = budget(spec, device);
        let square = (crate::DEFAULT_MATRIX_SIZE * crate::DEFAULT_MATRIX_SIZE) as u64;
        println!(
            "  {:<18}{} ({}): {} fp32 result matrices",
            "default budget",
            format_bytes(budget.bytes),
            mem::describe(spec, &budget),
            mem::result_slots(budget.bytes, square * 4, square * 4),
        );
        println!("  {:<18}not available for Apple GPUs yet", "telemetry");
    }
    Ok(())
}

/// Tests the selected GPUs in parallel and returns the process exit status.
pub fn run(args: &RunArgs) -> Result<u8> {
    match args.precision {
        Precision::Fp32 => {}
        Precision::Fp64 => bail!("Apple GPUs do not support fp64; use --precision fp32"),
        other => bail!(
            "{} is not supported on Apple GPUs yet; use --precision fp32",
            other.name()
        ),
    }
    if args.matrix_size == 0 || args.matrix_size % burn::ALIGNMENT != 0 {
        bail!(
            "matrix size must be a multiple of {} on Apple GPUs",
            burn::ALIGNMENT
        );
    }
    if args.chunk_secs.is_nan() || args.chunk_secs <= 0.0 {
        bail!("chunk length must be greater than zero");
    }
    if args.tolerance.is_nan() || args.tolerance < 0.0 {
        bail!("tolerance must not be negative");
    }

    let devices = devices();
    if devices.is_empty() {
        bail!("no Metal devices found");
    }
    let ordinals =
        supervisor::select_devices(&args.devices, devices.len()).map_err(anyhow::Error::msg)?;

    let stop = Arc::new(AtomicBool::new(false));
    supervisor::install_stop_handler(stop.clone())?;

    let mut out = Output::new(args.format, std::io::stdout().lock());
    out.emit(Record::Start {
        version: env!("CARGO_PKG_VERSION"),
        precision: args.precision.name(),
        gpus: ordinals.len(),
    });
    let workers = ordinals
        .into_iter()
        .map(|ordinal| {
            let device = devices[ordinal].clone();
            let info = DeviceInfo::query(ordinal, &device);
            let target = Target {
                ordinal,
                name: info.name.clone(),
                detail: info.summary(),
            };
            let work: Work = match args.isolation {
                Isolation::Thread => {
                    let (args, stop) = (args.clone(), stop.clone());
                    Box::new(move |reporter| burn::worker(&device, &args, reporter, &stop))
                }
                // Apple GPUs share system memory with the CPU, but their budget
                // comes from Metal rather than being split here.
                Isolation::Process => {
                    isolation::child_work(args.worker_args(ordinal, 1), stop.clone())
                }
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
    // Progress lines have no telemetry: burnin cannot read it from Apple GPUs yet.
    let summary = supervisor::supervise(workers, &config, stop, &NoMonitor, &mut out);
    Ok(summary.exit_status())
}

/// Tests one GPU as a child process of `burnin run`; see [`crate::isolation`].
pub fn serve_worker(args: &WorkerArgs) -> ExitCode {
    isolation::serve(|stop| -> Work {
        let run = args.run.clone();
        let ordinal = args.ordinal;
        match devices().into_iter().nth(ordinal) {
            Some(device) => Box::new(move |reporter| burn::worker(&device, &run, reporter, &stop)),
            None => Box::new(move |_| bail!("Metal device {ordinal} does not exist")),
        }
    })
}
