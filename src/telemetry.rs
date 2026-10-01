//! GPU telemetry and hardware errors from NVML.
//!
//! Every reading is optional. Some GPUs report fields as not supported; for
//! example unified-memory parts have no dedicated memory to report.

use std::fmt::Display;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use nvml_wrapper::bitmasks::device::ThrottleReasons;
use nvml_wrapper::bitmasks::event::EventTypes;
use nvml_wrapper::enum_wrappers::device::{Clock, EccCounter, MemoryError, TemperatureSensor};
use nvml_wrapper::enums::event::XidError;
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::event::EventSet;
use nvml_wrapper::struct_wrappers::event::EventData;
use nvml_wrapper::{Device, Nvml};

use crate::monitor::{HardwareErrors, Monitor, NoMonitor, Reading};
use crate::units::format_bytes;

/// How long the Xid watcher waits for an event before checking whether to stop.
const XID_WAIT_MS: u32 = 500;

pub struct Telemetry {
    nvml: Nvml,
}

impl Telemetry {
    pub fn init() -> Result<Self, NvmlError> {
        Ok(Self {
            nvml: Nvml::init()?,
        })
    }

    pub fn driver_version(&self) -> Option<String> {
        self.nvml.sys_driver_version().ok()
    }

    /// Finds the NVML device for a CUDA device. Matching on the PCI address
    /// keeps readings with the right GPU even when CUDA and NVML number
    /// devices differently.
    pub fn device(&self, pci_bus_id: &str) -> Option<Device<'_>> {
        self.nvml.device_by_pci_bus_id(pci_bus_id).ok()
    }
}

/// Watches the GPUs at `pci_bus_ids` through NVML while `body` runs. Without
/// NVML, `body` gets a monitor that reports nothing.
pub fn watch<R>(
    telemetry: Option<&Telemetry>,
    pci_bus_ids: &[String],
    body: impl FnOnce(&dyn Monitor) -> R,
) -> R {
    let Some(telemetry) = telemetry else {
        return body(&NoMonitor);
    };
    let (monitor, events) = NvmlMonitor::new(telemetry, pci_bus_ids);
    let monitor = &monitor;
    thread::scope(|scope| {
        if let Some(events) = events {
            scope.spawn(move || monitor.watch_xids(events));
        }
        let result = body(monitor);
        monitor.stop.store(true, Ordering::SeqCst);
        result
    })
}

/// ECC error totals, where the GPU reports them.
#[derive(Clone, Copy, Default)]
struct EccCounts {
    corrected: Option<u64>,
    uncorrected: Option<u64>,
}

impl EccCounts {
    fn read(device: &Device) -> Self {
        let count = |kind| device.total_ecc_errors(kind, EccCounter::Volatile).ok();
        Self {
            corrected: count(MemoryError::Corrected),
            uncorrected: count(MemoryError::Uncorrected),
        }
    }
}

struct NvmlMonitor<'nvml> {
    devices: Vec<Option<Device<'nvml>>>,
    pci_bus_ids: Vec<String>,
    /// ECC totals when the run started, so only new errors are counted.
    baseline: Vec<EccCounts>,
    /// Xid codes seen for each GPU, or `None` where they can't be watched.
    xids: Mutex<Vec<Option<Vec<u64>>>>,
    stop: AtomicBool,
}

impl<'nvml> NvmlMonitor<'nvml> {
    /// Records the starting ECC counts and registers for critical Xid events.
    /// Returns the event set to watch, if any GPU supports them.
    fn new(telemetry: &'nvml Telemetry, pci_bus_ids: &[String]) -> (Self, Option<EventSet<'nvml>>) {
        let devices: Vec<_> = pci_bus_ids.iter().map(|id| telemetry.device(id)).collect();
        let baseline = devices
            .iter()
            .map(|device| device.as_ref().map(EccCounts::read).unwrap_or_default())
            .collect();

        let mut xids = vec![None; devices.len()];
        let mut set = telemetry.nvml.create_event_set().ok();
        for (index, device) in devices.iter().enumerate() {
            let (Some(device), Some(unregistered)) = (device, set.take()) else {
                continue;
            };
            match device.register_events(EventTypes::CRITICAL_XID_ERROR, unregistered) {
                Ok(registered) => {
                    set = Some(registered);
                    xids[index] = Some(Vec::new());
                }
                // A failed registration frees the set, so start a new one.
                Err(_) => set = telemetry.nvml.create_event_set().ok(),
            }
        }
        let watching = xids.iter().any(Option::is_some);
        let monitor = Self {
            devices,
            pci_bus_ids: pci_bus_ids.to_vec(),
            baseline,
            xids: Mutex::new(xids),
            stop: AtomicBool::new(false),
        };
        (monitor, set.filter(|_| watching))
    }

    fn watch_xids(&self, events: EventSet<'nvml>) {
        while !self.stop.load(Ordering::SeqCst) {
            match events.wait(XID_WAIT_MS) {
                Ok(event) => self.record(&event),
                Err(NvmlError::Timeout) => {}
                // Don't spin if waiting keeps failing.
                Err(_) => thread::sleep(Duration::from_millis(u64::from(XID_WAIT_MS))),
            }
        }
    }

    fn record(&self, event: &EventData) {
        let xid = match event.event_data {
            Some(XidError::Value(xid)) => xid,
            // Still a critical error, just without a code.
            Some(XidError::Unknown) | None => 0,
        };
        let Ok(bus_id) = event.device.pci_info().map(|info| info.bus_id) else {
            return;
        };
        let Some(index) = self
            .pci_bus_ids
            .iter()
            .position(|id| id.eq_ignore_ascii_case(&bus_id))
        else {
            return;
        };
        let mut xids = self
            .xids
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(seen) = xids[index].as_mut() {
            seen.push(xid);
        }
    }
}

impl Monitor for NvmlMonitor<'_> {
    fn reading(&self, gpu: usize) -> Option<Reading> {
        let device = self.devices[gpu].as_ref()?;
        Some(Reading {
            temperature_c: device.temperature(TemperatureSensor::Gpu).ok(),
            power_w: device.power_usage().ok().map(milliwatts_to_watts),
            sm_clock_mhz: device.clock_info(Clock::SM).ok(),
            throttle: device
                .current_throttle_reasons()
                .map(|reasons| throttle_list(reasons - ThrottleReasons::GPU_IDLE))
                .unwrap_or_default(),
        })
    }

    fn errors(&self, gpu: usize) -> HardwareErrors {
        let now = self.devices[gpu]
            .as_ref()
            .map(EccCounts::read)
            .unwrap_or_default();
        let base = self.baseline[gpu];
        let new = |now: Option<u64>, base: Option<u64>| {
            now.zip(base).map(|(now, base)| now.saturating_sub(base))
        };
        let xids = self
            .xids
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        HardwareErrors {
            ecc_corrected: new(now.corrected, base.corrected),
            ecc_uncorrected: new(now.uncorrected, base.uncorrected),
            xids: xids[gpu].clone(),
        }
    }
}

fn milliwatts_to_watts(milliwatts: u32) -> f64 {
    f64::from(milliwatts) / 1000.0
}

fn throttle_list(reasons: ThrottleReasons) -> Vec<String> {
    reasons
        .iter_names()
        .map(|(name, _)| name.to_ascii_lowercase().replace('_', "-"))
        .collect()
}

fn throttle_names(reasons: ThrottleReasons) -> String {
    if reasons.is_empty() {
        return "none".to_string();
    }
    throttle_list(reasons).join(", ")
}

/// Prints every field burnin reads, including the ones this GPU does not support.
pub fn print_probe(device: &Device) {
    fn show<T: Display>(label: &str, value: Result<T, NvmlError>) {
        match value {
            Ok(value) => println!("  {label:<18}{value}"),
            Err(NvmlError::NotSupported) => println!("  {label:<18}not supported"),
            Err(err) => println!("  {label:<18}error: {err}"),
        }
    }

    show(
        "temperature",
        device
            .temperature(TemperatureSensor::Gpu)
            .map(|t| format!("{t} C")),
    );
    show(
        "power draw",
        device
            .power_usage()
            .map(|mw| format!("{:.1} W", milliwatts_to_watts(mw))),
    );
    show(
        "power limit",
        device
            .enforced_power_limit()
            .map(|mw| format!("{:.0} W", milliwatts_to_watts(mw))),
    );
    show(
        "graphics clock",
        device
            .clock_info(Clock::Graphics)
            .map(|c| format!("{c} MHz")),
    );
    show(
        "sm clock",
        device.clock_info(Clock::SM).map(|c| format!("{c} MHz")),
    );
    show(
        "memory clock",
        device.clock_info(Clock::Memory).map(|c| format!("{c} MHz")),
    );
    show(
        "memory",
        device
            .memory_info()
            .map(|m| format!("{} used of {}", format_bytes(m.used), format_bytes(m.total))),
    );
    show(
        "utilization",
        device
            .utilization_rates()
            .map(|u| format!("{}% gpu, {}% memory", u.gpu, u.memory)),
    );
    show(
        "performance state",
        device.performance_state().map(|p| format!("{p:?}")),
    );
    show(
        "throttle reasons",
        device.current_throttle_reasons().map(throttle_names),
    );
    show(
        "ecc mode",
        device
            .is_ecc_enabled()
            .map(|e| if e.currently_enabled { "on" } else { "off" }),
    );
    show(
        "ecc corrected",
        device.total_ecc_errors(MemoryError::Corrected, EccCounter::Volatile),
    );
    show(
        "ecc uncorrected",
        device.total_ecc_errors(MemoryError::Uncorrected, EccCounter::Volatile),
    );
    show(
        "xid events",
        device.supported_event_types().map(|types| {
            if types.contains(EventTypes::CRITICAL_XID_ERROR) {
                "supported"
            } else {
                "not supported"
            }
        }),
    );
}
