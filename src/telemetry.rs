//! GPU telemetry from NVML.
//!
//! Every reading is optional. Some GPUs report fields as not supported; for
//! example unified-memory parts have no dedicated memory to report.

use std::fmt::Display;

use nvml_wrapper::bitmasks::device::ThrottleReasons;
use nvml_wrapper::enum_wrappers::device::{Clock, EccCounter, MemoryError, TemperatureSensor};
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::{Device, Nvml};

use crate::units::format_bytes;

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

/// A point-in-time reading taken during a run.
pub struct Sample {
    pub temperature_c: Option<u32>,
    pub power_w: Option<f64>,
    pub sm_clock_mhz: Option<u32>,
    pub throttle: Option<ThrottleReasons>,
}

impl Sample {
    pub fn take(device: &Device) -> Self {
        Self {
            temperature_c: device.temperature(TemperatureSensor::Gpu).ok(),
            power_w: device.power_usage().ok().map(milliwatts_to_watts),
            sm_clock_mhz: device.clock_info(Clock::SM).ok(),
            throttle: device.current_throttle_reasons().ok(),
        }
    }
}

impl std::fmt::Display for Sample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.temperature_c {
            Some(t) => write!(f, "{t:>3} C")?,
            None => write!(f, " -- C")?,
        }
        match self.power_w {
            Some(w) => write!(f, "  {w:>5.0} W")?,
            None => write!(f, "     -- W")?,
        }
        match self.sm_clock_mhz {
            Some(mhz) => write!(f, "  {mhz:>4} MHz")?,
            None => write!(f, "    -- MHz")?,
        }
        if let Some(reasons) = self.throttle {
            let reasons = reasons - ThrottleReasons::GPU_IDLE;
            if !reasons.is_empty() {
                write!(f, "  throttled: {}", throttle_names(reasons))?;
            }
        }
        Ok(())
    }
}

fn milliwatts_to_watts(milliwatts: u32) -> f64 {
    f64::from(milliwatts) / 1000.0
}

fn throttle_names(reasons: ThrottleReasons) -> String {
    if reasons.is_empty() {
        return "none".to_string();
    }
    reasons
        .iter_names()
        .map(|(name, _)| name.to_ascii_lowercase().replace('_', "-"))
        .collect::<Vec<_>>()
        .join(", ")
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
}
