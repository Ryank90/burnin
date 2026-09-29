//! Deciding how much GPU memory a run may use.

use std::fmt;
use std::str::FromStr;

use crate::units::format_bytes;

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// Default share of usable memory a run takes.
pub const DEFAULT_PERCENT: f64 = 90.0;

/// How much memory the user asked for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MemSpec {
    /// A percentage of the memory that is safe to take.
    Percent(f64),
    /// An exact number of bytes.
    Bytes(u64),
}

impl Default for MemSpec {
    fn default() -> Self {
        Self::Percent(DEFAULT_PERCENT)
    }
}

impl FromStr for MemSpec {
    type Err = String;

    /// Accepts `90%`, or a size such as `4096`, `512M` or `16G`. Sizes without a unit are MiB.
    fn from_str(text: &str) -> Result<Self, String> {
        let text = text.trim();
        if let Some(percent) = text.strip_suffix('%') {
            let value: f64 = percent
                .trim()
                .parse()
                .map_err(|_| format!("invalid percentage `{text}`"))?;
            if !(value > 0.0 && value <= 100.0) {
                return Err(format!(
                    "percentage must be above 0 and at most 100, got `{text}`"
                ));
            }
            return Ok(Self::Percent(value));
        }

        let split = text
            .find(|c: char| c.is_ascii_alphabetic())
            .unwrap_or(text.len());
        let (number, unit) = text.split_at(split);
        let value: u64 = number
            .trim()
            .parse()
            .map_err(|_| format!("invalid memory size `{text}`"))?;
        let scale = match unit.to_ascii_lowercase().as_str() {
            "" | "m" | "mb" | "mib" => MIB,
            "g" | "gb" | "gib" => GIB,
            other => return Err(format!("unknown size unit `{other}` (use M or G)")),
        };
        if value == 0 {
            return Err("memory size must be greater than zero".into());
        }
        Ok(Self::Bytes(value.saturating_mul(scale)))
    }
}

impl fmt::Display for MemSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Percent(percent) => write!(f, "{percent}%"),
            Self::Bytes(bytes) => write!(f, "{}M", bytes / MIB),
        }
    }
}

/// The parts of `/proc/meminfo` that matter for unified-memory GPUs, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostMemory {
    pub total: u64,
    pub free: u64,
    pub available: u64,
}

impl HostMemory {
    /// Parses the text of `/proc/meminfo`.
    pub fn parse(meminfo: &str) -> Option<Self> {
        let (mut total, mut free, mut available) = (None, None, None);
        for line in meminfo.lines() {
            let mut fields = line.split_whitespace();
            let (Some(key), Some(value)) = (fields.next(), fields.next()) else {
                continue;
            };
            let Ok(kib) = value.parse::<u64>() else {
                continue;
            };
            let bytes = Some(kib * 1024);
            match key {
                "MemTotal:" => total = bytes,
                "MemFree:" => free = bytes,
                "MemAvailable:" => available = bytes,
                _ => {}
            }
        }
        Some(Self {
            total: total?,
            free: free?,
            available: available?,
        })
    }

    #[cfg(target_os = "linux")]
    pub fn read() -> std::io::Result<Self> {
        let text = std::fs::read_to_string("/proc/meminfo")?;
        Self::parse(&text).ok_or_else(|| std::io::Error::other("unexpected /proc/meminfo format"))
    }

    /// Memory left for the OS when the GPU shares system RAM: 10% of RAM, and at least 2 GiB.
    pub fn os_reserve(&self) -> u64 {
        (self.total / 10).max(2 * GIB)
    }
}

/// What a memory budget was calculated from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    /// The user asked for an exact size.
    Explicit,
    /// The free memory the GPU reports.
    DeviceFree { free: u64 },
    /// Available system RAM minus a reserve for the OS (unified-memory GPUs).
    HostAvailable { available: u64, reserve: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub bytes: u64,
    pub basis: Basis,
}

/// Picks how many bytes a run may allocate.
///
/// A unified-memory GPU reports the kernel's `MemFree` as its free memory. That
/// ignores reclaimable page cache, so it can be far below what is really
/// usable, or close to all of the machine's RAM. For those GPUs the budget comes
/// from `MemAvailable` minus a reserve for the OS instead.
pub fn budget(
    spec: MemSpec,
    device_free: u64,
    integrated: bool,
    host: Option<HostMemory>,
) -> Budget {
    let percent = match spec {
        MemSpec::Bytes(bytes) => {
            return Budget {
                bytes,
                basis: Basis::Explicit,
            };
        }
        MemSpec::Percent(percent) => percent,
    };
    let (base, basis) = match host {
        Some(host) if integrated => {
            let reserve = host.os_reserve();
            (
                host.available.saturating_sub(reserve),
                Basis::HostAvailable {
                    available: host.available,
                    reserve,
                },
            )
        }
        _ => (device_free, Basis::DeviceFree { free: device_free }),
    };
    Budget {
        bytes: (base as f64 * percent / 100.0) as u64,
        basis,
    }
}

/// Describes how a budget was reached, for example `90% of 79.2 GiB free on the device`.
pub fn describe(spec: MemSpec, budget: &Budget) -> String {
    match (spec, budget.basis) {
        (_, Basis::Explicit) => "as requested".to_string(),
        (MemSpec::Percent(percent), Basis::DeviceFree { free }) => {
            format!("{percent}% of {} free on the device", format_bytes(free))
        }
        (MemSpec::Percent(percent), Basis::HostAvailable { available, reserve }) => format!(
            "{percent}% of {} available system memory, less a {} reserve for the OS",
            format_bytes(available),
            format_bytes(reserve)
        ),
        (MemSpec::Bytes(_), _) => unreachable!("an explicit size always has an explicit basis"),
    }
}

/// Number of result matrices that fit in `budget` alongside the two input matrices.
pub fn result_slots(budget: u64, matrix_bytes: u64) -> u64 {
    (budget / matrix_bytes).saturating_sub(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MEMINFO: &str = "\
MemTotal:       127535304 kB
MemFree:          1628160 kB
MemAvailable:    64856012 kB
Buffers:           120456 kB
Cached:          61203412 kB
";

    #[test]
    fn parses_specs() {
        assert_eq!("90%".parse(), Ok(MemSpec::Percent(90.0)));
        assert_eq!("12.5 %".parse(), Ok(MemSpec::Percent(12.5)));
        assert_eq!("4096".parse(), Ok(MemSpec::Bytes(4096 * MIB)));
        assert_eq!("512M".parse(), Ok(MemSpec::Bytes(512 * MIB)));
        assert_eq!("16G".parse(), Ok(MemSpec::Bytes(16 * GIB)));
        assert_eq!("16GiB".parse(), Ok(MemSpec::Bytes(16 * GIB)));
    }

    #[test]
    fn rejects_bad_specs() {
        for bad in ["", "0", "0%", "101%", "-5%", "ten", "5T", "%"] {
            assert!(bad.parse::<MemSpec>().is_err(), "accepted `{bad}`");
        }
    }

    #[test]
    fn parses_meminfo() {
        let host = HostMemory::parse(MEMINFO).unwrap();
        assert_eq!(host.total, 127_535_304 * 1024);
        assert_eq!(host.free, 1_628_160 * 1024);
        assert_eq!(host.available, 64_856_012 * 1024);
        assert_eq!(HostMemory::parse("MemTotal: 1 kB\n"), None);
    }

    #[test]
    fn discrete_gpu_uses_device_free_memory() {
        let b = budget(MemSpec::Percent(50.0), 80 * GIB, false, None);
        assert_eq!(b.bytes, 40 * GIB);
        assert_eq!(b.basis, Basis::DeviceFree { free: 80 * GIB });
    }

    #[test]
    fn unified_gpu_uses_available_host_memory() {
        let host = HostMemory::parse(MEMINFO).unwrap();
        // The device reports only ~1.6 GiB free, but ~62 GiB is really available.
        let b = budget(MemSpec::Percent(100.0), host.free, true, Some(host));
        assert_eq!(b.bytes, host.available - host.os_reserve());
        assert!(b.bytes > 40 * GIB);
    }

    #[test]
    fn unified_gpu_without_meminfo_falls_back_to_device() {
        let b = budget(MemSpec::Percent(100.0), 8 * GIB, true, None);
        assert_eq!(b.basis, Basis::DeviceFree { free: 8 * GIB });
    }

    #[test]
    fn explicit_size_wins() {
        let host = HostMemory::parse(MEMINFO).unwrap();
        let b = budget(MemSpec::Bytes(3 * GIB), 80 * GIB, true, Some(host));
        assert_eq!(
            b,
            Budget {
                bytes: 3 * GIB,
                basis: Basis::Explicit
            }
        );
    }

    #[test]
    fn os_reserve_has_a_floor() {
        let small = HostMemory {
            total: 8 * GIB,
            free: 0,
            available: 0,
        };
        assert_eq!(small.os_reserve(), 2 * GIB);
    }

    #[test]
    fn counts_result_slots() {
        let matrix = 256 * MIB;
        assert_eq!(result_slots(10 * GIB, matrix), 38);
        assert_eq!(result_slots(matrix, matrix), 0);
    }
}
