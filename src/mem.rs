//! Deciding how much GPU memory a run may use.

use std::fmt;
use std::str::FromStr;

use crate::units::format_bytes;

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// Default share of usable memory a run takes. A Mac is often a laptop in use
/// while it is tested, and macOS counts the memory apps are using as available,
/// so there a run takes half.
pub const DEFAULT_PERCENT: f64 = if cfg!(target_os = "macos") {
    50.0
} else {
    90.0
};

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

/// The system memory figures that matter for unified-memory GPUs, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostMemory {
    pub total: u64,
    pub free: u64,
    pub available: u64,
}

impl HostMemory {
    /// Parses the text of `/proc/meminfo`.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
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

    /// Reads the kernel's memory figures. macOS keeps RAM full of cache and
    /// makes room for new memory by dropping cache and compressing idle pages,
    /// so memory counts as available unless it is wired or already compressed.
    /// That is the share `memory_pressure` reports as free.
    #[cfg(target_os = "macos")]
    pub fn read() -> std::io::Result<Self> {
        let total = sysctl_number(c"hw.memsize")?;
        let page_size = sysctl_number(c"hw.pagesize")?;
        let free_pages = sysctl_number(c"vm.page_free_count")?;
        let available_percent = sysctl_number(c"kern.memorystatus_level")?.min(100);
        Ok(Self {
            total,
            free: free_pages * page_size,
            available: total / 100 * available_percent,
        })
    }

    /// Memory left for the OS when the GPU shares system RAM: 10% of RAM, and at least 2 GiB.
    pub fn os_reserve(&self) -> u64 {
        (self.total / 10).max(2 * GIB)
    }
}

/// Reads a numeric sysctl, which the kernel stores in 4 or 8 bytes.
#[cfg(target_os = "macos")]
fn sysctl_number(name: &std::ffi::CStr) -> std::io::Result<u64> {
    let mut value = [0u8; 8];
    let mut len = value.len();
    // SAFETY: `name` is NUL-terminated, and the kernel writes at most `len`
    // bytes into `value`, then stores how many it wrote in `len`.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            value.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 {
        return Err(std::io::Error::last_os_error());
    }
    match len {
        4 => Ok(u32::from_ne_bytes([value[0], value[1], value[2], value[3]]).into()),
        8 => Ok(u64::from_ne_bytes(value)),
        _ => Err(std::io::Error::other(format!(
            "unexpected size for sysctl {}",
            name.to_string_lossy()
        ))),
    }
}

/// What a memory budget was calculated from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    /// The user asked for an exact size.
    Explicit,
    /// The free memory the GPU reports.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    DeviceFree { free: u64 },
    /// The working set Metal recommends for the GPU, less what this process
    /// already has allocated on it.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    WorkingSet { recommended: u64, in_use: u64 },
    /// Available system RAM minus a reserve for the OS, split between the
    /// unified-memory GPUs under test.
    HostAvailable {
        available: u64,
        reserve: u64,
        shared_by: u64,
    },
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
/// from `MemAvailable` minus a reserve for the OS instead, split evenly between
/// the `host_sharers` unified-memory GPUs being tested at once.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub fn budget(
    spec: MemSpec,
    device_free: u64,
    integrated: bool,
    host: Option<HostMemory>,
    host_sharers: u64,
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
            let shared_by = host_sharers.max(1);
            (
                host.available.saturating_sub(reserve) / shared_by,
                Basis::HostAvailable {
                    available: host.available,
                    reserve,
                    shared_by,
                },
            )
        }
        _ => (device_free, Basis::DeviceFree { free: device_free }),
    };
    Budget {
        bytes: percent_of(base, percent),
        basis,
    }
}

/// Picks how many bytes a run may allocate on a Metal device.
///
/// Metal recommends a working set for each GPU: roughly the most memory it can
/// use before performance suffers. The budget comes from what is left of that
/// after `in_use`, the memory this process already has on the device. A GPU
/// that shares system RAM is also held to the available system memory less a
/// reserve for the OS, so that a busy Mac is not pushed into swap. Whichever
/// is smaller becomes the basis.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub fn working_set_budget(
    spec: MemSpec,
    recommended: u64,
    in_use: u64,
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
    let mut base = recommended.saturating_sub(in_use);
    let mut basis = Basis::WorkingSet {
        recommended,
        in_use,
    };
    if let Some(host) = host {
        let reserve = host.os_reserve();
        let usable = host.available.saturating_sub(reserve);
        if usable < base {
            base = usable;
            basis = Basis::HostAvailable {
                available: host.available,
                reserve,
                shared_by: 1,
            };
        }
    }
    Budget {
        bytes: percent_of(base, percent),
        basis,
    }
}

fn percent_of(bytes: u64, percent: f64) -> u64 {
    (bytes as f64 * percent / 100.0) as u64
}

/// Describes how a budget was reached, for example `90% of 79.2 GiB free on the device`.
pub fn describe(spec: MemSpec, budget: &Budget) -> String {
    match (spec, budget.basis) {
        (_, Basis::Explicit) => "as requested".to_string(),
        (MemSpec::Percent(percent), Basis::DeviceFree { free }) => {
            format!("{percent}% of {} free on the device", format_bytes(free))
        }
        (
            MemSpec::Percent(percent),
            Basis::WorkingSet {
                recommended,
                in_use,
            },
        ) => {
            let in_use = if in_use >= MIB {
                format!(", less {} already in use", format_bytes(in_use))
            } else {
                String::new()
            };
            format!(
                "{percent}% of the {} recommended working set{in_use}",
                format_bytes(recommended)
            )
        }
        (
            MemSpec::Percent(percent),
            Basis::HostAvailable {
                available,
                reserve,
                shared_by,
            },
        ) => {
            let split = if shared_by > 1 {
                format!(", split between {shared_by} GPUs")
            } else {
                String::new()
            };
            format!(
                "{percent}% of {} available system memory, less a {} reserve for the OS{split}",
                format_bytes(available),
                format_bytes(reserve)
            )
        }
        (MemSpec::Bytes(_), _) => unreachable!("an explicit size always has an explicit basis"),
    }
}

/// Number of result matrices that fit in `budget` alongside the two input
/// matrices. Inputs can be smaller than results, as with FP8.
pub fn result_slots(budget: u64, input_bytes: u64, result_bytes: u64) -> u64 {
    budget.saturating_sub(2 * input_bytes) / result_bytes
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
        let b = budget(MemSpec::Percent(50.0), 80 * GIB, false, None, 1);
        assert_eq!(b.bytes, 40 * GIB);
        assert_eq!(b.basis, Basis::DeviceFree { free: 80 * GIB });
    }

    #[test]
    fn unified_gpu_uses_available_host_memory() {
        let host = HostMemory::parse(MEMINFO).unwrap();
        // The device reports only ~1.6 GiB free, but ~62 GiB is really available.
        let b = budget(MemSpec::Percent(100.0), host.free, true, Some(host), 1);
        assert_eq!(b.bytes, host.available - host.os_reserve());
        assert!(b.bytes > 40 * GIB);
    }

    #[test]
    fn unified_gpus_split_host_memory() {
        let host = HostMemory::parse(MEMINFO).unwrap();
        let alone = budget(MemSpec::Percent(100.0), host.free, true, Some(host), 1);
        let shared = budget(MemSpec::Percent(100.0), host.free, true, Some(host), 2);
        assert_eq!(shared.bytes, alone.bytes / 2);
        assert!(describe(MemSpec::Percent(100.0), &shared).ends_with("split between 2 GPUs"));
    }

    #[test]
    fn unified_gpu_without_meminfo_falls_back_to_device() {
        let b = budget(MemSpec::Percent(100.0), 8 * GIB, true, None, 1);
        assert_eq!(b.basis, Basis::DeviceFree { free: 8 * GIB });
    }

    #[test]
    fn explicit_size_wins() {
        let host = HostMemory::parse(MEMINFO).unwrap();
        let b = budget(MemSpec::Bytes(3 * GIB), 80 * GIB, true, Some(host), 1);
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
    fn metal_gpu_uses_what_is_left_of_the_working_set() {
        let b = working_set_budget(MemSpec::Percent(50.0), 12 * GIB, 2 * GIB, None);
        assert_eq!(b.bytes, 5 * GIB);
        assert_eq!(
            b.basis,
            Basis::WorkingSet {
                recommended: 12 * GIB,
                in_use: 2 * GIB
            }
        );
        assert_eq!(
            describe(MemSpec::Percent(50.0), &b),
            "50% of the 12.0 GiB recommended working set, less 2.0 GiB already in use"
        );
        let idle = working_set_budget(MemSpec::Percent(90.0), 12 * GIB, 0, None);
        assert_eq!(
            describe(MemSpec::Percent(90.0), &idle),
            "90% of the 12.0 GiB recommended working set"
        );
    }

    #[test]
    fn busy_mac_is_held_to_available_memory() {
        let host = HostMemory {
            total: 16 * GIB,
            free: 0,
            available: 6 * GIB,
        };
        let b = working_set_budget(MemSpec::Percent(100.0), 12 * GIB, 0, Some(host));
        assert_eq!(b.bytes, 4 * GIB);
        assert_eq!(
            b.basis,
            Basis::HostAvailable {
                available: 6 * GIB,
                reserve: 2 * GIB,
                shared_by: 1
            }
        );
        // Less busy, the working set is the smaller limit again.
        let idle = HostMemory {
            available: 15 * GIB,
            ..host
        };
        let b = working_set_budget(MemSpec::Percent(100.0), 12 * GIB, 0, Some(idle));
        assert_eq!(b.bytes, 12 * GIB);
        // Nothing to spare leaves nothing to take.
        let full = HostMemory {
            available: GIB,
            ..host
        };
        assert_eq!(
            working_set_budget(MemSpec::Percent(100.0), 12 * GIB, 0, Some(full)).bytes,
            0
        );
    }

    #[test]
    fn explicit_size_wins_on_metal() {
        let b = working_set_budget(MemSpec::Bytes(3 * GIB), 12 * GIB, 0, None);
        assert_eq!(
            b,
            Budget {
                bytes: 3 * GIB,
                basis: Basis::Explicit
            }
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn reads_host_memory_on_macos() {
        let host = HostMemory::read().unwrap();
        assert!(host.total >= GIB);
        assert!(host.free <= host.total);
        assert!(host.available <= host.total);
    }

    #[test]
    fn counts_result_slots() {
        let matrix = 256 * MIB;
        assert_eq!(result_slots(10 * GIB, matrix, matrix), 38);
        assert_eq!(result_slots(matrix, matrix, matrix), 0);
    }

    #[test]
    fn smaller_inputs_leave_room_for_more_results() {
        // FP8 inputs are a quarter the size of their FP32 results.
        assert_eq!(result_slots(10 * GIB, 64 * MIB, 256 * MIB), 39);
    }
}
