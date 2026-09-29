//! Backend-agnostic telemetry: sensor readings for progress lines, statistics
//! for the summary, and hardware errors that fail a GPU.

use std::collections::BTreeSet;
use std::fmt;

/// Reads a GPU's sensors and error counters during a run. GPUs are identified
/// by their index in the run.
pub trait Monitor {
    /// Current sensor readings, or `None` when the GPU can't be read.
    fn reading(&self, gpu: usize) -> Option<Reading>;
    /// Hardware errors since the run started.
    fn errors(&self, gpu: usize) -> HardwareErrors;
}

/// A monitor for runs without telemetry.
pub struct NoMonitor;

impl Monitor for NoMonitor {
    fn reading(&self, _gpu: usize) -> Option<Reading> {
        None
    }

    fn errors(&self, _gpu: usize) -> HardwareErrors {
        HardwareErrors::default()
    }
}

/// A point-in-time reading. Fields the GPU doesn't report are `None`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reading {
    pub temperature_c: Option<u32>,
    pub power_w: Option<f64>,
    pub sm_clock_mhz: Option<u32>,
    /// Active reasons the GPU is running below its maximum clocks, such as `sw-power-cap`.
    pub throttle: Vec<String>,
}

impl fmt::Display for Reading {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
        if !self.throttle.is_empty() {
            write!(f, "  throttled: {}", self.throttle.join(", "))?;
        }
        Ok(())
    }
}

/// Hardware errors reported during a run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HardwareErrors {
    /// New corrected ECC memory errors, or `None` when the GPU doesn't report ECC.
    pub ecc_corrected: Option<u64>,
    /// New uncorrected ECC memory errors, or `None` when the GPU doesn't report ECC.
    pub ecc_uncorrected: Option<u64>,
    /// Critical driver errors (Xid codes) in the order they happened, or
    /// `None` when they can't be watched.
    pub xids: Option<Vec<u64>>,
}

impl HardwareErrors {
    fn xids(&self) -> &[u64] {
        self.xids.as_deref().unwrap_or_default()
    }

    /// Uncorrected memory errors and critical driver errors mean a GPU can't be
    /// trusted, even if every result matched. Corrected errors are only reported.
    pub fn is_failure(&self) -> bool {
        self.ecc_uncorrected.unwrap_or(0) > 0 || !self.xids().is_empty()
    }

    /// What made this a failure, such as `2 uncorrected ECC errors, Xid 79`.
    pub fn failure(&self) -> String {
        let mut parts = Vec::new();
        if let Some(count) = self.ecc_uncorrected.filter(|&count| count > 0) {
            parts.push(format!("{count} uncorrected ECC errors"));
        }
        if !self.xids().is_empty() {
            parts.push(format!("{} from the driver", xid_list(self.xids())));
        }
        parts.join(", ")
    }

    /// Whether the GPU reports anything at all; if not there is nothing to summarise.
    pub fn is_monitored(&self) -> bool {
        self.ecc_corrected.is_some() || self.ecc_uncorrected.is_some() || self.xids.is_some()
    }

    /// One line for the summary.
    pub fn summary(&self) -> String {
        let ecc = match (self.ecc_corrected, self.ecc_uncorrected) {
            (Some(corrected), Some(uncorrected)) => {
                format!("ECC {corrected} corrected, {uncorrected} uncorrected")
            }
            _ => "ECC not reported".to_string(),
        };
        let driver = match &self.xids {
            None => "driver errors not watched".to_string(),
            Some(xids) if xids.is_empty() => "no driver errors".to_string(),
            Some(xids) => format!("driver errors {}", xid_list(xids)),
        };
        format!("{ecc}; {driver}")
    }

    /// Describes what is new since `before`, for printing as it happens.
    pub fn changes_since(&self, before: &Self) -> Vec<String> {
        let mut changes: Vec<String> = self
            .xids()
            .iter()
            .skip(before.xids().len())
            .map(|xid| format!("the driver reported critical error Xid {xid}"))
            .collect();
        let grew = |now: Option<u64>, then: Option<u64>| {
            now.map(|now| (now, now.saturating_sub(then.unwrap_or(0))))
                .filter(|&(_, new)| new > 0)
        };
        if let Some((total, new)) = grew(self.ecc_uncorrected, before.ecc_uncorrected) {
            changes.push(format!(
                "{new} new uncorrected ECC errors ({total} this run)"
            ));
        }
        if let Some((total, new)) = grew(self.ecc_corrected, before.ecc_corrected) {
            changes.push(format!("{new} new corrected ECC errors ({total} this run)"));
        }
        changes
    }
}

fn xid_list(xids: &[u64]) -> String {
    let codes: Vec<String> = xids.iter().map(u64::to_string).collect();
    format!("Xid {}", codes.join(", "))
}

/// Readings accumulated over a run, for the summary.
#[derive(Default)]
pub struct Stats {
    temperature_max: Option<u32>,
    power_total: f64,
    power_samples: u32,
    power_max: f64,
    clock_total: u64,
    clock_samples: u32,
    clock_min: Option<u32>,
    throttle: BTreeSet<String>,
}

impl Stats {
    pub fn add(&mut self, reading: &Reading) {
        if let Some(t) = reading.temperature_c {
            self.temperature_max = Some(self.temperature_max.map_or(t, |max| max.max(t)));
        }
        if let Some(w) = reading.power_w {
            self.power_total += w;
            self.power_samples += 1;
            self.power_max = self.power_max.max(w);
        }
        if let Some(mhz) = reading.sm_clock_mhz {
            self.clock_total += u64::from(mhz);
            self.clock_samples += 1;
            self.clock_min = Some(self.clock_min.map_or(mhz, |min| min.min(mhz)));
        }
        self.throttle.extend(reading.throttle.iter().cloned());
    }

    /// One line for the summary, or `None` when nothing was read.
    pub fn summary(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(max) = self.temperature_max {
            parts.push(format!("{max} C peak"));
        }
        if self.power_samples > 0 {
            parts.push(format!(
                "{:.0} W average, {:.0} W peak",
                self.power_total / f64::from(self.power_samples),
                self.power_max
            ));
        }
        if let Some(min) = self.clock_min {
            parts.push(format!(
                "{} MHz average, {min} MHz lowest",
                self.clock_total / u64::from(self.clock_samples)
            ));
        }
        if !self.throttle.is_empty() {
            let reasons: Vec<&str> = self.throttle.iter().map(String::as_str).collect();
            parts.push(format!("throttled: {}", reasons.join(", ")));
        }
        (!parts.is_empty()).then(|| parts.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(temperature_c: u32, power_w: f64, sm_clock_mhz: u32, throttle: &[&str]) -> Reading {
        Reading {
            temperature_c: Some(temperature_c),
            power_w: Some(power_w),
            sm_clock_mhz: Some(sm_clock_mhz),
            throttle: throttle.iter().map(|reason| reason.to_string()).collect(),
        }
    }

    #[test]
    fn formats_a_reading() {
        assert_eq!(
            reading(64, 182.4, 1980, &[]).to_string(),
            " 64 C    182 W  1980 MHz"
        );
        assert_eq!(Reading::default().to_string(), " -- C     -- W    -- MHz");
        assert!(
            reading(90, 700.0, 1500, &["hw-slowdown"])
                .to_string()
                .ends_with("throttled: hw-slowdown")
        );
    }

    #[test]
    fn summarises_readings() {
        let mut stats = Stats::default();
        assert_eq!(stats.summary(), None);
        stats.add(&reading(60, 600.0, 1900, &[]));
        stats.add(&reading(71, 700.0, 1700, &["sw-power-cap"]));
        stats.add(&Reading::default());
        assert_eq!(
            stats.summary().unwrap(),
            "71 C peak; 650 W average, 700 W peak; 1800 MHz average, 1700 MHz lowest; \
             throttled: sw-power-cap"
        );
    }

    #[test]
    fn only_serious_errors_fail_a_gpu() {
        let corrected = HardwareErrors {
            ecc_corrected: Some(3),
            ecc_uncorrected: Some(0),
            xids: Some(vec![]),
        };
        assert!(!corrected.is_failure());
        assert_eq!(
            corrected.summary(),
            "ECC 3 corrected, 0 uncorrected; no driver errors"
        );

        let uncorrected = HardwareErrors {
            ecc_uncorrected: Some(2),
            ..corrected.clone()
        };
        assert!(uncorrected.is_failure());
        assert_eq!(uncorrected.failure(), "2 uncorrected ECC errors");

        let xid = HardwareErrors {
            xids: Some(vec![79, 48]),
            ..corrected
        };
        assert!(xid.is_failure());
        assert_eq!(xid.failure(), "Xid 79, 48 from the driver");
    }

    #[test]
    fn unmonitored_errors() {
        let none = HardwareErrors::default();
        assert!(!none.is_failure());
        assert!(!none.is_monitored());
        assert_eq!(
            none.summary(),
            "ECC not reported; driver errors not watched"
        );
    }

    #[test]
    fn reports_only_what_is_new() {
        let before = HardwareErrors {
            ecc_corrected: Some(1),
            ecc_uncorrected: Some(0),
            xids: Some(vec![13]),
        };
        let after = HardwareErrors {
            ecc_corrected: Some(4),
            ecc_uncorrected: Some(0),
            xids: Some(vec![13, 79]),
        };
        assert_eq!(
            after.changes_since(&before),
            vec![
                "the driver reported critical error Xid 79".to_string(),
                "3 new corrected ECC errors (4 this run)".to_string(),
            ]
        );
        assert!(after.changes_since(&after).is_empty());
    }
}
