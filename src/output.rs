//! Everything a run prints, as typed records that are rendered either for
//! people or as JSON Lines for automation.

use std::io::{self, Write};
use std::time::Duration;

use clap::ValueEnum;
use serde::Serialize;

use crate::monitor::{HardwareErrors, Reading, TelemetrySummary};
use crate::units::format_duration;

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// Progress lines and a summary for people to read.
    Human,
    /// One JSON object per line, ending with a `summary` object.
    Json,
}

/// Something that happened during a run. GPUs are identified by device number.
#[derive(Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Record<'a> {
    /// The run is starting.
    Start {
        version: &'a str,
        precision: &'a str,
        gpus: usize,
    },
    /// A GPU that will be tested.
    Gpu {
        gpu: usize,
        name: &'a str,
        detail: &'a str,
    },
    /// A GPU is set up and testing has begun.
    Ready { gpu: usize, detail: &'a str },
    /// Every GPU is ready and the clock has started.
    Running { duration_secs: f64 },
    Progress {
        elapsed_secs: f64,
        gpu: usize,
        pass: u64,
        /// Throughput since the previous progress record, when it can be measured.
        tflops: Option<f64>,
        mismatches: u64,
        reading: Option<&'a Reading>,
    },
    /// Results `first..=last` of a pass had `count` values that differ from the reference.
    Mismatch {
        gpu: usize,
        pass: u64,
        first: usize,
        last: usize,
        count: u64,
    },
    /// A new hardware error, such as an ECC error or driver Xid event.
    Hardware { gpu: usize, message: &'a str },
    /// A GPU has made no progress for a while.
    Stalled { gpu: usize, quiet_secs: f64 },
    /// A stalled GPU is making progress again.
    Recovered { gpu: usize },
    /// burnin has given up on a GPU.
    Hung { gpu: usize, reason: &'a str },
    Error {
        gpu: usize,
        /// Whether testing had already begun.
        during_run: bool,
        message: &'a str,
    },
    /// The last record of a run.
    Summary {
        elapsed_secs: f64,
        /// `pass`, `fail` or `error`, matching the exit status.
        result: &'a str,
        exit_status: u8,
        gpus: &'a [GpuSummary],
    },
}

/// One GPU's outcome.
#[derive(Serialize)]
pub struct GpuSummary {
    pub gpu: usize,
    pub name: String,
    /// `pass`, `fail`, `hung` or `error`.
    pub verdict: &'static str,
    /// Explains the verdict.
    pub detail: String,
    pub gemms: u64,
    pub mismatches: u64,
    pub tflops_average: Option<f64>,
    pub telemetry: Option<TelemetrySummary>,
    pub hardware_errors: HardwareErrors,
}

/// Writes records in the chosen format.
pub struct Output<W> {
    format: Format,
    writer: W,
}

impl<W: Write> Output<W> {
    pub fn new(format: Format, writer: W) -> Self {
        Self { format, writer }
    }

    #[cfg(test)]
    pub fn into_inner(self) -> W {
        self.writer
    }

    /// Writes a record. Errors such as a closed pipe are ignored: losing output
    /// must not stop a test that is still running.
    pub fn emit(&mut self, record: Record) {
        let _ = match self.format {
            Format::Human => write_human(&mut self.writer, &record),
            Format::Json => serde_json::to_writer(&mut self.writer, &record)
                .map_err(io::Error::from)
                .and_then(|()| writeln!(self.writer)),
        };
    }
}

fn secs(secs: f64) -> Duration {
    Duration::from_secs_f64(secs.max(0.0))
}

fn write_human(w: &mut impl Write, record: &Record) -> io::Result<()> {
    match *record {
        Record::Start {
            precision, gpus, ..
        } => {
            let noun = if gpus == 1 { "GPU" } else { "GPUs" };
            writeln!(
                w,
                "testing {gpus} {noun} with {precision}; Ctrl-C stops early"
            )
        }
        Record::Gpu { gpu, name, detail } => writeln!(w, "gpu {gpu}  {name}  ({detail})"),
        Record::Ready { gpu, detail } => writeln!(w, "gpu {gpu}  ready: {detail}"),
        Record::Running { duration_secs } => {
            writeln!(w, "running for {}", format_duration(secs(duration_secs)))
        }
        Record::Progress {
            elapsed_secs,
            gpu,
            pass,
            tflops,
            mismatches,
            reading,
        } => {
            let rate = match tflops {
                Some(tflops) => format!("{tflops:>8.2}"),
                None => format!("{:>8}", "--"),
            };
            let line = format!(
                "{:>8}  gpu {gpu:<2} pass {pass:<4} {rate} TFLOP/s  mismatches {mismatches:<6} {}",
                format_duration(secs(elapsed_secs)),
                reading.map(Reading::to_string).unwrap_or_default()
            );
            writeln!(w, "{}", line.trim_end())
        }
        Record::Mismatch {
            gpu,
            pass,
            first,
            last,
            count,
        } => writeln!(
            w,
            "MISMATCH  gpu {gpu}  pass {pass}, results {first}-{last}: \
             {count} values differ from the reference"
        ),
        Record::Hardware { gpu, message } => writeln!(w, "HARDWARE  gpu {gpu}  {message}"),
        Record::Stalled { gpu, quiet_secs } => writeln!(
            w,
            "WARNING  gpu {gpu}  no progress for {}",
            format_duration(secs(quiet_secs))
        ),
        Record::Recovered { gpu } => writeln!(w, "gpu {gpu}  progressing again"),
        Record::Hung { gpu, reason } => writeln!(w, "HUNG  gpu {gpu}  {reason}; giving up on it"),
        Record::Error {
            gpu,
            during_run,
            message,
        } => {
            let when = if during_run {
                "failed during the run"
            } else {
                "could not start"
            };
            writeln!(w, "ERROR  gpu {gpu}  {when}: {message}")
        }
        Record::Summary {
            elapsed_secs, gpus, ..
        } => write_human_summary(w, elapsed_secs, gpus),
    }
}

fn write_human_summary(
    w: &mut impl Write,
    elapsed_secs: f64,
    gpus: &[GpuSummary],
) -> io::Result<()> {
    let name_width = gpus.iter().map(|gpu| gpu.name.len()).max().unwrap_or(0);
    writeln!(w)?;
    writeln!(w, "summary after {}", format_duration(secs(elapsed_secs)))?;
    for gpu in gpus {
        writeln!(
            w,
            "  gpu {:<2} {:<name_width$}  {:<5}  {}",
            gpu.gpu,
            gpu.name,
            gpu.verdict.to_ascii_uppercase(),
            gpu.detail
        )?;
        // Further lines line up under the GPU's name.
        if let Some(telemetry) = &gpu.telemetry {
            writeln!(w, "         {telemetry}")?;
        }
        if gpu.hardware_errors.is_monitored() {
            writeln!(w, "         {}", gpu.hardware_errors.summary())?;
        }
    }

    let total = gpus.len();
    let count = |verdicts: &[&str]| {
        gpus.iter()
            .filter(|gpu| verdicts.contains(&gpu.verdict))
            .count()
    };
    let failed = count(&["fail", "hung"]);
    let untested = count(&["error"]);
    if failed > 0 {
        writeln!(w, "FAIL  {failed} of {total} GPU(s) failed")
    } else if untested > 0 {
        writeln!(w, "ERROR  {untested} of {total} GPU(s) could not be tested")
    } else {
        writeln!(w, "PASS  all {total} GPU(s) passed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(format: Format, record: Record) -> String {
        let mut output = Output::new(format, Vec::new());
        output.emit(record);
        String::from_utf8(output.into_inner()).unwrap()
    }

    #[test]
    fn progress_for_people() {
        let reading = Reading {
            temperature_c: Some(64),
            power_w: Some(182.0),
            sm_clock_mhz: Some(1980),
            throttle: vec![],
        };
        let record = Record::Progress {
            elapsed_secs: 12.34,
            gpu: 1,
            pass: 2,
            tflops: Some(51.2),
            mismatches: 0,
            reading: Some(&reading),
        };
        assert_eq!(
            render(Format::Human, record),
            "   12.3s  gpu 1  pass 2       51.20 TFLOP/s  mismatches 0       64 C    182 W  1980 MHz\n"
        );
    }

    #[test]
    fn progress_as_json() {
        let record = Record::Progress {
            elapsed_secs: 5.0,
            gpu: 0,
            pass: 1,
            tflops: None,
            mismatches: 3,
            reading: None,
        };
        let line = render(Format::Json, record);
        assert!(line.ends_with('\n'));
        let value: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(value["event"], "progress");
        assert_eq!(value["gpu"], 0);
        assert_eq!(value["mismatches"], 3);
        assert!(value["tflops"].is_null());
        assert!(value["reading"].is_null());
    }

    #[test]
    fn summary_result_line() {
        let gpu = |verdict| GpuSummary {
            gpu: 0,
            name: "Mock".to_string(),
            verdict,
            detail: "detail".to_string(),
            gemms: 1,
            mismatches: 0,
            tflops_average: None,
            telemetry: None,
            hardware_errors: HardwareErrors::default(),
        };
        let summary = |gpus: &[GpuSummary]| {
            render(
                Format::Human,
                Record::Summary {
                    elapsed_secs: 1.0,
                    result: "",
                    exit_status: 0,
                    gpus,
                },
            )
        };
        assert!(summary(&[gpu("pass")]).ends_with("PASS  all 1 GPU(s) passed\n"));
        assert!(summary(&[gpu("pass"), gpu("hung")]).ends_with("FAIL  1 of 2 GPU(s) failed\n"));
        assert!(
            summary(&[gpu("pass"), gpu("error")])
                .ends_with("ERROR  1 of 2 GPU(s) could not be tested\n")
        );
    }
}
