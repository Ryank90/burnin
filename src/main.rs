//! burnin: GPU burn-in and stress testing.

// The GPU backends are platform-specific; on other platforms much of the shared
// code is unused.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

mod mem;
mod units;

#[cfg(target_os = "linux")]
mod cuda;
#[cfg(target_os = "linux")]
mod telemetry;

use std::process::ExitCode;
use std::time::Duration;

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::mem::MemSpec;

/// Default width of the square matrices.
pub const DEFAULT_MATRIX_SIZE: usize = 8192;

#[derive(Parser)]
#[command(
    name = "burnin",
    version,
    about = "GPU burn-in and stress testing",
    after_help = "Exit status: 0 when every result matched, 1 when mismatches were found, 2 on error."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List the GPUs burnin can see.
    List,
    /// Print device, memory and telemetry details, useful when bringing up new hardware.
    Probe {
        /// Only probe this device.
        #[arg(short, long)]
        device: Option<usize>,
    },
    /// Stress a GPU and check that every result it computes matches.
    Run(RunArgs),
}

#[derive(Args, Clone, Debug)]
pub struct RunArgs {
    /// How long to run: plain seconds, or with a unit such as 90s, 10m or 2h.
    #[arg(default_value = "60s", value_parser = units::parse_duration)]
    pub duration: Duration,

    /// Device to test.
    #[arg(short, long, default_value_t = 0)]
    pub device: usize,

    /// Arithmetic precision of the matrix multiplies.
    #[arg(short, long, value_enum, default_value_t = Precision::Fp32)]
    pub precision: Precision,

    /// Memory to use: a percentage such as 90%, or a size such as 4096, 512M or 16G (MiB when no unit).
    #[arg(short, long, default_value_t = MemSpec::default())]
    pub mem: MemSpec,

    /// Width of the square matrices.
    #[arg(long, default_value_t = DEFAULT_MATRIX_SIZE)]
    pub matrix_size: usize,

    /// Target length of one verified chunk of work, in seconds.
    #[arg(long, default_value_t = 1.5)]
    pub chunk_secs: f64,

    /// Time between progress lines.
    #[arg(long, default_value = "5s", value_parser = units::parse_duration)]
    pub report_every: Duration,

    /// Largest absolute difference still counted as a match; 0 requires identical results.
    #[arg(long, default_value_t = 0.0)]
    pub tolerance: f64,

    /// Corrupt one result on purpose to check that mismatches are detected.
    #[arg(long)]
    pub inject_fault: bool,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Fp32,
    Fp64,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match dispatch(cli.command) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::from(2)
        }
    }
}

#[cfg(target_os = "linux")]
fn dispatch(command: Command) -> anyhow::Result<ExitCode> {
    match command {
        Command::List => cuda::list().map(|()| ExitCode::SUCCESS),
        Command::Probe { device } => cuda::probe(device).map(|()| ExitCode::SUCCESS),
        Command::Run(args) => {
            let outcome = cuda::run(&args)?;
            Ok(if outcome.passed() {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            })
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn dispatch(_command: Command) -> anyhow::Result<ExitCode> {
    anyhow::bail!("there is no GPU backend for this platform yet; the CUDA backend requires Linux")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_args(argv: &[&str]) -> RunArgs {
        let cli = Cli::try_parse_from(argv).unwrap();
        match cli.command {
            Command::Run(args) => args,
            _ => panic!("not a run command"),
        }
    }

    #[test]
    fn run_defaults() {
        let args = run_args(&["burnin", "run"]);
        assert_eq!(args.duration, Duration::from_secs(60));
        assert_eq!(args.device, 0);
        assert_eq!(args.precision, Precision::Fp32);
        assert_eq!(args.mem, MemSpec::Percent(mem::DEFAULT_PERCENT));
        assert_eq!(args.matrix_size, DEFAULT_MATRIX_SIZE);
        assert!(!args.inject_fault);
    }

    #[test]
    fn duration_can_come_before_or_after_options() {
        let before = run_args(&["burnin", "run", "10m", "-p", "fp64"]);
        let after = run_args(&["burnin", "run", "-p", "fp64", "10m"]);
        assert_eq!(before.duration, Duration::from_secs(600));
        assert_eq!(after.duration, before.duration);
        assert_eq!(after.precision, Precision::Fp64);
    }

    #[test]
    fn memory_forms() {
        assert_eq!(
            run_args(&["burnin", "run", "-m", "50%"]).mem,
            MemSpec::Percent(50.0)
        );
        assert_eq!(
            run_args(&["burnin", "run", "--mem=16G"]).mem,
            MemSpec::Bytes(16 << 30)
        );
    }

    #[test]
    fn rejects_unknown_precision() {
        assert!(Cli::try_parse_from(["burnin", "run", "-p", "fp128"]).is_err());
    }

    #[test]
    fn cli_definition_is_valid() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
