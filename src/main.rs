//! burnin: GPU burn-in and stress testing.

// The GPU backends are platform-specific; on other platforms much of the shared
// code is unused.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

mod mem;
mod supervisor;
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
    after_help = "Exit status: 0 when every GPU passed, 1 when any GPU failed, \
                  2 when a GPU could not be tested or on error."
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
    /// Stress GPUs in parallel and check that every result they compute matches.
    Run(RunArgs),
}

#[derive(Args, Clone, Debug)]
pub struct RunArgs {
    /// How long to run: plain seconds, or with a unit such as 90s, 10m or 2h.
    #[arg(default_value = "60s", value_parser = units::parse_duration)]
    pub duration: Duration,

    /// GPUs to test, as a comma-separated list such as 0,2,3. Every GPU when omitted.
    #[arg(
        short = 'd',
        long,
        visible_alias = "device",
        value_name = "LIST",
        value_delimiter = ','
    )]
    pub devices: Vec<usize>,

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
    /// Single precision.
    Fp32,
    /// FP32 inputs and results, multiplied on TF32 tensor cores. Needs compute capability 8.0 or newer.
    Tf32,
    /// Half precision on tensor cores. Needs compute capability 7.0 or newer.
    Fp16,
    /// Bfloat16 on tensor cores. Needs compute capability 8.0 or newer.
    Bf16,
    /// Double precision.
    Fp64,
    /// FP8 (E4M3) inputs with FP32 results, on tensor cores. Needs compute capability 8.9 or newer.
    Fp8,
}

impl Precision {
    pub fn name(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Tf32 => "tf32",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Fp64 => "fp64",
            Self::Fp8 => "fp8",
        }
    }

    /// Checks that a GPU with this compute capability has tensor cores for
    /// this precision. FP32 and FP64 run on every GPU.
    pub fn check_support(self, (major, minor): (i32, i32)) -> Result<(), String> {
        let needed = match self {
            Self::Fp32 | Self::Fp64 => return Ok(()),
            Self::Fp16 => (7, 0),
            Self::Tf32 | Self::Bf16 => (8, 0),
            Self::Fp8 => (8, 9),
        };
        if (major, minor) < needed {
            return Err(format!(
                "{} needs compute capability {}.{} or newer, but this GPU has {major}.{minor}",
                self.name(),
                needed.0,
                needed.1
            ));
        }
        Ok(())
    }
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
        Command::Run(args) => cuda::run(&args).map(ExitCode::from),
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
        assert!(args.devices.is_empty(), "every GPU by default");
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
    fn device_lists() {
        assert_eq!(
            run_args(&["burnin", "run", "-d", "0,2"]).devices,
            vec![0, 2]
        );
        assert_eq!(
            run_args(&["burnin", "run", "-d", "1", "-d", "3"]).devices,
            vec![1, 3]
        );
        assert_eq!(
            run_args(&["burnin", "run", "--device", "2"]).devices,
            vec![2]
        );
        assert!(Cli::try_parse_from(["burnin", "run", "-d", "x"]).is_err());
    }

    #[test]
    fn parses_every_precision() {
        for precision in Precision::value_variants() {
            let parsed = run_args(&["burnin", "run", "-p", precision.name()]).precision;
            assert_eq!(parsed, *precision);
        }
        assert_eq!(
            run_args(&["burnin", "run", "--precision", "bf16"]).precision,
            Precision::Bf16
        );
        assert_eq!(
            run_args(&["burnin", "run", "-p", "tf32"]).precision,
            Precision::Tf32
        );
    }

    #[test]
    fn rejects_unknown_precision() {
        assert!(Cli::try_parse_from(["burnin", "run", "-p", "fp128"]).is_err());
        assert!(Cli::try_parse_from(["burnin", "run", "-p", "fp4"]).is_err());
    }

    #[test]
    fn fp32_and_fp64_run_anywhere() {
        for precision in [Precision::Fp32, Precision::Fp64] {
            assert_eq!(precision.check_support((5, 2)), Ok(()));
        }
    }

    #[test]
    fn tensor_core_precisions_need_new_enough_gpus() {
        let cases = [
            (Precision::Fp16, (6, 1), false),
            (Precision::Fp16, (7, 0), true),
            (Precision::Tf32, (7, 5), false),
            (Precision::Tf32, (8, 0), true),
            (Precision::Bf16, (7, 5), false),
            (Precision::Bf16, (8, 6), true),
            (Precision::Fp8, (8, 6), false),
            (Precision::Fp8, (8, 9), true),
            (Precision::Fp8, (9, 0), true),
            (Precision::Fp8, (12, 0), true),
        ];
        for (precision, capability, supported) in cases {
            assert_eq!(
                precision.check_support(capability).is_ok(),
                supported,
                "{} on {capability:?}",
                precision.name()
            );
        }
    }

    #[test]
    fn unsupported_precision_explains_why() {
        assert_eq!(
            Precision::Fp8.check_support((8, 0)),
            Err("fp8 needs compute capability 8.9 or newer, but this GPU has 8.0".to_string())
        );
    }

    #[test]
    fn cli_definition_is_valid() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
