//! burnin: GPU burn-in and stress testing.

// The GPU backends are platform-specific; on other platforms much of the shared
// code is unused.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

mod isolation;
mod mem;
mod monitor;
mod supervisor;
mod units;

#[cfg(target_os = "linux")]
mod cuda;
#[cfg(target_os = "linux")]
mod telemetry;

use std::ffi::OsString;
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
    /// Test one GPU on behalf of a parent `burnin run`, reporting on stdout.
    #[command(hide = true)]
    Worker(WorkerArgs),
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

    /// Where each GPU is tested: in its own process, so a hung GPU can be
    /// stopped, or in a thread of this process.
    #[arg(long, value_enum, default_value_t = Isolation::Process)]
    pub isolation: Isolation,

    /// Give up on a GPU that makes no progress for this long, and stop its process.
    #[arg(long, default_value = "3m", value_parser = units::parse_duration)]
    pub hang_timeout: Duration,
}

impl RunArgs {
    /// Arguments that start a child `burnin worker` testing one GPU with these settings.
    pub fn worker_args(&self, ordinal: usize, host_sharers: u64) -> Vec<OsString> {
        let precision = self
            .precision
            .to_possible_value()
            .expect("every precision has a name");
        let mut args: Vec<OsString> = [
            "worker".to_string(),
            "--ordinal".to_string(),
            ordinal.to_string(),
            "--host-sharers".to_string(),
            host_sharers.to_string(),
            "--precision".to_string(),
            precision.get_name().to_string(),
            "--mem".to_string(),
            self.mem.to_string(),
            "--matrix-size".to_string(),
            self.matrix_size.to_string(),
            "--chunk-secs".to_string(),
            self.chunk_secs.to_string(),
            "--tolerance".to_string(),
            self.tolerance.to_string(),
        ]
        .into_iter()
        .map(OsString::from)
        .collect();
        if self.inject_fault {
            args.push("--inject-fault".into());
        }
        args
    }
}

#[derive(Args, Clone, Debug)]
pub struct WorkerArgs {
    /// CUDA device to test.
    #[arg(long)]
    pub ordinal: usize,

    /// Number of unified-memory GPUs sharing host memory in the run.
    #[arg(long, default_value_t = 1)]
    pub host_sharers: u64,

    #[command(flatten)]
    pub run: RunArgs,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isolation {
    /// Each GPU in its own process.
    Process,
    /// Each GPU in a thread of this process.
    Thread,
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
        Command::Worker(args) => Ok(cuda::serve_worker(&args)),
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
        assert_eq!(args.isolation, Isolation::Process);
        assert_eq!(args.hang_timeout, Duration::from_secs(180));
    }

    #[test]
    fn worker_args_carry_the_run_settings() {
        let run = run_args(&[
            "burnin",
            "run",
            "-p",
            "fp64",
            "-m",
            "12.5%",
            "--matrix-size",
            "4096",
            "--chunk-secs",
            "0.75",
            "--tolerance",
            "0.001",
            "--inject-fault",
        ]);
        let mut argv = vec![OsString::from("burnin")];
        argv.extend(run.worker_args(3, 2));
        let Command::Worker(worker) = Cli::try_parse_from(argv).unwrap().command else {
            panic!("not a worker command");
        };
        assert_eq!(worker.ordinal, 3);
        assert_eq!(worker.host_sharers, 2);
        assert_eq!(worker.run.precision, run.precision);
        assert_eq!(worker.run.mem, run.mem);
        assert_eq!(worker.run.matrix_size, run.matrix_size);
        assert_eq!(worker.run.chunk_secs, run.chunk_secs);
        assert_eq!(worker.run.tolerance, run.tolerance);
        assert!(worker.run.inject_fault);
    }

    #[test]
    fn worker_args_keep_exact_memory_sizes() {
        let run = run_args(&["burnin", "run", "-m", "16G"]);
        let mut argv = vec![OsString::from("burnin")];
        argv.extend(run.worker_args(0, 1));
        let Command::Worker(worker) = Cli::try_parse_from(argv).unwrap().command else {
            panic!("not a worker command");
        };
        assert_eq!(worker.run.mem, MemSpec::Bytes(16 << 30));
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
