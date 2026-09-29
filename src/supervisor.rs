//! Runs one worker thread per GPU, collects what they report, prints progress
//! and decides a verdict for each GPU.
//!
//! The supervisor knows nothing about CUDA. A worker is any closure that
//! reports through a [`Reporter`] and returns once the shared stop flag is set,
//! so scheduling, reporting and failure handling can be tested without a GPU.

use std::any::Any;
use std::io::Write;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant};

use crate::units::format_duration;

/// How often the supervisor checks the stop flag when nothing else is due.
const POLL: Duration = Duration::from_millis(200);

/// Writes a line, ignoring errors such as a closed pipe: losing output must
/// not stop a test that is still running.
macro_rules! say {
    ($out:expr) => {{ let _ = writeln!($out); }};
    ($out:expr, $($arg:tt)*) => {{ let _ = writeln!($out, $($arg)*); }};
}

/// A GPU to test.
pub struct Target {
    pub ordinal: usize,
    pub name: String,
    /// Extra facts for the header, such as architecture and PCI address.
    pub detail: String,
}

/// The body of a worker. It runs until the stop flag is set, reporting as it goes.
pub type Work = Box<dyn FnOnce(&Reporter) -> anyhow::Result<()> + Send>;

/// What a worker reports once it is set up and about to start testing.
pub struct Ready {
    /// One line describing the setup, such as memory used and chunk size.
    pub detail: String,
    pub flops_per_gemm: f64,
    /// Expected wall time of one chunk of work.
    pub chunk_secs: f64,
}

enum Event {
    Ready(Ready),
    Progress {
        pass: u64,
        gemms: u64,
    },
    Mismatch {
        pass: u64,
        first: usize,
        last: usize,
        count: u64,
    },
    Finished,
    Failed(String),
}

struct Message {
    gpu: usize,
    event: Event,
}

/// Lets a worker send events to the supervisor.
pub struct Reporter {
    gpu: usize,
    tx: Sender<Message>,
}

impl Reporter {
    fn send(&self, event: Event) {
        // The supervisor only stops listening once it has given up on the
        // workers, so a failed send can be ignored.
        let _ = self.tx.send(Message {
            gpu: self.gpu,
            event,
        });
    }

    pub fn ready(&self, ready: Ready) {
        self.send(Event::Ready(ready));
    }

    /// Reports the current pass and the GEMMs completed since the worker became ready.
    pub fn progress(&self, pass: u64, gemms: u64) {
        self.send(Event::Progress { pass, gemms });
    }

    /// Reports mismatched values found in results `first..=last`.
    pub fn mismatch(&self, pass: u64, first: usize, last: usize, count: u64) {
        self.send(Event::Mismatch {
            pass,
            first,
            last,
            count,
        });
    }
}

pub struct Config {
    /// How long to test once every GPU is ready.
    pub duration: Duration,
    pub report_every: Duration,
    /// Longest wait for every GPU to become ready before the clock starts anyway.
    pub max_setup: Duration,
    /// Shortest time without progress before a GPU is reported as stalled.
    pub min_stall: Duration,
    /// Shortest wait for workers to finish their current chunk after the run ends.
    pub min_shutdown_grace: Duration,
}

/// The result for one GPU.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    Pass,
    /// Some results differed from the reference.
    Mismatches {
        values: u64,
        chunks: u64,
    },
    /// Failed with an error after testing began.
    Died(String),
    /// Did not finish within the grace period after the run ended.
    Hung,
    /// Never started testing, so its health is unknown.
    NotTested(String),
}

impl Verdict {
    fn label(&self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Mismatches { .. } | Self::Died(_) => "FAIL",
            Self::Hung => "HUNG",
            Self::NotTested(_) => "ERROR",
        }
    }

    fn is_failure(&self) -> bool {
        matches!(self, Self::Mismatches { .. } | Self::Died(_) | Self::Hung)
    }
}

pub struct Summary {
    /// One verdict per GPU, in the order the workers were given.
    pub verdicts: Vec<Verdict>,
}

impl Summary {
    /// 0 when every GPU passed, 1 when any GPU failed, and 2 when none failed
    /// but at least one could not be tested.
    pub fn exit_status(&self) -> u8 {
        let verdicts = || self.verdicts.iter();
        if verdicts().any(Verdict::is_failure) {
            1
        } else if verdicts().any(|v| matches!(v, Verdict::NotTested(_))) {
            2
        } else {
            0
        }
    }
}

/// Resolves the `--devices` list: every device when empty, otherwise the
/// requested ones in order with duplicates removed.
pub fn select_devices(requested: &[usize], count: usize) -> Result<Vec<usize>, String> {
    if requested.is_empty() {
        return Ok((0..count).collect());
    }
    let mut selected = Vec::new();
    for &ordinal in requested {
        if ordinal >= count {
            return Err(format!(
                "device {ordinal} does not exist; found {count} device(s)"
            ));
        }
        if !selected.contains(&ordinal) {
            selected.push(ordinal);
        }
    }
    Ok(selected)
}

enum Phase {
    Starting,
    Running,
    Finished,
    Failed(String),
}

struct Gpu {
    target: Target,
    phase: Phase,
    ready: Option<Ready>,
    ready_at: Option<Instant>,
    ended_at: Option<Instant>,
    pass: u64,
    gemms: u64,
    mismatches: u64,
    bad_chunks: u64,
    last_progress: Instant,
    /// GEMM count and time of the latest progress at the previous report,
    /// used to measure throughput between reports.
    reported_gemms: u64,
    reported_at: Instant,
    stalled: bool,
}

impl Gpu {
    fn new(target: Target, now: Instant) -> Self {
        Self {
            target,
            phase: Phase::Starting,
            ready: None,
            ready_at: None,
            ended_at: None,
            pass: 0,
            gemms: 0,
            mismatches: 0,
            bad_chunks: 0,
            last_progress: now,
            reported_gemms: 0,
            reported_at: now,
            stalled: false,
        }
    }

    fn is_done(&self) -> bool {
        matches!(self.phase, Phase::Finished | Phase::Failed(_))
    }

    fn flops_per_gemm(&self) -> f64 {
        self.ready
            .as_ref()
            .map_or(0.0, |ready| ready.flops_per_gemm)
    }

    fn verdict(&self) -> Verdict {
        match &self.phase {
            Phase::Failed(error) if self.ready_at.is_some() => Verdict::Died(error.clone()),
            Phase::Failed(error) => Verdict::NotTested(error.clone()),
            Phase::Starting | Phase::Running => Verdict::Hung,
            Phase::Finished if self.mismatches > 0 => Verdict::Mismatches {
                values: self.mismatches,
                chunks: self.bad_chunks,
            },
            Phase::Finished if self.gemms == 0 => {
                Verdict::NotTested("stopped before testing began".to_string())
            }
            Phase::Finished => Verdict::Pass,
        }
    }

    /// Average throughput between becoming ready and finishing.
    fn average_tflops(&self, fallback_end: Instant) -> Option<f64> {
        let secs = (self.ended_at.unwrap_or(fallback_end) - self.ready_at?).as_secs_f64();
        (secs > 0.0).then(|| self.gemms as f64 * self.flops_per_gemm() / secs / 1e12)
    }

    fn apply(&mut self, event: Event, now: Instant, out: &mut impl Write) {
        let ordinal = self.target.ordinal;
        match event {
            Event::Ready(ready) => {
                say!(out, "gpu {ordinal}  ready: {}", ready.detail);
                self.phase = Phase::Running;
                self.ready = Some(ready);
                self.ready_at = Some(now);
                self.last_progress = now;
                self.reported_at = now;
            }
            Event::Progress { pass, gemms } => {
                self.pass = pass;
                self.gemms = gemms;
                self.last_progress = now;
                if self.stalled {
                    self.stalled = false;
                    say!(out, "gpu {ordinal}  progressing again");
                }
            }
            Event::Mismatch {
                pass,
                first,
                last,
                count,
            } => {
                self.mismatches += count;
                self.bad_chunks += 1;
                say!(
                    out,
                    "MISMATCH  gpu {ordinal}  pass {pass}, results {first}-{last}: \
                     {count} values differ from the reference"
                );
            }
            Event::Finished => {
                self.phase = Phase::Finished;
                self.ended_at = Some(now);
            }
            Event::Failed(error) => {
                let when = if self.ready_at.is_some() {
                    "failed during the run"
                } else {
                    "could not start"
                };
                say!(out, "ERROR  gpu {ordinal}  {when}: {error}");
                self.phase = Phase::Failed(error);
                self.ended_at = Some(now);
            }
        }
    }
}

/// Runs every worker to completion and returns each GPU's verdict.
///
/// The test clock starts once every GPU is ready (or after
/// [`Config::max_setup`]). When it runs out, or `stop` is set, every worker is
/// asked to stop and given a grace period to finish its current chunk. Any
/// that do not are reported as hung; their threads are abandoned and end when
/// the process exits.
pub fn supervise(
    workers: Vec<(Target, Work)>,
    config: &Config,
    stop: Arc<AtomicBool>,
    sample: impl Fn(usize) -> Option<String>,
    out: &mut impl Write,
) -> Summary {
    let launched = Instant::now();
    let (tx, rx) = mpsc::channel();
    let mut gpus = Vec::with_capacity(workers.len());
    for (index, (target, work)) in workers.into_iter().enumerate() {
        say!(
            out,
            "gpu {}  {}  ({})",
            target.ordinal,
            target.name,
            target.detail
        );
        let mut gpu = Gpu::new(target, launched);
        let reporter = Reporter {
            gpu: index,
            tx: tx.clone(),
        };
        let spawned = thread::Builder::new()
            .name(format!("gpu-{}", gpu.target.ordinal))
            .spawn(move || run_worker(work, reporter));
        if let Err(err) = spawned {
            gpu.apply(
                Event::Failed(format!("could not start a worker thread: {err}")),
                launched,
                out,
            );
        }
        gpus.push(gpu);
    }
    // Only workers hold senders now, so the channel disconnects once they have all ended.
    drop(tx);

    let clock = run_until_done(&mut gpus, &rx, config, launched, &stop, &sample, out);
    stop.store(true, Ordering::SeqCst);
    let ended = Instant::now();
    wait_for_workers(&mut gpus, &rx, config, ended, out);
    print_summary(
        &gpus,
        clock.map_or(Duration::ZERO, |start| ended - start),
        ended,
        out,
    );

    Summary {
        verdicts: gpus.iter().map(Gpu::verdict).collect(),
    }
}

/// Runs a worker, turning errors and panics into events.
fn run_worker(work: Work, reporter: Reporter) {
    let event = match panic::catch_unwind(AssertUnwindSafe(|| work(&reporter))) {
        Ok(Ok(())) => Event::Finished,
        Ok(Err(err)) => Event::Failed(format!("{err:#}")),
        Err(payload) => Event::Failed(format!("worker panicked: {}", panic_text(&*payload))),
    };
    reporter.send(event);
}

fn panic_text(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Handles events until the run is over, and returns when the test clock started.
fn run_until_done(
    gpus: &mut [Gpu],
    rx: &Receiver<Message>,
    config: &Config,
    launched: Instant,
    stop: &AtomicBool,
    sample: &impl Fn(usize) -> Option<String>,
    out: &mut impl Write,
) -> Option<Instant> {
    let mut clock: Option<(Instant, Instant)> = None;
    let mut next_report = None;

    loop {
        let now = Instant::now();
        let mut wake = now + POLL;
        if let Some(at) = next_report {
            wake = wake.min(at);
        }
        if let Some((_, deadline)) = clock {
            wake = wake.min(deadline);
        }
        match rx.recv_timeout(wake.saturating_duration_since(now)) {
            Ok(message) => gpus[message.gpu].apply(message.event, Instant::now(), out),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }

        let now = Instant::now();
        let all_ready = gpus.iter().all(|gpu| !matches!(gpu.phase, Phase::Starting));
        if clock.is_none() && (all_ready || now >= launched + config.max_setup) {
            clock = Some((now, now + config.duration));
            next_report = Some(now + config.report_every);
            if gpus.iter().any(|gpu| matches!(gpu.phase, Phase::Running)) {
                say!(out, "running for {}", format_duration(config.duration));
            }
        }
        if gpus.iter().all(Gpu::is_done) || stop.load(Ordering::SeqCst) {
            break;
        }
        let Some((start, deadline)) = clock else {
            continue;
        };
        if now >= deadline {
            break;
        }
        if next_report.is_some_and(|at| now >= at) {
            report(gpus, start, now, sample, out);
            next_report = Some(now + config.report_every);
        }
        warn_about_stalls(gpus, config, now, out);
    }
    clock.map(|(start, _)| start)
}

fn report(
    gpus: &mut [Gpu],
    start: Instant,
    now: Instant,
    sample: &impl Fn(usize) -> Option<String>,
    out: &mut impl Write,
) {
    for (index, gpu) in gpus.iter_mut().enumerate() {
        if !matches!(gpu.phase, Phase::Running) {
            continue;
        }
        let secs = (gpu.last_progress - gpu.reported_at).as_secs_f64();
        let rate = if secs > 0.0 {
            let tflops =
                (gpu.gemms - gpu.reported_gemms) as f64 * gpu.flops_per_gemm() / secs / 1e12;
            format!("{tflops:>8.2}")
        } else {
            format!("{:>8}", "--")
        };
        let line = format!(
            "{:>8}  gpu {:<2} pass {:<4} {rate} TFLOP/s  mismatches {:<6} {}",
            format_duration(now - start),
            gpu.target.ordinal,
            gpu.pass,
            gpu.mismatches,
            sample(index).unwrap_or_default()
        );
        say!(out, "{}", line.trim_end());
        gpu.reported_gemms = gpu.gemms;
        gpu.reported_at = gpu.last_progress;
    }
}

fn warn_about_stalls(gpus: &mut [Gpu], config: &Config, now: Instant, out: &mut impl Write) {
    for gpu in gpus.iter_mut() {
        let Some(ready) = &gpu.ready else { continue };
        if gpu.stalled || !matches!(gpu.phase, Phase::Running) {
            continue;
        }
        let threshold = config
            .min_stall
            .max(Duration::from_secs_f64(ready.chunk_secs * 10.0));
        let quiet = now - gpu.last_progress;
        if quiet > threshold {
            gpu.stalled = true;
            say!(
                out,
                "WARNING  gpu {}  no progress for {}",
                gpu.target.ordinal,
                format_duration(quiet)
            );
        }
    }
}

/// Gives workers time to finish their current chunk after being told to stop.
fn wait_for_workers(
    gpus: &mut [Gpu],
    rx: &Receiver<Message>,
    config: &Config,
    ended: Instant,
    out: &mut impl Write,
) {
    let slowest_chunk = gpus
        .iter()
        .filter_map(|gpu| gpu.ready.as_ref().map(|ready| ready.chunk_secs))
        .fold(0.0, f64::max);
    let grace = config
        .min_shutdown_grace
        .max(Duration::from_secs_f64(slowest_chunk * 5.0));
    let give_up = ended + grace;
    while !gpus.iter().all(Gpu::is_done) {
        let now = Instant::now();
        if now >= give_up {
            break;
        }
        match rx.recv_timeout(give_up - now) {
            Ok(message) => gpus[message.gpu].apply(message.event, Instant::now(), out),
            Err(_) => break,
        }
    }
}

fn print_summary(gpus: &[Gpu], elapsed: Duration, ended: Instant, out: &mut impl Write) {
    let name_width = gpus
        .iter()
        .map(|gpu| gpu.target.name.len())
        .max()
        .unwrap_or(0);
    say!(out);
    say!(out, "summary after {}", format_duration(elapsed));
    for gpu in gpus {
        let verdict = gpu.verdict();
        let detail = match &verdict {
            Verdict::Pass => match gpu.average_tflops(ended) {
                Some(tflops) => format!("{} GEMMs, {tflops:.2} TFLOP/s average", gpu.gemms),
                None => format!("{} GEMMs", gpu.gemms),
            },
            Verdict::Mismatches { values, chunks } => {
                format!("{values} mismatched values in {chunks} chunk(s)")
            }
            Verdict::Died(error) => format!("failed during the run: {error}"),
            Verdict::Hung => "did not stop within the grace period".to_string(),
            Verdict::NotTested(error) => format!("not tested: {error}"),
        };
        say!(
            out,
            "  gpu {:<2} {:<name_width$}  {:<5}  {detail}",
            gpu.target.ordinal,
            gpu.target.name,
            verdict.label()
        );
    }

    let total = gpus.len();
    let failed = gpus.iter().filter(|gpu| gpu.verdict().is_failure()).count();
    let untested = gpus
        .iter()
        .filter(|gpu| matches!(gpu.verdict(), Verdict::NotTested(_)))
        .count();
    if failed > 0 {
        say!(out, "FAIL  {failed} of {total} GPU(s) failed");
    } else if untested > 0 {
        say!(
            out,
            "ERROR  {untested} of {total} GPU(s) could not be tested"
        );
    } else {
        say!(out, "PASS  all {total} GPU(s) passed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config {
            duration: Duration::from_millis(150),
            report_every: Duration::from_millis(40),
            max_setup: Duration::from_secs(5),
            min_stall: Duration::from_millis(60),
            min_shutdown_grace: Duration::from_millis(300),
        }
    }

    fn target(ordinal: usize) -> Target {
        Target {
            ordinal,
            name: format!("Mock GPU {ordinal}"),
            detail: "mock".to_string(),
        }
    }

    fn ready(reporter: &Reporter) {
        reporter.ready(Ready {
            detail: "mock setup".to_string(),
            flops_per_gemm: 1e12,
            chunk_secs: 0.005,
        });
    }

    /// A worker that keeps completing chunks until told to stop, optionally
    /// reporting a mismatch in its second chunk.
    fn healthy(stop: &Arc<AtomicBool>, mismatch: bool) -> Work {
        let stop = stop.clone();
        Box::new(move |reporter| {
            ready(reporter);
            let mut gemms = 0;
            while !stop.load(Ordering::SeqCst) {
                thread::sleep(Duration::from_millis(5));
                gemms += 4;
                if mismatch && gemms == 8 {
                    reporter.mismatch(1, 5, 8, 3);
                }
                reporter.progress(1, gemms);
            }
            Ok(())
        })
    }

    fn run(workers: Vec<(Target, Work)>, stop: Arc<AtomicBool>, cfg: &Config) -> (Summary, String) {
        let mut out = Vec::new();
        let summary = supervise(workers, cfg, stop, |_| None, &mut out);
        (summary, String::from_utf8(out).unwrap())
    }

    fn verdicts(summary: &Summary) -> Vec<Verdict> {
        summary.verdicts.clone()
    }

    #[test]
    fn all_healthy_gpus_pass() {
        let stop = Arc::new(AtomicBool::new(false));
        let workers = (0..3).map(|i| (target(i), healthy(&stop, false))).collect();
        let (summary, out) = run(workers, stop, &config());
        // A pass also means the GPU completed GEMMs; otherwise it is not tested.
        assert_eq!(verdicts(&summary), vec![Verdict::Pass; 3]);
        assert_eq!(summary.exit_status(), 0);
        assert!(out.contains("PASS  all 3 GPU(s) passed"), "{out}");
        assert!(out.contains("TFLOP/s"), "{out}");
    }

    #[test]
    fn a_mismatch_fails_only_that_gpu() {
        let stop = Arc::new(AtomicBool::new(false));
        let workers = vec![
            (target(0), healthy(&stop, false)),
            (target(1), healthy(&stop, true)),
        ];
        let (summary, out) = run(workers, stop, &config());
        assert_eq!(
            verdicts(&summary),
            vec![
                Verdict::Pass,
                Verdict::Mismatches {
                    values: 3,
                    chunks: 1
                }
            ]
        );
        assert_eq!(summary.exit_status(), 1);
        assert!(
            out.contains("MISMATCH  gpu 1  pass 1, results 5-8"),
            "{out}"
        );
    }

    #[test]
    fn setup_failure_means_not_tested() {
        let stop = Arc::new(AtomicBool::new(false));
        let workers: Vec<(Target, Work)> = vec![
            (target(0), healthy(&stop, false)),
            (target(1), Box::new(|_| anyhow::bail!("out of memory"))),
        ];
        let (summary, out) = run(workers, stop, &config());
        assert_eq!(
            verdicts(&summary),
            vec![Verdict::Pass, Verdict::NotTested("out of memory".into())]
        );
        assert_eq!(summary.exit_status(), 2);
        assert!(
            out.contains("ERROR  gpu 1  could not start: out of memory"),
            "{out}"
        );
    }

    #[test]
    fn error_during_the_run_fails_the_gpu() {
        let stop = Arc::new(AtomicBool::new(false));
        let workers: Vec<(Target, Work)> = vec![
            (target(0), healthy(&stop, false)),
            (
                target(1),
                Box::new(|reporter| {
                    ready(reporter);
                    reporter.progress(1, 4);
                    anyhow::bail!("illegal address")
                }),
            ),
        ];
        let (summary, _) = run(workers, stop, &config());
        assert_eq!(
            verdicts(&summary)[1],
            Verdict::Died("illegal address".into())
        );
        assert_eq!(summary.exit_status(), 1);
    }

    #[test]
    fn a_panicking_worker_is_reported() {
        let stop = Arc::new(AtomicBool::new(false));
        let workers: Vec<(Target, Work)> = vec![(
            target(0),
            Box::new(|reporter| {
                ready(reporter);
                panic!("boom")
            }),
        )];
        let (summary, _) = run(workers, stop, &config());
        assert_eq!(
            verdicts(&summary),
            vec![Verdict::Died("worker panicked: boom".into())]
        );
    }

    #[test]
    fn a_gpu_that_ignores_stop_is_hung() {
        let stop = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let hung_release = release.clone();
        let workers: Vec<(Target, Work)> = vec![
            (target(0), healthy(&stop, false)),
            (
                target(1),
                Box::new(move |reporter| {
                    ready(reporter);
                    while !hung_release.load(Ordering::SeqCst) {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Ok(())
                }),
            ),
        ];
        let (summary, out) = run(workers, stop, &config());
        release.store(true, Ordering::SeqCst);
        assert_eq!(verdicts(&summary), vec![Verdict::Pass, Verdict::Hung]);
        assert_eq!(summary.exit_status(), 1);
        assert!(out.contains("WARNING  gpu 1  no progress"), "{out}");
    }

    #[test]
    fn stops_early_when_every_gpu_has_failed() {
        let stop = Arc::new(AtomicBool::new(false));
        let workers: Vec<(Target, Work)> = (0..2)
            .map(|i| {
                (
                    target(i),
                    Box::new(|_: &Reporter| anyhow::bail!("no device")) as Work,
                )
            })
            .collect();
        let cfg = Config {
            duration: Duration::from_secs(60),
            ..config()
        };
        let started = Instant::now();
        let (summary, _) = run(workers, stop, &cfg);
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(summary.exit_status(), 2);
    }

    #[test]
    fn stop_flag_ends_the_run_early() {
        let stop = Arc::new(AtomicBool::new(false));
        let workers = vec![(target(0), healthy(&stop, false))];
        let cfg = Config {
            duration: Duration::from_secs(60),
            ..config()
        };
        let trigger = stop.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            trigger.store(true, Ordering::SeqCst);
        });
        let started = Instant::now();
        let (summary, _) = run(workers, stop, &cfg);
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(verdicts(&summary), vec![Verdict::Pass]);
    }

    #[test]
    fn selects_devices() {
        assert_eq!(select_devices(&[], 3), Ok(vec![0, 1, 2]));
        assert_eq!(select_devices(&[2, 0, 2], 3), Ok(vec![2, 0]));
        assert!(select_devices(&[3], 3).is_err());
        assert_eq!(select_devices(&[], 0), Ok(vec![]));
    }
}
