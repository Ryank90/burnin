//! Runs one worker thread per GPU, collects what they report, prints progress
//! and decides a verdict for each GPU.
//!
//! The supervisor knows nothing about CUDA or Metal. A worker is any closure
//! that reports through a [`Reporter`] and returns once the shared stop flag is
//! set, so scheduling, reporting and failure handling can be tested without a
//! GPU. A worker thread may itself relay events from a child process; see
//! [`crate::isolation`].

use std::any::Any;
use std::io::Write;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::monitor::{HardwareErrors, Monitor, Stats};
use crate::output::{GpuSummary, Output, Record};
use crate::units::format_duration;

/// How often the supervisor checks the stop flag when nothing else is due.
const POLL: Duration = Duration::from_millis(200);

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
#[derive(Serialize, Deserialize)]
pub struct Ready {
    /// One line describing the setup, such as memory used and chunk size.
    pub detail: String,
    pub flops_per_gemm: f64,
    /// Expected wall time of one chunk of work.
    pub chunk_secs: f64,
}

/// Something a worker reports. Serializable so that workers in child
/// processes can send events over a pipe.
#[derive(Serialize, Deserialize)]
pub(crate) enum Event {
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

pub(crate) struct Message {
    gpu: usize,
    pub(crate) event: Event,
}

/// Lets a worker send events to the supervisor.
#[derive(Clone)]
pub struct Reporter {
    gpu: usize,
    tx: Sender<Message>,
    abandoned: Arc<AtomicBool>,
}

impl Reporter {
    /// A reporter that isn't attached to a supervisor, for a worker running in
    /// a child process. Its events arrive on the returned receiver.
    pub(crate) fn detached() -> (Self, Receiver<Message>) {
        let (tx, rx) = mpsc::channel();
        let reporter = Self {
            gpu: 0,
            tx,
            abandoned: Arc::new(AtomicBool::new(false)),
        };
        (reporter, rx)
    }

    /// True once the supervisor has given up on this worker. A worker that can
    /// be interrupted, such as a child process, should then be stopped.
    pub fn abandoned(&self) -> bool {
        self.abandoned.load(Ordering::SeqCst)
    }

    #[cfg(test)]
    pub(crate) fn abandon(&self) {
        self.abandoned.store(true, Ordering::SeqCst);
    }

    pub(crate) fn send(&self, event: Event) {
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
    /// Shortest time without progress before the supervisor gives up on a GPU
    /// and marks it hung.
    pub min_hang: Duration,
    /// Shortest wait for workers to finish their current chunk after the run ends.
    pub min_shutdown_grace: Duration,
    /// Longest wait for workers that were given up on to exit before the
    /// summary is printed.
    pub reap_grace: Duration,
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
    /// Every result matched, but the GPU reported uncorrected memory errors
    /// or critical driver errors.
    HardwareErrors(String),
    /// Failed with an error after testing began.
    Died(String),
    /// Stopped making progress, or did not stop when the run ended.
    Hung(String),
    /// Never started testing, so its health is unknown.
    NotTested(String),
}

impl Verdict {
    /// `pass`, `fail`, `hung` or `error`.
    fn name(&self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Mismatches { .. } | Self::HardwareErrors(_) | Self::Died(_) => "fail",
            Self::Hung(_) => "hung",
            Self::NotTested(_) => "error",
        }
    }

    fn is_failure(&self) -> bool {
        matches!(
            self,
            Self::Mismatches { .. } | Self::HardwareErrors(_) | Self::Died(_) | Self::Hung(_)
        )
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
    /// Given up on, with the reason.
    Hung(String),
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
    /// Shared with the worker's reporter; set when the supervisor gives up on it.
    abandoned: Arc<AtomicBool>,
    /// The worker has returned, even if it was given up on first.
    exited: bool,
    stats: Stats,
    /// Hardware errors as of the last check.
    errors: HardwareErrors,
}

impl Gpu {
    fn new(target: Target, now: Instant, abandoned: Arc<AtomicBool>) -> Self {
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
            abandoned,
            exited: false,
            stats: Stats::default(),
            errors: HardwareErrors::default(),
        }
    }

    fn is_done(&self) -> bool {
        matches!(
            self.phase,
            Phase::Finished | Phase::Failed(_) | Phase::Hung(_)
        )
    }

    /// Stops waiting for this GPU and tells its worker to stop if it can.
    fn give_up(&mut self, reason: String, now: Instant, out: &mut Output<impl Write>) {
        out.emit(Record::Hung {
            gpu: self.target.ordinal,
            reason: &reason,
        });
        self.abandoned.store(true, Ordering::SeqCst);
        self.phase = Phase::Hung(reason);
        self.ended_at = Some(now);
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
            Phase::Hung(reason) => Verdict::Hung(reason.clone()),
            Phase::Starting | Phase::Running => Verdict::Hung("did not stop".to_string()),
            Phase::Finished if self.mismatches > 0 => Verdict::Mismatches {
                values: self.mismatches,
                chunks: self.bad_chunks,
            },
            Phase::Finished if self.errors.is_failure() => {
                Verdict::HardwareErrors(self.errors.failure())
            }
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

    fn summary(&self, ended: Instant) -> GpuSummary {
        let verdict = self.verdict();
        let tflops_average = self.average_tflops(ended);
        let detail = match &verdict {
            Verdict::Pass => match tflops_average {
                Some(tflops) => format!("{} GEMMs, {tflops:.2} TFLOP/s average", self.gemms),
                None => format!("{} GEMMs", self.gemms),
            },
            Verdict::Mismatches { values, chunks } => {
                format!("{values} mismatched values in {chunks} chunk(s)")
            }
            Verdict::HardwareErrors(errors) => format!("every result matched, but {errors}"),
            Verdict::Died(error) => format!("failed during the run: {error}"),
            Verdict::Hung(reason) => reason.clone(),
            Verdict::NotTested(error) => format!("not tested: {error}"),
        };
        GpuSummary {
            gpu: self.target.ordinal,
            name: self.target.name.clone(),
            verdict: verdict.name(),
            detail,
            gemms: self.gemms,
            mismatches: self.mismatches,
            tflops_average,
            telemetry: self.stats.summary(),
            hardware_errors: self.errors.clone(),
        }
    }

    fn apply(&mut self, event: Event, now: Instant, out: &mut Output<impl Write>) {
        let ordinal = self.target.ordinal;
        if matches!(self.phase, Phase::Hung(_)) {
            // Given up on: only note when its worker finally returns.
            self.exited |= matches!(event, Event::Finished | Event::Failed(_));
            return;
        }
        match event {
            Event::Ready(ready) => {
                out.emit(Record::Ready {
                    gpu: ordinal,
                    detail: &ready.detail,
                });
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
                    out.emit(Record::Recovered { gpu: ordinal });
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
                out.emit(Record::Mismatch {
                    gpu: ordinal,
                    pass,
                    first,
                    last,
                    count,
                });
            }
            Event::Finished => {
                self.phase = Phase::Finished;
                self.ended_at = Some(now);
                self.exited = true;
            }
            Event::Failed(error) => {
                out.emit(Record::Error {
                    gpu: ordinal,
                    during_run: self.ready_at.is_some(),
                    message: &error,
                });
                self.phase = Phase::Failed(error);
                self.ended_at = Some(now);
                self.exited = true;
            }
        }
    }
}

/// Runs every worker to completion and returns each GPU's verdict.
///
/// The test clock starts once every GPU is ready (or after
/// [`Config::max_setup`]). A GPU that stops making progress for too long is
/// given up on and marked hung while the others carry on. When the clock runs
/// out, or `stop` is set, every worker is asked to stop and given a grace
/// period to finish its current chunk; any that do not are also marked hung.
/// Workers that were given up on are told so through their reporter. A worker
/// thread that can't be interrupted is left behind and ends when the process
/// exits.
pub fn supervise(
    workers: Vec<(Target, Work)>,
    config: &Config,
    stop: Arc<AtomicBool>,
    monitor: &dyn Monitor,
    out: &mut Output<impl Write>,
) -> Summary {
    let launched = Instant::now();
    let (tx, rx) = mpsc::channel();
    let mut gpus = Vec::with_capacity(workers.len());
    for (index, (target, work)) in workers.into_iter().enumerate() {
        out.emit(Record::Gpu {
            gpu: target.ordinal,
            name: &target.name,
            detail: &target.detail,
        });
        let abandoned = Arc::new(AtomicBool::new(false));
        let mut gpu = Gpu::new(target, launched, abandoned.clone());
        let reporter = Reporter {
            gpu: index,
            tx: tx.clone(),
            abandoned,
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

    let clock = run_until_done(&mut gpus, &rx, config, launched, &stop, monitor, out);
    stop.store(true, Ordering::SeqCst);
    let ended = Instant::now();
    wait_for_workers(&mut gpus, &rx, config, ended, out);
    reap(&mut gpus, &rx, config.reap_grace, out);
    check_hardware(&mut gpus, monitor, out);

    let summary = Summary {
        verdicts: gpus.iter().map(Gpu::verdict).collect(),
    };
    let exit_status = summary.exit_status();
    let gpu_summaries: Vec<GpuSummary> = gpus.iter().map(|gpu| gpu.summary(ended)).collect();
    out.emit(Record::Summary {
        elapsed_secs: clock
            .map_or(Duration::ZERO, |start| ended - start)
            .as_secs_f64(),
        result: match exit_status {
            0 => "pass",
            1 => "fail",
            _ => "error",
        },
        exit_status,
        gpus: &gpu_summaries,
    });
    summary
}

/// The first Ctrl-C (or SIGTERM) lets every GPU finish its current chunk and
/// prints the summary; a second one exits immediately.
pub fn install_stop_handler(stop: Arc<AtomicBool>) -> anyhow::Result<()> {
    let presses = AtomicUsize::new(0);
    ctrlc::set_handler(move || {
        if presses.fetch_add(1, Ordering::SeqCst) == 0 {
            eprintln!("\nstopping after the current chunk; press Ctrl-C again to quit now");
            stop.store(true, Ordering::SeqCst);
        } else {
            std::process::exit(130);
        }
    })
    .context("could not install the Ctrl-C handler")
}

/// Runs a worker, turning errors and panics into events.
pub(crate) fn run_worker(work: Work, reporter: Reporter) {
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
    monitor: &dyn Monitor,
    out: &mut Output<impl Write>,
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
                out.emit(Record::Running {
                    duration_secs: config.duration.as_secs_f64(),
                });
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
            report(gpus, start, now, monitor, out);
            check_hardware(gpus, monitor, out);
            next_report = Some(now + config.report_every);
        }
        warn_about_stalls(gpus, config, now, out);
        give_up_on_hung(gpus, config, now, out);
    }
    clock.map(|(start, _)| start)
}

fn report(
    gpus: &mut [Gpu],
    start: Instant,
    now: Instant,
    monitor: &dyn Monitor,
    out: &mut Output<impl Write>,
) {
    for (index, gpu) in gpus.iter_mut().enumerate() {
        if !matches!(gpu.phase, Phase::Running) {
            continue;
        }
        let secs = (gpu.last_progress - gpu.reported_at).as_secs_f64();
        let tflops = (secs > 0.0)
            .then(|| (gpu.gemms - gpu.reported_gemms) as f64 * gpu.flops_per_gemm() / secs / 1e12);
        let reading = monitor.reading(index);
        if let Some(reading) = &reading {
            gpu.stats.add(reading);
        }
        out.emit(Record::Progress {
            elapsed_secs: (now - start).as_secs_f64(),
            gpu: gpu.target.ordinal,
            pass: gpu.pass,
            tflops,
            mismatches: gpu.mismatches,
            reading: reading.as_ref(),
        });
        gpu.reported_gemms = gpu.gemms;
        gpu.reported_at = gpu.last_progress;
    }
}

/// Reports hardware errors as they appear.
fn check_hardware(gpus: &mut [Gpu], monitor: &dyn Monitor, out: &mut Output<impl Write>) {
    for (index, gpu) in gpus.iter_mut().enumerate() {
        let errors = monitor.errors(index);
        for change in errors.changes_since(&gpu.errors) {
            out.emit(Record::Hardware {
                gpu: gpu.target.ordinal,
                message: &change,
            });
        }
        gpu.errors = errors;
    }
}

fn warn_about_stalls(
    gpus: &mut [Gpu],
    config: &Config,
    now: Instant,
    out: &mut Output<impl Write>,
) {
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
            out.emit(Record::Stalled {
                gpu: gpu.target.ordinal,
                quiet_secs: quiet.as_secs_f64(),
            });
        }
    }
}

/// Gives up on running GPUs that have made no progress for too long.
fn give_up_on_hung(gpus: &mut [Gpu], config: &Config, now: Instant, out: &mut Output<impl Write>) {
    for gpu in gpus.iter_mut() {
        let chunk_secs = match (&gpu.phase, &gpu.ready) {
            (Phase::Running, Some(ready)) => ready.chunk_secs,
            _ => continue,
        };
        let limit = config
            .min_hang
            .max(Duration::from_secs_f64(chunk_secs * 30.0));
        let quiet = now - gpu.last_progress;
        if quiet > limit {
            gpu.give_up(
                format!("no progress for {}", format_duration(quiet)),
                now,
                out,
            );
        }
    }
}

/// Gives workers time to finish their current chunk after being told to stop,
/// and gives up on any that don't.
fn wait_for_workers(
    gpus: &mut [Gpu],
    rx: &Receiver<Message>,
    config: &Config,
    ended: Instant,
    out: &mut Output<impl Write>,
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
    let now = Instant::now();
    for gpu in gpus.iter_mut().filter(|gpu| !gpu.is_done()) {
        let reason = format!(
            "did not stop within {} of the end of the run",
            format_duration(grace)
        );
        gpu.give_up(reason, now, out);
    }
}

/// Waits briefly for workers that were given up on to exit, so that child
/// processes are killed and collected before burnin exits.
fn reap(gpus: &mut [Gpu], rx: &Receiver<Message>, grace: Duration, out: &mut Output<impl Write>) {
    let deadline = Instant::now() + grace;
    while gpus.iter().any(|gpu| !gpu.exited) {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        match rx.recv_timeout(deadline - now) {
            Ok(message) => gpus[message.gpu].apply(message.event, now, out),
            Err(_) => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::{NoMonitor, Reading};
    use crate::output::Format;

    fn config() -> Config {
        Config {
            duration: Duration::from_millis(150),
            report_every: Duration::from_millis(40),
            max_setup: Duration::from_secs(5),
            min_stall: Duration::from_millis(60),
            min_hang: Duration::from_secs(60),
            min_shutdown_grace: Duration::from_millis(300),
            reap_grace: Duration::from_millis(100),
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
        run_with(workers, stop, cfg, &NoMonitor)
    }

    fn run_with(
        workers: Vec<(Target, Work)>,
        stop: Arc<AtomicBool>,
        cfg: &Config,
        monitor: &dyn Monitor,
    ) -> (Summary, String) {
        let mut out = Output::new(Format::Human, Vec::new());
        let summary = supervise(workers, cfg, stop, monitor, &mut out);
        (summary, String::from_utf8(out.into_inner()).unwrap())
    }

    /// Reports the same readings and errors for every GPU.
    struct FakeMonitor {
        reading: Reading,
        errors: HardwareErrors,
    }

    impl Monitor for FakeMonitor {
        fn reading(&self, _gpu: usize) -> Option<Reading> {
            Some(self.reading.clone())
        }

        fn errors(&self, _gpu: usize) -> HardwareErrors {
            self.errors.clone()
        }
    }

    fn healthy_readings(errors: HardwareErrors) -> FakeMonitor {
        FakeMonitor {
            reading: Reading {
                temperature_c: Some(70),
                power_w: Some(300.0),
                sm_clock_mhz: Some(1800),
                throttle: vec!["sw-power-cap".to_string()],
            },
            errors,
        }
    }

    #[test]
    fn telemetry_appears_in_progress_and_summary() {
        let stop = Arc::new(AtomicBool::new(false));
        let workers = vec![(target(0), healthy(&stop, false))];
        let monitor = healthy_readings(HardwareErrors {
            ecc_corrected: Some(0),
            ecc_uncorrected: Some(0),
            xids: Some(vec![]),
        });
        let (summary, out) = run_with(workers, stop, &config(), &monitor);
        assert_eq!(verdicts(&summary), vec![Verdict::Pass]);
        assert!(
            out.contains(" 70 C    300 W  1800 MHz  throttled: sw-power-cap"),
            "{out}"
        );
        assert!(
            out.contains("70 C peak; 300 W average, 300 W peak"),
            "{out}"
        );
        assert!(
            out.contains("ECC 0 corrected, 0 uncorrected; no driver errors"),
            "{out}"
        );
    }

    #[test]
    fn a_driver_error_fails_a_gpu_whose_results_matched() {
        let stop = Arc::new(AtomicBool::new(false));
        let workers = vec![(target(0), healthy(&stop, false))];
        let monitor = healthy_readings(HardwareErrors {
            ecc_corrected: Some(0),
            ecc_uncorrected: Some(0),
            xids: Some(vec![79]),
        });
        let (summary, out) = run_with(workers, stop, &config(), &monitor);
        assert_eq!(
            verdicts(&summary),
            vec![Verdict::HardwareErrors("Xid 79 from the driver".into())]
        );
        assert_eq!(summary.exit_status(), 1);
        assert!(
            out.contains("HARDWARE  gpu 0  the driver reported critical error Xid 79"),
            "{out}"
        );
    }

    #[test]
    fn corrected_memory_errors_are_reported_but_pass() {
        let stop = Arc::new(AtomicBool::new(false));
        let workers = vec![(target(0), healthy(&stop, false))];
        let monitor = healthy_readings(HardwareErrors {
            ecc_corrected: Some(5),
            ecc_uncorrected: Some(0),
            xids: Some(vec![]),
        });
        let (summary, out) = run_with(workers, stop, &config(), &monitor);
        assert_eq!(verdicts(&summary), vec![Verdict::Pass]);
        assert!(
            out.contains("HARDWARE  gpu 0  5 new corrected ECC errors"),
            "{out}"
        );
        assert!(out.contains("ECC 5 corrected, 0 uncorrected"), "{out}");
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
        match &verdicts(&summary)[..] {
            [Verdict::Pass, Verdict::Hung(reason)] => {
                assert!(reason.starts_with("did not stop within"), "{reason}")
            }
            other => panic!("unexpected verdicts {other:?}"),
        }
        assert_eq!(summary.exit_status(), 1);
        assert!(out.contains("WARNING  gpu 1  no progress"), "{out}");
    }

    #[test]
    fn watchdog_gives_up_on_a_stuck_gpu_and_the_rest_carry_on() {
        let stop = Arc::new(AtomicBool::new(false));
        let workers: Vec<(Target, Work)> = vec![
            (target(0), healthy(&stop, false)),
            (
                target(1),
                Box::new(|reporter| {
                    ready(reporter);
                    reporter.progress(1, 4);
                    // Stuck until the supervisor gives up, as a killed child process would be.
                    while !reporter.abandoned() {
                        thread::sleep(Duration::from_millis(5));
                    }
                    anyhow::bail!("killed")
                }),
            ),
        ];
        let cfg = Config {
            duration: Duration::from_millis(500),
            min_hang: Duration::from_millis(100),
            ..config()
        };
        let (summary, out) = run(workers, stop, &cfg);
        match &verdicts(&summary)[..] {
            [Verdict::Pass, Verdict::Hung(reason)] => {
                assert!(reason.starts_with("no progress for"), "{reason}")
            }
            other => panic!("unexpected verdicts {other:?}"),
        }
        let hung_at = out.find("HUNG  gpu 1").expect("hang reported");
        assert!(
            out[hung_at..].contains("gpu 0  pass"),
            "gpu 0 kept reporting after gpu 1 hung: {out}"
        );
        assert!(!out.contains("ERROR  gpu 1"), "{out}");
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
    fn json_output_ends_with_a_summary() {
        let stop = Arc::new(AtomicBool::new(false));
        let workers = vec![
            (target(0), healthy(&stop, false)),
            (target(1), healthy(&stop, true)),
        ];
        let mut out = Output::new(Format::Json, Vec::new());
        let summary = supervise(workers, &config(), stop, &NoMonitor, &mut out);
        let text = String::from_utf8(out.into_inner()).unwrap();
        let records: Vec<serde_json::Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).expect("every line is JSON"))
            .collect();

        let events: Vec<&str> = records
            .iter()
            .map(|r| r["event"].as_str().unwrap())
            .collect();
        assert_eq!(&events[..2], ["gpu", "gpu"]);
        assert!(events.contains(&"ready") && events.contains(&"progress"));
        assert!(events.contains(&"mismatch"));

        let last = records.last().unwrap();
        assert_eq!(last["event"], "summary");
        assert_eq!(last["result"], "fail");
        assert_eq!(last["exit_status"], summary.exit_status());
        assert_eq!(last["gpus"][0]["verdict"], "pass");
        assert_eq!(last["gpus"][1]["verdict"], "fail");
        assert_eq!(last["gpus"][1]["mismatches"], 3);
    }

    #[test]
    fn selects_devices() {
        assert_eq!(select_devices(&[], 3), Ok(vec![0, 1, 2]));
        assert_eq!(select_devices(&[2, 0, 2], 3), Ok(vec![2, 0]));
        assert!(select_devices(&[3], 3).is_err());
        assert_eq!(select_devices(&[], 0), Ok(vec![]));
    }
}
