//! Runs each GPU's worker in a child process, so a GPU that hangs can be
//! killed without affecting the others.
//!
//! For each GPU the parent starts `burnin worker ...`. The child writes its
//! events to stdout as JSON lines and stops when its stdin closes. The parent
//! closes stdin to end the run, and stdin also closes if the parent exits, so a
//! child never outlives it by more than a chunk. A child the supervisor has
//! given up on is killed.

use std::ffi::OsString;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::process::{Command, ExitCode, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};

use crate::supervisor::{self, Event, Reporter, Work};

/// How often the parent checks on a child.
const POLL: Duration = Duration::from_millis(50);

/// A worker that runs this executable again as a child process with `args`.
pub fn child_work(args: Vec<OsString>, stop: Arc<AtomicBool>) -> Work {
    Box::new(move |reporter| {
        let exe = std::env::current_exe().context("could not find the burnin executable")?;
        let mut command = Command::new(exe);
        command.args(&args);
        relay(command, &stop, reporter)
    })
}

/// Runs `command` as a worker process and forwards its events to `reporter`
/// until it exits.
fn relay(mut command: Command, stop: &AtomicBool, reporter: &Reporter) -> Result<()> {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("could not start a worker process")?;
    let stdout = child
        .stdout
        .take()
        .context("the worker process has no stdout")?;
    let mut stdin = child.stdin.take();

    // Events are forwarded on another thread so this one can watch for stop
    // and give-up requests.
    let forwarder = reporter.clone();
    let reader = thread::spawn(move || forward_events(stdout, &forwarder));

    let status = loop {
        if stop.load(Ordering::SeqCst) {
            // Closing stdin asks the child to finish its current chunk and exit.
            drop(stdin.take());
        }
        if reporter.abandoned() {
            // The GPU may be stuck in the driver, so don't wait for it.
            let _ = child.kill();
        }
        if let Some(status) = child
            .try_wait()
            .context("could not check on the worker process")?
        {
            break status;
        }
        thread::sleep(POLL);
    };

    let outcome = reader
        .join()
        .map_err(|_| anyhow!("the thread reading worker output panicked"))?;
    match outcome {
        Some(Ok(())) => Ok(()),
        Some(Err(message)) => Err(anyhow!(message)),
        None => Err(anyhow!(
            "the worker process exited ({status}) without reporting a result"
        )),
    }
}

/// Forwards a child's events until it reports how it ended, and returns that
/// outcome. Returns `None` if its output ends first.
fn forward_events(stdout: impl Read, reporter: &Reporter) -> Option<Result<(), String>> {
    for line in BufReader::new(stdout).lines() {
        let Ok(line) = line else { break };
        match serde_json::from_str(&line) {
            Ok(Event::Finished) => return Some(Ok(())),
            Ok(Event::Failed(message)) => return Some(Err(message)),
            Ok(event) => reporter.send(event),
            // Only events should reach stdout, so anything else is a bug worth seeing.
            Err(_) => eprintln!("burnin: ignoring unexpected worker output: {line}"),
        }
    }
    None
}

/// Runs a worker in this process on behalf of a parent `burnin`, writing its
/// events to stdout. `make_work` receives the stop flag, which is set when
/// stdin closes.
pub fn serve(make_work: impl FnOnce(Arc<AtomicBool>) -> Work) -> ExitCode {
    let stop = Arc::new(AtomicBool::new(false));
    let closed = stop.clone();
    thread::spawn(move || {
        // Nothing is ever sent on stdin; it only closes.
        let _ = io::copy(&mut io::stdin().lock(), &mut io::sink());
        closed.store(true, Ordering::SeqCst);
    });
    ignore_first_interrupt();

    let (reporter, rx) = Reporter::detached();
    let writer = thread::spawn(move || {
        let mut finished = false;
        for message in rx {
            finished = matches!(message.event, Event::Finished);
            let line = serde_json::to_string(&message.event).expect("events always serialize");
            // Lock per line so that a stray print elsewhere can't block the writer.
            let mut stdout = io::stdout().lock();
            // If the parent has gone there is no one left to tell.
            let _ = writeln!(stdout, "{line}").and_then(|()| stdout.flush());
        }
        finished
    });
    // The reporter is dropped when the worker returns, which ends the writer.
    supervisor::run_worker(make_work(stop), reporter);
    match writer.join() {
        Ok(true) => ExitCode::SUCCESS,
        _ => ExitCode::FAILURE,
    }
}

/// The parent handles Ctrl-C and stops its children by closing their stdin, so
/// a child ignores the first interrupt. A second one exits at once, as it does
/// in the parent.
fn ignore_first_interrupt() {
    let presses = AtomicUsize::new(0);
    let _ = ctrlc::set_handler(move || {
        if presses.fetch_add(1, Ordering::SeqCst) > 0 {
            std::process::exit(130);
        }
    });
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::supervisor::Ready;
    use std::time::Instant;

    /// A shell command that prints each event as a line, after running `before`.
    fn script(before: &str, events: &[Event]) -> Command {
        let lines: Vec<String> = events
            .iter()
            .map(|event| serde_json::to_string(event).unwrap())
            .collect();
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg(format!("{before} printf '%s\\n' \"$@\""))
            .arg("sh")
            .args(lines);
        command
    }

    fn ready() -> Event {
        Event::Ready(Ready {
            detail: "child".to_string(),
            flops_per_gemm: 1.0,
            chunk_secs: 0.1,
        })
    }

    #[test]
    fn forwards_events_and_reports_success() {
        let (reporter, rx) = Reporter::detached();
        let stop = AtomicBool::new(false);
        let command = script(
            "",
            &[
                ready(),
                Event::Progress { pass: 1, gemms: 4 },
                Event::Finished,
            ],
        );
        relay(command, &stop, &reporter).unwrap();
        let events: Vec<Event> = rx.try_iter().map(|message| message.event).collect();
        assert!(matches!(
            events[..],
            [Event::Ready(_), Event::Progress { pass: 1, gemms: 4 }]
        ));
    }

    #[test]
    fn reports_the_child_error() {
        let (reporter, _rx) = Reporter::detached();
        let stop = AtomicBool::new(false);
        let command = script("", &[ready(), Event::Failed("out of memory".into())]);
        let err = relay(command, &stop, &reporter).unwrap_err();
        assert_eq!(err.to_string(), "out of memory");
    }

    #[test]
    fn reports_a_child_that_dies_without_a_result() {
        let (reporter, _rx) = Reporter::detached();
        let stop = AtomicBool::new(false);
        let err = relay(script("exit 3;", &[]), &stop, &reporter).unwrap_err();
        assert!(err.to_string().contains("exit status: 3"), "{err}");
    }

    #[test]
    fn stopping_closes_the_childs_stdin() {
        let (reporter, _rx) = Reporter::detached();
        let stop = AtomicBool::new(true);
        // The child only finishes once its stdin closes.
        let command = script("cat > /dev/null;", &[Event::Finished]);
        relay(command, &stop, &reporter).unwrap();
    }

    #[test]
    fn kills_a_child_that_was_given_up_on() {
        let (reporter, _rx) = Reporter::detached();
        reporter.abandon();
        let stop = AtomicBool::new(true);
        // Ignores stdin closing, like a GPU stuck in the driver.
        let command = script("exec sleep 30;", &[Event::Finished]);
        let started = Instant::now();
        let err = relay(command, &stop, &reporter).unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(err.to_string().contains("signal"), "{err}");
    }
}
