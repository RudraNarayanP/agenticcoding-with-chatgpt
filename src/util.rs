//! Small cross-platform helpers shared across the crate.
//!
//! Two things the browser-facing code kept getting wrong on Windows: where the
//! user's home directory is, and how to get random bytes. Both are answered
//! here once so no caller has to carry a `#[cfg]` branch.

use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

/// User home directory (`HOME` on Unix, `USERPROFILE` on Windows).
///
/// The first Windows port read `HOME` directly, which is unset in a normal
/// cmd/PowerShell session — so `init`, the ledger, the surface lock and the
/// project cache all silently resolved to the current directory.
pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Read `n` cryptographically suitable random bytes.
///
/// `getrandom` is already in the tree (through `jsonschema`), so this adds no
/// weight to the binary and retires the hand-rolled `RtlGenRandom` FFI the
/// first Windows port needed.
pub fn random_bytes(n: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    // `map_err`, not `with_context`: `getrandom::Error` only implements
    // `std::error::Error` under its `std` feature, which the rest of the graph
    // does not enable, and anyhow's `Context` requires that bound.
    getrandom::fill(&mut buf).map_err(|e| anyhow::anyhow!("obtaining {n} random bytes: {e}"))?;
    Ok(buf)
}

/// Build a `std::process::Command` for `program`, resolving it the way a shell would.
///
/// On Windows `CreateProcessW` only runs PE executables, so an `.exe` on `PATH`
/// is found by itself but an npm shim — `codex.cmd`, `claude.cmd` — is not, and
/// fails as "program not found" or "not a valid Win32 application". Those have
/// to go through `cmd.exe /c`. On Unix this is a plain `Command::new`.
///
/// `program` should be a bare name or a full path, never a command line: it is
/// passed as a single argument, so a caller cannot smuggle extra arguments in.
pub fn command(program: &str) -> Command {
    #[cfg(windows)]
    {
        if let Some(resolved) = which(program) {
            return command_at(&resolved);
        }
    }
    Command::new(program)
}

/// Launch a program whose location we already know, applying the Windows
/// `.cmd`/`.bat` rule from `command()`. Callers that resolved a binary
/// themselves, such as the chrome-use lookup in `crate::channel`, use this.
pub fn command_at(resolved: &std::path::Path) -> Command {
    #[cfg(windows)]
    {
        let ext = resolved
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        if ext == "cmd" || ext == "bat" {
            // `cmd.exe /C <path> <args…>`: the shim's own path is a single
            // argument, so standard escaping keeps `C:\Program Files\…` intact.
            let mut cmd = Command::new("cmd");
            cmd.arg("/C").arg(resolved);
            return cmd;
        }
    }
    #[cfg(not(windows))]
    {
        let _ = resolved;
    }
    Command::new(resolved)
}

/// Locate an executable on `PATH`, honouring `PATHEXT` on Windows.
///
/// Deliberately our own rather than relying on `CreateProcess`'s search, because
/// callers need the *path* — both to report it in errors and to decide how to
/// launch it (see `command`).
pub fn which(program: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    // `split_paths`, not `split(':')`: Windows separates on ';' and every entry
    // contains a ':' anyway (`C:\Windows`), so a hand-rolled split turns one
    // PATH into garbage.
    for dir in std::env::split_paths(&path_var) {
        if dir.as_os_str().is_empty() {
            continue;
        }
        for candidate in candidate_paths(&dir, program) {
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// File names worth trying for `program`: the bare name, then on Windows every
/// `PATHEXT` extension.
fn candidate_names(program: &str) -> Vec<String> {
    // `mut` is only used on Windows, where PATHEXT adds entries below.
    #[allow(unused_mut)]
    let mut out = vec![program.to_string()];
    #[cfg(windows)]
    {
        // `PATHEXT` entries carry their own leading dot and are usually
        // uppercase (`.EXE`); a few setups drop the dot, so normalize it back.
        let exts = std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string())
            .split(';')
            .filter(|e| !e.is_empty())
            .map(|e| {
                let e = e.to_ascii_lowercase();
                if e.starts_with('.') {
                    e
                } else {
                    format!(".{e}")
                }
            })
            .collect::<Vec<_>>();
        // A name that already carries a known extension must not get a second.
        // Compare in the same dotted form the list holds — `Path::extension`
        // strips the dot, and matching a bare `exe` against `.EXE` is how this
        // ends up producing `chrome-use.exe.exe`.
        let already = std::path::Path::new(program)
            .extension()
            .map(|e| exts.contains(&format!(".{}", e.to_string_lossy().to_ascii_lowercase())))
            .unwrap_or(false);
        if !already {
            out.extend(exts.into_iter().map(|ext| format!("{program}{ext}")));
        }
    }
    out
}

/// `dir` joined to every name worth trying for `program`.
pub fn candidate_paths(dir: &std::path::Path, program: &str) -> Vec<PathBuf> {
    candidate_names(program)
        .into_iter()
        .map(|n| dir.join(n))
        .collect()
}

/// Read `pipe` to EOF on a worker thread and hand the bytes to `tx`. Generic over
/// the pipe type because stdout and stderr are distinct types.
fn drain<R: std::io::Read + Send + 'static>(
    pipe: Option<R>,
    tx: std::sync::mpsc::SyncSender<Vec<u8>>,
) {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut p) = pipe {
            let _ = std::io::Read::read_to_end(&mut p, &mut buf);
        }
        // try_send: the receiver may already have given up on its bounded join.
        let _ = tx.try_send(buf);
    });
}

/// Run `cmd` to completion, capturing its streams, but never for longer than
/// `timeout_secs`.
///
/// This exists because `Command::output()` blocks until the child exits, and the
/// chrome-use transport took a timeout argument and ignored it. Every call site
/// passes a budget, so `--timeout` looked like it governed a run while a single
/// wedged chrome-use call actually held the process forever -- a hang no deadline
/// inside the caller can interrupt.
///
/// The streams are drained on reader threads rather than by `output()` so a
/// chatty child cannot deadlock on a full pipe while we wait to notice it
/// finished. The joins are bounded too, on purpose: `chrome-use` hands work to
/// long-lived session daemons, and a daemon that inherits a pipe handle can keep
/// it open after its client exits, which would otherwise turn a hang we removed
/// back into a hang we reintroduced. A caller that waits for output it never gets
/// is worse off than one that waits briefly and reports what it has.
pub fn run_bounded(cmd: &mut Command, timeout_secs: f64) -> Result<std::process::Output> {
    let budget = Duration::try_from_secs_f64(timeout_secs.max(1.0))
        .unwrap_or_else(|_| Duration::from_secs(1));
    let mut child = cmd
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {}", cmd.get_program().to_string_lossy()))?;

    let (out_tx, out_rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(1);
    let (err_tx, err_rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(1);
    drain(child.stdout.take(), out_tx);
    drain(child.stderr.take(), err_tx);

    let deadline = Instant::now() + budget;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    bail!(
                        "timed out after {}s (the process was stopped; it may have been wedged)",
                        budget.as_secs_f64()
                    );
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(e).context("waiting for the child process"),
        }
    };

    // Bounded join, per the note above. 2s is far past any real drain of a
    // finished child and still keeps us off the hang path.
    let wait = |rx: std::sync::mpsc::Receiver<Vec<u8>>| {
        rx.recv_timeout(Duration::from_secs(2)).unwrap_or_default()
    };
    Ok(std::process::Output {
        status,
        stdout: wait(out_rx),
        stderr: wait(err_rx),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_bounded_stops_a_hung_child_instead_of_waiting_for_it() {
        // The bug this retires: `Command::output()` waits forever, so a wedged
        // chrome-use call held chatgpt-use indefinitely and `--timeout` never
        // fired. A child that sleeps far past the budget must produce an error,
        // quickly, and must not be left running.
        let mut cmd = if cfg!(windows) {
            let mut c = Command::new("powershell");
            c.args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Start-Sleep -Seconds 30",
            ]);
            c
        } else {
            let mut c = Command::new("sleep");
            c.arg("30");
            c
        };
        let start = Instant::now();
        let err = run_bounded(&mut cmd, 1.0).expect_err("a 30s child must not finish in 1s");
        let elapsed = start.elapsed();
        assert!(err.to_string().contains("timed out"), "{err}");
        assert!(
            elapsed < Duration::from_secs(10),
            "took {elapsed:?} to give up"
        );
    }

    #[test]
    fn run_bounded_returns_exit_status_and_output() {
        let mut cmd = if cfg!(windows) {
            let mut c = Command::new("cmd");
            c.args(["/C", "echo bounded-ok && exit 7"]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", "echo bounded-ok; exit 7"]);
            c
        };
        let out = run_bounded(&mut cmd, 30.0).unwrap();
        assert_eq!(
            out.status.code(),
            Some(7),
            "the child's exit code must survive"
        );
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("bounded-ok"),
            "stdout must survive the reader threads"
        );
    }

    #[test]
    fn random_bytes_are_random_and_the_right_length() {
        let a = random_bytes(16).unwrap();
        let b = random_bytes(16).unwrap();
        assert_eq!(a.len(), 16);
        assert_ne!(a, b, "two draws must not be identical");
        assert!(random_bytes(0).unwrap().is_empty());
    }

    #[test]
    fn home_dir_is_absolute_when_present() {
        if let Some(h) = home_dir() {
            assert!(h.is_absolute(), "{} should be absolute", h.display());
        }
    }

    #[test]
    fn which_finds_a_binary_that_is_really_on_path() {
        // Whatever runs these tests has *something* on PATH that we can find by
        // name, and `which` must agree with the OS about it.
        let found = which("git")
            .or_else(|| which("cmd"))
            .or_else(|| which("sh"));
        assert!(
            found.is_some(),
            "git, cmd or sh should be resolvable on PATH"
        );
        assert!(found.unwrap().is_file());
    }

    #[test]
    fn which_rejects_names_that_are_not_programs() {
        assert!(which("definitely-not-a-real-program-xyzzy-9271").is_none());
        // A directory on PATH must not be reported as an executable.
        assert!(which(".").is_none());
    }

    #[cfg(windows)]
    #[test]
    fn candidates_add_pathext_but_never_a_second_extension() {
        // The bug this guards: `chrome-use` must be able to resolve to
        // `chrome-use.exe`, and `chrome-use.exe` must not become
        // `chrome-use.exe.exe`.
        let bare = candidate_paths(std::path::Path::new("C:\\bin"), "chrome-use");
        assert!(
            bare.iter().any(|p| p.ends_with("chrome-use.exe")),
            "{bare:?}"
        );
        assert!(
            bare.iter().any(|p| p.ends_with("chrome-use.cmd")),
            "{bare:?}"
        );

        let with_ext = candidate_paths(std::path::Path::new("C:\\bin"), "chrome-use.exe");
        assert_eq!(
            with_ext.len(),
            1,
            "an explicit extension must not be doubled"
        );
        assert!(with_ext[0].ends_with("chrome-use.exe"));
    }

    #[cfg(windows)]
    #[test]
    fn a_bat_shim_only_runs_through_cmd_exe() {
        // CreateProcessW rejects a `.bat`, which is exactly how npm installs
        // `codex` and `claude`. Prove `command_at` handles it by running
        // one, rather than by asserting about a Command we cannot inspect.
        let dir = std::env::temp_dir().join(format!("cgu-which-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bat = dir.join("cgu-probe.bat");
        std::fs::write(&bat, "@echo shim-ran\r\n").unwrap();

        let out = command_at(&bat).output();
        let _ = std::fs::remove_dir_all(&dir);

        let out = out.expect("a .bat must be launchable through cmd.exe");
        assert!(out.status.success(), "bat exit {:?}", out.status);
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "shim-ran");
    }
}
