//! Small cross-platform helpers shared across the crate.
//!
//! Two things the browser-facing code kept getting wrong on Windows: where the
//! user's home directory is, and how to get random bytes. Both are answered
//! here once so no caller has to carry a `#[cfg]` branch.

use anyhow::Result;
use std::path::PathBuf;
use std::process::Command;

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

#[cfg(test)]
mod tests {
    use super::*;

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
