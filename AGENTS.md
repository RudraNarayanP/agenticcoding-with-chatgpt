# Agent rules for chatgpt-use

## Never test against the live ChatGPT

Do **not** run anything that reaches the real chatgpt.com to verify a change. That means:
- live `chatgpt-use ask / run / serve / work / resume / cancel` runs;
- probe scripts;
- `chrome-use site chatgpt/*` calls;
- opening chatgpt.com under throwaway `--session` names.

It drives the user's own signed-in account, and chatgpt.com throttles that account by **request count**. A full page load is about 45 backend requests. One afternoon of scripted live checks (roughly 23 full loads, mostly through fresh session names) tripped "Too many requests" on the account the user works in.

Verify offline instead:
- `cargo test`, which is fully offline and never calls chrome-use;
- pure functions for anything the browser decides (see `judge_record`, `classify`, `receipt::live_state`, `structured::evaluate`), tested with fixtures.

If something truly can only be confirmed live, say so and leave it marked unverified. Never run it yourself.

## Build

This fork is developed on Windows; the upstream build box (`leo@192.168.0.190`) is not ours and is
not reachable. Anything that touches the user's browser still must not be run, per the rule above.

- **MSRV is 1.89**, and it is measured, not assumed: `cargo check --all-targets` fails on 1.88
  because the surface lock uses `File::lock`, unstable until 1.89. `rust-version` in `Cargo.toml`
  records it.
- **Check both `cfg` families, not just the host's.** The bug that broke macOS and Linux was inside
  a `#[cfg(unix)]` block that the Windows host never compiled. `cargo check --target
  x86_64-pc-windows-gnu` and `--target aarch64-apple-darwin` type-check the foreign branches without
  a linker, from any OS (`.github/workflows/ci.yml` does this). From Linux you can also use Docker:
  `docker run --rm -v "$PWD":/src -w /src -e CARGO_TARGET_DIR=/tmp/t rust:1.98 cargo test` runs the
  real Unix suite.
- **Run the Windows suite twice.** The `bash` tool resolves to a POSIX shell when one exists and to
  PowerShell when not, and those are different code paths. `cargo test`, then
  `CHATGPT_USE_SHELL=powershell cargo test`. On a machine with Git for Windows the second one is the
  only way the fallback gets exercised.
- **Fresh binaries under `Desktop` are blocked here** by an application-control policy:
  `An Application Control policy has blocked this file (os error 4551)`, and `./target/debug/*.exe`
  then answers "Permission denied". Build with `CARGO_TARGET_DIR` pointed somewhere outside
  `Desktop` (e.g. `C:\Users\<you>\cgu-build`) and the same binaries run fine. Do not conclude a
  binary is broken when the loader refused it.
- **`core.autocrlf=true` on this machine**, so the working tree is CRLF while blobs are LF. That is
  why `.gitattributes` exists: a CRLF `install.sh` breaks `curl | sh` on Unix, and Windows PowerShell
  5.1 will not parse a here-string unless its terminator is CRLF. Prefer plain `Write-Host` lines in
  `.ps1` files over here-strings.
- Formatting and clippy are **not** clean repo-wide and were not made so: `cargo fmt --check`
  reports ~190 hunks (172 of them pre-existing) and clippy 18 warnings, all in upstream code. They
  are advisory CI steps. Do not "fix" them inside a functional change — a mechanical reformat buries
  the diff that needs reviewing.
