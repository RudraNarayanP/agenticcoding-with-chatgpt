//! Small cross-platform helpers shared across the crate.

use anyhow::Result;
use std::path::PathBuf;

/// User home directory (`HOME` on Unix, `USERPROFILE` on Windows).
pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Read `n` cryptographically suitable random bytes without pulling in an RNG crate.
pub fn random_bytes(n: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    fill_random(&mut buf)?;
    Ok(buf)
}

fn fill_random(buf: &mut [u8]) -> Result<()> {
    #[cfg(unix)]
    {
        let mut f = std::fs::File::open("/dev/urandom").context("opening /dev/urandom")?;
        f.read_exact(buf).context("reading random bytes")?;
        return Ok(());
    }

    #[cfg(windows)]
    {
        // RtlGenRandom (SystemFunction036) — available on Windows XP+.
        #[link(name = "advapi32")]
        extern "system" {
            #[link_name = "SystemFunction036"]
            fn rtl_gen_random(buffer: *mut u8, len: u32) -> u8;
        }

        let ok = unsafe { rtl_gen_random(buf.as_mut_ptr(), buf.len() as u32) };
        if ok == 0 {
            anyhow::bail!("RtlGenRandom failed");
        }
        return Ok(());
    }
}
