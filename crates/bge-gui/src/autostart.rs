//! "Start when I log in" (Settings screen, design/DESIGN.md "Access &
//! startup"): a value under HKCU's Run key pointing at this executable.
//! Windows-only - a no-op stub on other targets, since eframe also builds
//! for x11/wayland (Cargo.toml) but this app has no autostart mechanism
//! there yet.

#[cfg(windows)]
mod imp {
    use std::io;
    use windows_sys::Win32::Foundation::{ERROR_FILE_NOT_FOUND, ERROR_SUCCESS};
    use windows_sys::Win32::System::Registry::{
        HKEY, HKEY_CURRENT_USER, KEY_QUERY_VALUE, KEY_SET_VALUE, REG_SZ, RegCloseKey,
        RegDeleteValueW, RegOpenKeyExW, RegQueryValueExW, RegSetValueExW,
    };

    const RUN_KEY: &str = "Software\\Microsoft\\Windows\\CurrentVersion\\Run";
    /// Value name under `RUN_KEY`; also doubles as the app's registry-visible
    /// identity, so it must stay stable across releases.
    const VALUE_NAME: &str = "bge-embed-rs";

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    fn open(access: u32) -> io::Result<HKEY> {
        let mut hkey: HKEY = std::ptr::null_mut();
        let subkey = wide(RUN_KEY);
        let status =
            unsafe { RegOpenKeyExW(HKEY_CURRENT_USER, subkey.as_ptr(), 0, access, &mut hkey) };
        if status != ERROR_SUCCESS {
            return Err(io::Error::from_raw_os_error(status as i32));
        }
        Ok(hkey)
    }

    /// True iff a Run value exists under our name. A stale value (pointing at
    /// a since-moved exe) still counts as "on" for the toggle - `set_enabled`
    /// always rewrites the path, so flipping the toggle off then on again
    /// self-heals it.
    pub fn is_enabled() -> bool {
        let Ok(hkey) = open(KEY_QUERY_VALUE) else {
            return false;
        };
        let name = wide(VALUE_NAME);
        let status = unsafe {
            RegQueryValueExW(
                hkey,
                name.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        unsafe { RegCloseKey(hkey) };
        status == ERROR_SUCCESS
    }

    pub fn set_enabled(enabled: bool) -> io::Result<()> {
        let hkey = open(KEY_SET_VALUE)?;
        let name = wide(VALUE_NAME);
        let result = if enabled {
            let exe = std::env::current_exe()?;
            // Run values are parsed as a raw command line; an unquoted path
            // containing spaces (e.g. "Program Files") would be split at the
            // first space, so the whole path is quoted.
            let command = wide(&format!("\"{}\"", exe.display()));
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(command.as_ptr().cast::<u8>(), command.len() * 2)
            };
            let status = unsafe {
                RegSetValueExW(
                    hkey,
                    name.as_ptr(),
                    0,
                    REG_SZ,
                    bytes.as_ptr(),
                    bytes.len() as u32,
                )
            };
            status_result(status)
        } else {
            let status = unsafe { RegDeleteValueW(hkey, name.as_ptr()) };
            // Already absent is success for our purposes - "off" is the goal.
            if status == ERROR_FILE_NOT_FOUND {
                Ok(())
            } else {
                status_result(status)
            }
        };
        unsafe { RegCloseKey(hkey) };
        result
    }

    fn status_result(status: u32) -> io::Result<()> {
        if status == ERROR_SUCCESS {
            Ok(())
        } else {
            Err(io::Error::from_raw_os_error(status as i32))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Writes the real HKCU Run key. Restoring only restores on/off: an
        /// existing entry would come back pointing at the *test* binary
        /// (current_exe), which then runs at every login - hence ignored by
        /// default. Run with `cargo test -- --ignored` on a machine where
        /// autostart is off.
        #[test]
        #[ignore = "mutates the real HKCU Run key"]
        fn round_trip_enable_disable() {
            let was_enabled = is_enabled();

            set_enabled(true).unwrap();
            assert!(is_enabled());

            set_enabled(false).unwrap();
            assert!(!is_enabled());

            if was_enabled {
                set_enabled(true).unwrap();
            }
        }
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn is_enabled() -> bool {
        false
    }

    pub fn set_enabled(_enabled: bool) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "start-on-login is only implemented on Windows",
        ))
    }
}

pub use imp::{is_enabled, set_enabled};
