// Embeds assets/ikon.ico as the Windows executable's icon (Explorer, taskbar
// before the window opens). The window itself gets the PNG at runtime
// (`with_icon` in main.rs), on every platform.
fn main() {
    println!("cargo:rerun-if-changed=assets/ikon.ico");
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/ikon.ico");
        res.compile()
            .expect("Windows resource (icon) failed to compile");
    }
}
