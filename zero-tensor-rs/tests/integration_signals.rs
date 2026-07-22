use std::{
    path::Path,
    process::{Command, Stdio},
    thread,
    time::Duration,
};

use tempfile::tempdir;

#[test]
fn test_cleanup_on_intercept() {
    let dir = tempdir().unwrap();
    let sock_path = dir.path().join("integration_test.sock");
    let shm_name = "zt_integration_test_shm";

    let bin_path = env!("CARGO_BIN_EXE_throughput_bench");

    let mut child = Command::new(bin_path)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("Failed to run executable");

    thread::sleep(Duration::from_millis(500));

    let pid = child.id();
    unsafe {
        libc::kill(pid as i32, libc::SIGINT);
    }
    let status = child.wait().expect("Failed to wait on child process");
    assert!(
        !status.success(),
        "Process should exit with non-zero code on SIGINT"
    );
    assert!(
        !sock_path.exists(),
        "Socket file must be cleaned up on SIGINT!"
    );

    #[cfg(target_os = "linux")]
    {
        let shm_path = format!("/dev/shm/{}", shm_name);
        assert!(
            !Path::new(&shm_path).exists(),
            "SHM segment must be unlinked on SIGINT!"
        );
    }
}
