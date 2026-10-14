use sha2::{Digest, Sha256};
use std::{path::Path, process::Command};
fn main() {
    fn collect(path: &Path, files: &mut Vec<std::path::PathBuf>) {
        if path.is_dir() {
            for entry in std::fs::read_dir(path).unwrap() {
                collect(&entry.unwrap().path(), files);
            }
        } else {
            files.push(path.into());
        }
    }
    let root = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let root = Path::new(&root);
    let mut files = Vec::new();
    collect(&root.join("src"), &mut files);
    files.extend(
        [
            "Cargo.toml",
            "Cargo.lock",
            "build.rs",
            "requirements.txt",
            "resources/settings.json",
        ]
        .map(|s| root.join(s)),
    );
    files.sort();
    let mut hash = Sha256::new();
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        hash.update(
            path.strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .as_bytes(),
        );
        hash.update(std::fs::read(path).unwrap());
    }
    let revision = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".into());
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(root)
        .output()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(true);
    println!("cargo:rustc-env=HERETIC_SOURCE_REVISION={revision}");
    println!("cargo:rustc-env=HERETIC_SOURCE_DIRTY={dirty}");
    println!("cargo:rustc-env=HERETIC_SOURCE_HASH={:x}", hash.finalize());
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/heads");
    println!("cargo:rerun-if-changed=.git/index");
}
