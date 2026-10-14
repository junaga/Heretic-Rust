// SPDX-License-Identifier: AGPL-3.0-or-later
//! Application utilities implemented without a Python dependency.
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path, time::Duration};

pub fn load_json(path: &str) -> Result<Value> {
    let text = if path.to_ascii_lowercase().starts_with("https://")
        || path.to_ascii_lowercase().starts_with("http://")
    {
        let url = path
            .replace("/blob/", "/raw/")
            .replace("/src/branch/", "/raw/branch/");
        ureq::get(&url)
            .timeout(Duration::from_secs(60))
            .call()
            .with_context(|| format!("Downloading {url}"))?
            .into_string()?
    } else {
        std::fs::read_to_string(path).with_context(|| format!("Reading {path}"))?
    };
    serde_json::from_str(&text).context("Invalid reproduction JSON")
}
pub fn timestamp() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string()
}
pub fn config_toml(settings: &crate::config::Settings) -> Result<String> {
    let mut value = settings.exported();
    fn clean(v: &mut Value) {
        match v {
            Value::Object(map) => {
                map.retain(|_, v| !v.is_null());
                for v in map.values_mut() {
                    clean(v);
                }
            }
            Value::Array(a) => {
                for v in a {
                    clean(v);
                }
            }
            _ => {}
        }
    }
    clean(&mut value);
    Ok(toml::to_string_pretty(&value)?)
}
pub fn resident_memory() -> Result<u64> {
    let pid = sysinfo::get_current_pid().map_err(anyhow::Error::msg)?;
    let mut system = sysinfo::System::new();
    system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    system
        .process(pid)
        .map(|p| p.memory())
        .context("Reading process memory usage")
}
pub fn cpu_info() -> Value {
    let mut system = sysinfo::System::new();
    system.refresh_cpu_all();
    let cpu = system.cpus().first();
    let mut result = json!({"brand":cpu.map(|c|c.brand().trim()).filter(|s|!s.is_empty()),"vendor":cpu.map(|c|c.vendor_id().trim()).filter(|s|!s.is_empty()),"family":null,"model":null,"stepping":null});
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::__cpuid;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::__cpuid;
        // CPUID leaf 1 is available on the x86 processors supported by Rust.
        let eax = unsafe { __cpuid(1) }.eax;
        let base_family = (eax >> 8) & 15;
        let family = base_family
            + if base_family == 15 {
                (eax >> 20) & 255
            } else {
                0
            };
        let model = ((eax >> 4) & 15)
            + if base_family == 6 || base_family == 15 {
                ((eax >> 16) & 15) << 4
            } else {
                0
            };
        result["family"] = json!(family);
        result["model"] = json!(model);
        result["stepping"] = json!(eax & 15);
    }
    // /proc exposes the same identifiers used by py-cpuinfo on Linux.
    if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
        let fields: BTreeMap<_, _> = text
            .split("\n\n")
            .next()
            .unwrap_or("")
            .lines()
            .filter_map(|s| s.split_once(':'))
            .map(|(k, v)| (k.trim(), v.trim()))
            .collect();
        if let Some(brand) = fields.get("model name") {
            result["brand"] = json!(brand);
        }
        if let Some(vendor) = fields.get("vendor_id") {
            result["vendor"] = json!(vendor);
        }
        for (key, name) in [
            ("family", "cpu family"),
            ("model", "model"),
            ("stepping", "stepping"),
        ] {
            result[key] = fields
                .get(name)
                .and_then(|s| s.parse::<u64>().ok())
                .map(|n| json!(n))
                .unwrap_or(Value::Null);
        }
    }
    result
}
pub fn format_duration(seconds: f64) -> String {
    let seconds = seconds.round_ties_even() as u64;
    if seconds < 60 {
        format!("{seconds}s")
    } else if seconds < 3600 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
    }
}
pub fn sha256sums(hashes: &Value) -> String {
    let mut entries: Vec<_> = hashes
        .as_object()
        .into_iter()
        .flat_map(|m| m.iter())
        .collect();
    entries.sort_by_key(|(name, _)| *name);
    entries
        .into_iter()
        .map(|(name, sha)| format!("{} *{name}\n", sha.as_str().unwrap_or("")))
        .collect()
}
pub fn safe_filename(name: &str) -> bool {
    !name.is_empty()
        && Path::new(name)
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_)))
        && !name.contains('\\')
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn timestamp_and_cpu() {
        let time = timestamp();
        assert_eq!(time.len(), 19);
        assert!(!time.ends_with('Z'));
        assert!(cpu_info().get("brand").is_some());
        assert!(resident_memory().unwrap() > 0);
    }
    #[test]
    fn checksums_sorted() {
        assert_eq!(sha256sums(&json!({"z":"bb","a":"aa"})), "aa *a\nbb *z\n");
        assert!(!safe_filename("../weights"));
        assert!(!safe_filename("/weights"));
    }
}
