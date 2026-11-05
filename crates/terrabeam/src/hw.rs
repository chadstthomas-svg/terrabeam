//! Hardware detection: which llama.cpp build to fetch and how much memory a quant may use.

use std::process::Command;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Accel {
    /// NVIDIA with the highest CUDA version the driver supports (nvidia-smi header), e.g. (13, 0)
    Cuda { cuda: (u32, u32) },
    Metal,
    Vulkan,
    Cpu,
}

#[derive(Clone, Debug)]
pub struct Hw {
    pub os: &'static str,
    pub arch: &'static str,
    pub accel: Accel,
    pub gpus: Vec<(String, u64)>, // (name, total VRAM MiB)
    pub ram_mib: u64,
}

impl Hw {
    pub fn vram_mib(&self) -> u64 {
        self.gpus.iter().map(|g| g.1).sum()
    }
    pub fn describe(&self) -> String {
        let g: Vec<String> = self.gpus.iter().map(|(n, v)| format!("{n} {:.1} GB", *v as f64 / 1024.0)).collect();
        format!("{}-{} {:?}, {} RAM {:.0} GB", self.os, self.arch, self.accel, if g.is_empty() { "no GPU".into() } else { g.join(" + ") },
            self.ram_mib as f64 / 1024.0)
    }
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let o = Command::new(cmd).args(args).output().ok()?;
    o.status.success().then(|| String::from_utf8_lossy(&o.stdout).into_owned())
}

pub fn detect() -> Hw {
    let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
    let ram_mib = ram_mib();
    if os == "macos" {
        // Apple Silicon: unified memory, Metal build; treat ~70% of RAM as usable by the GPU
        let accel = if arch == "aarch64" { Accel::Metal } else { Accel::Cpu };
        let gpus = if arch == "aarch64" { vec![("Apple GPU (unified)".into(), ram_mib * 7 / 10)] } else { vec![] };
        return Hw { os, arch, accel, gpus, ram_mib };
    }
    if let Some(q) = run("nvidia-smi", &["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]) {
        let gpus: Vec<(String, u64)> = q
            .lines()
            .filter_map(|l| {
                let (n, m) = l.rsplit_once(',')?;
                Some((n.trim().to_string(), m.trim().parse().ok()?))
            })
            .collect();
        let cuda = run("nvidia-smi", &[])
            .and_then(|h| {
                let i = h.find("CUDA Version:")?;
                let v = h[i + 13..].split_whitespace().next()?.to_string();
                let (a, b) = v.split_once('.')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            })
            .unwrap_or((12, 0));
        if !gpus.is_empty() {
            return Hw { os, arch, accel: Accel::Cuda { cuda }, gpus, ram_mib };
        }
    }
    // AMD / Intel GPU: the Vulkan build (needs a Vulkan loader; vulkaninfo tells whether one is present)
    if let Some(v) = run("vulkaninfo", &["--summary"]) {
        let names: Vec<String> = v
            .lines()
            .filter(|l| l.trim_start().starts_with("deviceName"))
            .filter_map(|l| l.split('=').nth(1).map(|s| s.trim().to_string()))
            .filter(|n| !n.to_lowercase().contains("llvmpipe"))
            .collect();
        if !names.is_empty() {
            // VRAM is not in the summary; llama.cpp fits layers itself, so the quant choice uses RAM only
            return Hw { os, arch, accel: Accel::Vulkan, gpus: names.into_iter().map(|n| (n, 0)).collect(), ram_mib };
        }
    }
    Hw { os, arch, accel: Accel::Cpu, gpus: vec![], ram_mib }
}

fn ram_mib() -> u64 {
    if let Ok(m) = std::fs::read_to_string("/proc/meminfo") {
        if let Some(l) = m.lines().find(|l| l.starts_with("MemTotal:")) {
            return l.split_whitespace().nth(1).and_then(|k| k.parse::<u64>().ok()).unwrap_or(0) / 1024;
        }
    }
    if let Some(b) = run("sysctl", &["-n", "hw.memsize"]) {
        return b.trim().parse::<u64>().unwrap_or(0) / (1024 * 1024);
    }
    if let Some(w) = run("powershell", &["-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"]) {
        return w.trim().parse::<u64>().unwrap_or(0) / (1024 * 1024);
    }
    16 * 1024
}

/// llama.cpp release assets for this machine (the main archive first, then a CUDA runtime bundle when one is needed).
pub fn llama_assets(hw: &Hw, tag: &str) -> Vec<String> {
    let (os, arch) = (hw.os, hw.arch);
    let cuda_v = |c: (u32, u32)| if c >= (13, 4) { "13.4" } else { "12.8" };
    match (os, arch, &hw.accel) {
        ("linux", "x86_64", Accel::Cuda { cuda }) => {
            let v = cuda_v(*cuda);
            vec![format!("llama-{tag}-bin-ubuntu-cuda-{v}-x64.tar.gz"), format!("cudart-llama-{tag}-bin-ubuntu-cuda-{v}-x64.tar.gz")]
        }
        ("linux", "aarch64", Accel::Cuda { .. }) => vec![
            format!("llama-{tag}-bin-ubuntu-cuda-13.4-arm64.tar.gz"),
            format!("cudart-llama-{tag}-bin-ubuntu-cuda-13.4-arm64.tar.gz"),
        ],
        ("linux", "x86_64", Accel::Vulkan) => vec![format!("llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz")],
        ("linux", "aarch64", Accel::Vulkan) => vec![format!("llama-{tag}-bin-ubuntu-vulkan-arm64.tar.gz")],
        ("linux", "x86_64", _) => vec![format!("llama-{tag}-bin-ubuntu-x64.tar.gz")],
        ("linux", "aarch64", _) => vec![format!("llama-{tag}-bin-ubuntu-arm64.tar.gz")],
        ("macos", "aarch64", _) => vec![format!("llama-{tag}-bin-macos-arm64.tar.gz")],
        ("macos", _, _) => vec![format!("llama-{tag}-bin-macos-x64.tar.gz")],
        ("windows", "x86_64", Accel::Cuda { cuda }) => {
            let v = if *cuda >= (13, 4) { "13.4" } else { "12.4" };
            vec![format!("llama-{tag}-bin-win-cuda-{v}-x64.zip"), format!("cudart-llama-bin-win-cuda-{v}-x64.zip")]
        }
        ("windows", "x86_64", Accel::Vulkan) => vec![format!("llama-{tag}-bin-win-vulkan-x64.zip")],
        ("windows", "x86_64", _) => vec![format!("llama-{tag}-bin-win-cpu-x64.zip")],
        ("windows", "aarch64", _) => vec![format!("llama-{tag}-bin-win-cpu-arm64.zip")],
        _ => vec![format!("llama-{tag}-bin-ubuntu-x64.tar.gz")],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hw(os: &'static str, arch: &'static str, accel: Accel) -> Hw {
        Hw { os, arch, accel, gpus: vec![], ram_mib: 65536 }
    }

    #[test]
    fn picks_the_matching_release_assets() {
        let t = "b12078";
        assert_eq!(llama_assets(&hw("linux", "x86_64", Accel::Cuda { cuda: (13, 0) }), t)[0], "llama-b12078-bin-ubuntu-cuda-12.8-x64.tar.gz");
        assert_eq!(llama_assets(&hw("linux", "x86_64", Accel::Cuda { cuda: (13, 4) }), t)[1], "cudart-llama-b12078-bin-ubuntu-cuda-13.4-x64.tar.gz");
        assert_eq!(llama_assets(&hw("macos", "aarch64", Accel::Metal), t), vec!["llama-b12078-bin-macos-arm64.tar.gz"]);
        assert_eq!(llama_assets(&hw("windows", "x86_64", Accel::Cuda { cuda: (12, 6) }), t)[1], "cudart-llama-bin-win-cuda-12.4-x64.zip");
        assert_eq!(llama_assets(&hw("linux", "x86_64", Accel::Cpu), t), vec!["llama-b12078-bin-ubuntu-x64.tar.gz"]);
    }
}
