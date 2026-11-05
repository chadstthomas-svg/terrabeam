//! `tablinum.json` - the one file at the top of every Beam model repo that tells terrabeam how the model is meant to run.
//! Kept small and explicit so an agent reading the repo can follow it too (AGENTS.md points at it).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Manifest {
    pub schema: u32,
    pub name: String,
    /// pinned ggml-org/llama.cpp release tag (upstream builds run the model + MTP)
    pub llama_cpp: String,
    /// GGUF files, best quality first; terrabeam takes the best one that fits the hardware
    pub quants: Vec<Quant>,
    #[serde(default)]
    pub mtp: Option<Mtp>,
    #[serde(default = "default_ctx")]
    pub context: u32,
    /// llama-server defaults for requests that do not set them (Ling: T 1.0 / top_p .95 / top_k 20; greedy loops when thinking)
    #[serde(default)]
    pub sampling: BTreeMap<String, f64>,
    #[serde(default)]
    pub engram: Option<EngramSpec>,
}

impl Default for EngramSpec {
    fn default() -> Self {
        EngramSpec { default_pack: default_pack(), packs: BTreeMap::new(), format: "sources".into(), k: 3, ncand: default_ncand(), nprobe: default_nprobe() }
    }
}

fn default_ctx() -> u32 {
    32768
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Quant {
    pub name: String,
    pub file: String,
    pub bytes: u64,
    #[serde(default)]
    pub sha256: Option<String>,
    /// false = only on request (--quant), e.g. BF16 when Q8_0 has the same KLD at half the size
    #[serde(default = "yes")]
    pub auto: bool,
}

fn yes() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Mtp {
    pub file: String,
    pub bytes: u64,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default = "default_nmax")]
    pub n_max: u32,
}

fn default_nmax() -> u32 {
    2
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EngramSpec {
    /// pack from the terrabeam registry (lite / standard / full ...)
    #[serde(default = "default_pack")]
    pub default_pack: String,
    /// optional: this model's own table list per pack name (overrides the registry's list for that name)
    #[serde(default)]
    pub packs: BTreeMap<String, Pack>,
    /// injection layout the model was trained on ("tablinum") or generic source blocks ("sources")
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default = "default_ncand")]
    pub ncand: usize,
    #[serde(default = "default_nprobe")]
    pub nprobe: usize,
}

fn default_pack() -> String {
    "lite".into()
}
fn default_format() -> String {
    "tablinum".into()
}
fn default_k() -> usize {
    3
}
fn default_ncand() -> usize {
    48
}
fn default_nprobe() -> usize {
    128
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Pack {
    pub tables: Vec<String>,
    #[serde(default)]
    pub bytes: u64,
}

impl Manifest {
    pub fn quant(&self, name: &str) -> Option<&Quant> {
        self.quants.iter().find(|q| q.name.eq_ignore_ascii_case(name))
    }
}
