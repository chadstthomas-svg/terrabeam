//! `terrabeam run <model>` and `terrabeam proxy --upstream <url>`.
//!
//! run: a Tablinum repo (tablinum.json), ANY Hugging Face repo with GGUF files, or a local .gguf -> quant for this machine ->
//! pinned upstream llama.cpp + the GGUF (+ MTP draft) -> llama-server on an internal port -> the proxy (engram + memory) on
//! --port. proxy: the same proxy in front of a server you already run (Ollama, LM Studio, vLLM, llama.cpp, a cloud API).

use crate::{fetch, hw, inject, manifest::{EngramSpec, Manifest, Quant}, memory, packs, proxy};
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

// Pinned retrieval models (Apache-2.0); the same revisions the tables were built and gated with.
const EMBED_REPO: (&str, &str) = ("sentence-transformers/static-retrieval-mrl-en-v1", "f60985c706f192d45d218078e49e5a8b6f15283a");
const EMBED_FILES: &[&str] = &["0_StaticEmbedding/tokenizer.json", "0_StaticEmbedding/model.safetensors"];
const CE_REPO: (&str, &str) = ("cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a");
const CE_FILES: &[&str] = &["config.json", "tokenizer.json", "onnx/model.onnx"];
const DEFAULT_LLAMA_TAG: &str = "b12078";

/// Engram + memory options shared by run, proxy and mcp.
#[derive(clap::Args, Clone, Debug)]
pub struct LibArgs {
    /// engram pack: lite (~5 GB), standard (~21 GB), or any pack in the registry
    #[arg(long)]
    pub pack: Option<String>,
    /// use an existing engram root (wiki at the top, other tables under tables/<name>) instead of downloading a pack
    #[arg(long)]
    pub engram_root: Option<PathBuf>,
    /// tables to use (comma list; default: the pack's tables)
    #[arg(long, value_delimiter = ',')]
    pub tables: Option<Vec<String>>,
    /// no knowledge library
    #[arg(long)]
    pub no_engram: bool,
    /// how passages are added: "sources" (any model) or "tablinum" (models trained on it; their manifest sets this)
    #[arg(long)]
    pub format: Option<String>,
    /// passages per question
    #[arg(long)]
    pub k: Option<usize>,
    /// persistent memory: off, session (this conversation) or long-term (every conversation of the user)
    #[arg(long, value_enum, default_value = "off")]
    pub memory: memory::Mode,
}

#[derive(clap::Args, Clone, Debug)]
pub struct RunArgs {
    /// tablinum/Moon-Beam-8B-A1B-GGUF-Preview, any HF repo with .gguf files (e.g. unsloth/Qwen3-8B-GGUF), a local folder with a
    /// tablinum.json, or a local .gguf file
    pub model: String,
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    /// quant to use (default: the best one that fits this machine)
    #[arg(long)]
    pub quant: Option<String>,
    #[command(flatten)]
    pub lib: LibArgs,
    /// do not use the MTP draft head
    #[arg(long)]
    pub no_mtp: bool,
    /// ignore GPUs: CPU llama.cpp build, quant chosen for RAM
    #[arg(long)]
    pub cpu: bool,
    /// show what would be downloaded and run, then exit
    #[arg(long)]
    pub dry_run: bool,
    #[arg(long)]
    pub ctx: Option<u32>,
    #[arg(long, default_value = "main")]
    pub revision: String,
    /// use this llama-server binary instead of the pinned release
    #[arg(long, env = "TERRABEAM_LLAMA_SERVER")]
    pub llama_server: Option<PathBuf>,
    /// extra llama-server arguments (after --)
    #[arg(last = true)]
    pub extra: Vec<String>,
}

#[derive(clap::Args, Clone, Debug)]
pub struct ProxyArgs {
    /// the OpenAI-compatible server to put the engram in front of, e.g. http://127.0.0.1:11434 (Ollama),
    /// http://127.0.0.1:1234 (LM Studio), http://127.0.0.1:8000 (vLLM); with or without /v1
    #[arg(long)]
    pub upstream: String,
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    #[arg(long, default_value_t = 8090)]
    pub port: u16,
    #[command(flatten)]
    pub lib: LibArgs,
}

/// Load the engram for these options (download the pack if needed). None when off.
pub async fn load_library(c: &reqwest::Client, home: &Path, lib: &LibArgs, spec: &EngramSpec) -> Result<Option<(Arc<engram::Engram>, Vec<String>)>> {
    if lib.no_engram {
        return Ok(None);
    }
    let (root, tables) = match &lib.engram_root {
        Some(r) => (r.clone(), lib.tables.clone().unwrap_or_else(|| vec!["wiki".into()])),
        None => {
            let reg = packs::registry(c).await;
            let pn = lib.pack.clone().unwrap_or(spec.default_pack.clone());
            match (lib.tables.clone(), spec.packs.get(&pn)) {
                (Some(ts), _) => {
                    for t in &ts {
                        packs::ensure_table(c, home, &reg, t).await?;
                    }
                    (packs::engram_root(home), ts)
                }
                (None, Some(own)) => {
                    for t in &own.tables {
                        packs::ensure_table(c, home, &reg, t).await?;
                    }
                    (packs::engram_root(home), own.tables.clone())
                }
                (None, None) => packs::ensure_pack(c, home, &reg, &pn).await?,
            }
        }
    };
    let emb = fetch_aux(c, home, EMBED_REPO, EMBED_FILES).await?;
    let ce = fetch_aux(c, home, CE_REPO, CE_FILES).await?;
    let (ncand, nprobe) = (spec.ncand, spec.nprobe);
    let t0 = Instant::now();
    let ts = tables.clone();
    let e = tokio::task::spawn_blocking(move || -> Result<engram::Engram> {
        let names: Vec<&str> = ts.iter().map(|s| s.as_str()).collect();
        let mut e = engram::Engram::open(&root, &names, &emb.join("0_StaticEmbedding"), nprobe, ncand)?;
        let threads = std::thread::available_parallelism().map(|n| n.get().clamp(2, 8)).unwrap_or(4);
        e.rerank = Some(engram::rerank::CrossEncoder::open_with(&ce, true, threads)?);
        e.confident_at = 0.85;
        Ok(e)
    })
    .await??;
    eprintln!("terrabeam: engram ready in {:.1}s ({} tables)", t0.elapsed().as_secs_f32(), tables.len());
    Ok(Some((Arc::new(e), tables)))
}

fn fmt_of(lib: &LibArgs, spec: &EngramSpec) -> (inject::Format, usize) {
    let f = lib.format.clone().unwrap_or(spec.format.clone());
    (if f == "tablinum" { inject::Format::Tablinum } else { inject::Format::Sources }, lib.k.unwrap_or(spec.k))
}

pub async fn proxy_cmd(a: ProxyArgs) -> Result<()> {
    let c = fetch::client();
    let home = fetch::home();
    let up = a.upstream.trim_end_matches('/').trim_end_matches("/v1").to_string();
    let spec = EngramSpec::default();
    let lib = load_library(&c, &home, &a.lib, &spec).await?;
    let (format, k) = fmt_of(&a.lib, &spec);
    let mem = Arc::new(memory::Memory::open(&home, a.lib.memory));
    let px = Arc::new(proxy::Px {
        upstream: up.clone(), http: reqwest::Client::new(), eg: lib.as_ref().map(|l| l.0.clone()), tables: lib.as_ref().map(|l| l.1.clone()).unwrap_or_default(),
        format, k, model: format!("proxy -> {up}"), mem, mem_k: 5,
    });
    serve_px(px, &a.host, a.port, None).await
}

async fn serve_px(px: Arc<proxy::Px>, host: &str, port: u16, child: Option<&mut tokio::process::Child>) -> Result<()> {
    let addr = format!("{host}:{port}");
    let l = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("bind {addr}"))?;
    eprintln!(
        "\nterrabeam: ready at http://{addr}/v1  (OpenAI-compatible; {})\n  engram: {}\n  memory: {:?}\n  per request: \"engram\": false or header X-Engram: off; \"memory\": false or X-Memory: off; debug: GET /engram/search?q=...  GET /memory\n",
        px.model,
        if px.eg.is_some() { format!("on, {} tables ({}), {} passages per question, {:?} format", px.tables.len(), px.tables.join(","), px.k, px.format) } else { "off".into() },
        px.mem.mode
    );
    let app = proxy::router(px);
    match child {
        Some(ch) => tokio::select! {
            r = axum::serve(l, app) => r.map_err(anyhow::Error::from),
            _ = shutdown_signal() => { eprintln!("terrabeam: stopping"); Ok(()) }
            st = ch.wait() => Err(anyhow::anyhow!("llama-server exited: {:?}", st)),
        },
        None => tokio::select! {
            r = axum::serve(l, app) => r.map_err(anyhow::Error::from),
            _ = shutdown_signal() => { eprintln!("terrabeam: stopping"); Ok(()) }
        },
    }
}

pub async fn run(a: RunArgs) -> Result<()> {
    let c = fetch::client();
    let home = fetch::home();
    let (m, files) = resolve_model(&c, &home, &a).await?;
    let mut h = hw::detect();
    if a.cpu {
        h.accel = hw::Accel::Cpu;
        h.gpus.clear();
    }
    eprintln!("terrabeam: {} | {}", m.name, h.describe());
    let q = match &a.quant {
        Some(n) => m.quant(n).with_context(|| format!("no quant {n} (have: {})", m.quants.iter().map(|q| q.name.as_str()).collect::<Vec<_>>().join(", ")))?.clone(),
        None => pick_quant(&m, &h).clone(),
    };
    if a.dry_run {
        let spec = m.engram.clone().unwrap_or_default();
        let (format, k) = fmt_of(&a.lib, &spec);
        println!("model:    {}\nquant:    {} ({}, {:.2} GB){}\nquants:   {}\nllama.cpp {} ({})\nengram:   {}\nmemory:   {:?}",
            m.name, q.name, q.file, q.bytes as f64 / 1e9,
            if m.mtp.is_some() && !a.no_mtp && !matches!(h.accel, hw::Accel::Cpu) { " + MTP draft" } else { "" },
            m.quants.iter().map(|x| format!("{}{}", x.name, if x.auto { "" } else { "*" })).collect::<Vec<_>>().join(", "),
            if m.llama_cpp.is_empty() { DEFAULT_LLAMA_TAG } else { &m.llama_cpp }, hw::llama_assets(&h, if m.llama_cpp.is_empty() { DEFAULT_LLAMA_TAG } else { &m.llama_cpp }).join(" + "),
            if a.lib.no_engram { "off".to_string() } else { format!("pack {} , {:?} format, k {}", a.lib.pack.clone().unwrap_or(spec.default_pack.clone()), format, k) },
            a.lib.memory);
        return Ok(());
    }
    let gguf = files.get(&q, &c).await?;
    let gpu = matches!(h.accel, hw::Accel::Cuda { .. } | hw::Accel::Metal | hw::Accel::Vulkan);
    let mtp = match (&m.mtp, gpu && !a.no_mtp) {
        (Some(d), true) => Some((files.get_file(&d.file, Some(d.bytes), d.sha256.as_deref(), &c).await?, d.n_max)),
        _ => None,
    };
    let server = match &a.llama_server {
        Some(p) => p.clone(),
        None => ensure_llama(&c, &home, &h, if m.llama_cpp.is_empty() { DEFAULT_LLAMA_TAG } else { &m.llama_cpp }).await?,
    };
    let spec = m.engram.clone().unwrap_or_default();
    let lib = load_library(&c, &home, &a.lib, &spec).await?;
    let (format, k) = fmt_of(&a.lib, &spec);
    // llama-server on an internal port
    let iport = free_port()?;
    let mut args: Vec<String> = vec![
        "-m".into(), gguf.to_string_lossy().into(), "--jinja".into(), "-c".into(), a.ctx.unwrap_or(m.context).to_string(),
        "--host".into(), "127.0.0.1".into(), "--port".into(), iport.to_string(), "--no-webui".into(), "--alias".into(), m.name.clone(),
    ];
    // the model's own sampling defaults; min_p off unless the manifest sets it (llama.cpp's 0.05 is not part of model cards)
    let mut samp = m.sampling.clone();
    samp.entry("min_p".into()).or_insert(0.0);
    for (k_, v) in &samp {
        let flag = match k_.as_str() { "temperature" => "--temp", "top_p" => "--top-p", "top_k" => "--top-k", "min_p" => "--min-p", "presence_penalty" => "--presence-penalty", "repeat_penalty" => "--repeat-penalty", _ => continue };
        args.extend([flag.to_string(), v.to_string()]);
    }
    if let Some((d, n)) = &mtp {
        args.extend(["-md".into(), d.to_string_lossy().into(), "-ngld".into(), "99".into(), "--spec-type".into(), "draft-mtp".into(),
            "--spec-draft-n-max".into(), n.to_string()]);
    }
    args.extend(a.extra.iter().cloned());
    let logf = home.join("logs");
    std::fs::create_dir_all(&logf)?;
    let log = std::fs::File::create(logf.join("llama-server.log"))?;
    let bindir = server.parent().unwrap_or(Path::new(".")).to_path_buf();
    let mut cmd = tokio::process::Command::new(&server);
    cmd.args(&args).stdout(log.try_clone()?).stderr(log).kill_on_drop(true);
    if h.os == "linux" {
        let lp = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        cmd.env("LD_LIBRARY_PATH", format!("{}{}{lp}", bindir.display(), if lp.is_empty() { "" } else { ":" }));
    }
    let mut child = cmd.spawn().with_context(|| format!("start {}", server.display()))?;
    eprintln!("terrabeam: starting llama-server ({}{}), log {}", q.name, if mtp.is_some() { " + MTP draft" } else { "" }, logf.join("llama-server.log").display());
    let upstream = format!("http://127.0.0.1:{iport}");
    let t0 = Instant::now();
    loop {
        if let Some(st) = child.try_wait()? {
            bail!("llama-server exited ({st}) while loading; see {}", logf.join("llama-server.log").display());
        }
        if c.get(format!("{upstream}/health")).timeout(Duration::from_secs(2)).send().await.map(|r| r.status().is_success()).unwrap_or(false) {
            break;
        }
        if t0.elapsed() > Duration::from_secs(1800) {
            bail!("llama-server did not become ready in 30 min");
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    let mem = Arc::new(memory::Memory::open(&home, a.lib.memory));
    let px = Arc::new(proxy::Px {
        upstream, http: reqwest::Client::new(), eg: lib.as_ref().map(|l| l.0.clone()), tables: lib.as_ref().map(|l| l.1.clone()).unwrap_or_default(),
        format, k, model: format!("{} ({})", m.name, q.name), mem, mem_k: 5,
    });
    let res = serve_px(px, &a.host, a.port, Some(&mut child)).await;
    child.kill().await.ok(); // also on SIGTERM: a stopped runner must not leave its llama-server behind
    res
}

/// Where the model's files come from: a local folder/file or an HF repo.
enum Files {
    Local(PathBuf),
    Hf { repo: String, revision: String, dir: PathBuf },
}

impl Files {
    async fn get(&self, q: &Quant, c: &reqwest::Client) -> Result<PathBuf> {
        // split GGUFs (name-00001-of-0000N.gguf): fetch every part, return the first
        let parts = split_parts(&q.file);
        let single = parts.len() == 1;
        let mut first = None;
        for (i, p) in parts.iter().enumerate() {
            let path = self.get_file(p, if single { Some(q.bytes) } else { None }, if single { q.sha256.as_deref() } else { None }, c).await?;
            if i == 0 {
                first = Some(path);
            }
        }
        Ok(first.unwrap())
    }
    async fn get_file(&self, file: &str, size: Option<u64>, sha: Option<&str>, c: &reqwest::Client) -> Result<PathBuf> {
        match self {
            Files::Local(d) => {
                let p = if d.is_file() { d.clone() } else { d.join(file) };
                anyhow::ensure!(p.exists(), "{} missing", p.display());
                Ok(p)
            }
            Files::Hf { repo, revision, dir } => fetch::download(c, &fetch::hf_url(repo, false, revision, file), &dir.join(file), size, sha).await,
        }
    }
}

/// "...-00001-of-00003.gguf" -> every part; anything else -> itself
fn split_parts(file: &str) -> Vec<String> {
    if let Some(i) = file.rfind("-of-") {
        let (head, tail) = file.split_at(i);
        let n = head.rsplit('-').next().and_then(|x| x.parse::<u32>().ok());
        let total = tail[4..].strip_suffix(".gguf").and_then(|x| x.parse::<u32>().ok());
        if let (Some(1), Some(total)) = (n, total) {
            if total > 1 && head.len() >= 5 {
                let base = &head[..head.len() - 5];
                return (1..=total).map(|k| format!("{base}{k:05}-of-{total:05}.gguf")).collect();
            }
        }
    }
    vec![file.to_string()]
}

/// tablinum.json when the model has one; otherwise a manifest built from the .gguf files (any model on Hugging Face).
async fn resolve_model(c: &reqwest::Client, home: &Path, a: &RunArgs) -> Result<(Manifest, Files)> {
    let p = Path::new(&a.model);
    if p.is_file() && a.model.ends_with(".gguf") {
        let name = p.file_stem().unwrap().to_string_lossy().to_string();
        let q = Quant { name: quant_of(&name).unwrap_or("local".into()), file: p.file_name().unwrap().to_string_lossy().into(), bytes: std::fs::metadata(p)?.len(), sha256: None, auto: true };
        return Ok((generic_manifest(&name, vec![q]), Files::Local(p.to_path_buf())));
    }
    if p.is_dir() {
        let m: Manifest = serde_json::from_str(&std::fs::read_to_string(p.join("tablinum.json")).context("tablinum.json")?)?;
        return Ok((m, Files::Local(p.to_path_buf())));
    }
    let dir = home.join("models").join(a.model.replace('/', "--"));
    std::fs::create_dir_all(&dir)?;
    let files = Files::Hf { repo: a.model.clone(), revision: a.revision.clone(), dir: dir.clone() };
    let url = fetch::hf_url(&a.model, false, &a.revision, "tablinum.json");
    if let Ok(r) = c.get(&url).send().await.and_then(|r| r.error_for_status()) {
        let b = r.bytes().await?;
        std::fs::write(dir.join("tablinum.json"), &b)?;
        return Ok((serde_json::from_slice(&b).context("parse tablinum.json")?, files));
    }
    // any GGUF repo
    let tree = fetch::hf_tree(c, &a.model, false, &a.revision, "").await.with_context(|| format!("{} not found on Hugging Face", a.model))?;
    let mut qs: Vec<Quant> = vec![];
    for f in tree.iter().filter(|f| f.path.ends_with(".gguf") && !f.path.to_lowercase().contains("mmproj")) {
        let parts = split_parts(&f.path);
        if f.path.contains("-of-") && parts[0] != f.path {
            continue; // parts 2..N are fetched with part 1
        }
        let bytes: u64 = if parts.len() > 1 { tree.iter().filter(|g| parts.contains(&g.path)).map(|g| g.size).sum() } else { f.size };
        let stem = f.path.rsplit('/').next().unwrap_or(&f.path).trim_end_matches(".gguf");
        let name = quant_of(stem).unwrap_or_else(|| stem.to_string());
        let auto = !matches!(name.as_str(), "BF16" | "F16" | "F32");
        qs.push(Quant { name, file: f.path.clone(), bytes, sha256: if parts.len() > 1 { None } else { f.sha256.clone() }, auto });
    }
    anyhow::ensure!(!qs.is_empty(), "{} has no tablinum.json and no .gguf files", a.model);
    qs.sort_by(|x, y| y.bytes.cmp(&x.bytes)); // biggest (best) first
    let name = a.model.rsplit('/').next().unwrap_or(&a.model).trim_end_matches("-GGUF").trim_end_matches("-gguf").to_string();
    Ok((generic_manifest(&name, qs), files))
}

fn generic_manifest(name: &str, quants: Vec<Quant>) -> Manifest {
    Manifest {
        schema: 1, name: name.to_string(), llama_cpp: DEFAULT_LLAMA_TAG.into(), quants, mtp: None, context: 32768,
        sampling: Default::default(), // llama-server uses the model's own recommended sampling from the GGUF when present
        engram: Some(EngramSpec::default()),
    }
}

fn quant_of(stem: &str) -> Option<String> {
    let up = stem.to_uppercase();
    let pats = ["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL",
        "Q2_K_L", "Q2_K_XL", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q3_K_XL", "Q4_K_S", "Q4_K_M", "Q4_K_L", "Q4_K_XL", "Q4_0", "Q4_1",
        "Q5_K_S", "Q5_K_M", "Q5_K_L", "Q5_K_XL", "Q5_0", "Q6_K_L", "Q6_K_XL", "Q6_K", "Q8_K_XL", "Q8_0", "MXFP4", "BF16", "F16", "F32"];
    let mut best: Option<&str> = None;
    for p in pats {
        if up.contains(p) && best.is_none_or(|b| p.len() > b.len()) {
            best = Some(p);
        }
    }
    best.map(|b| if up.contains(&format!("UD-{b}")) { format!("UD-{b}") } else { b.to_string() })
}

/// Best quant that fits: on a GPU the best one that fits in VRAM with ~1.5 GB for KV/compute, else Q4_K_M (hybrid/CPU);
/// on CPU Q4_K_M (decode speed tracks bytes) when RAM allows, else the largest that fits half the RAM.
fn pick_quant<'a>(m: &'a Manifest, h: &hw::Hw) -> &'a Quant {
    let mib = |b: u64| b / (1024 * 1024);
    let q4 = m.quants.iter().find(|q| q.name.ends_with("Q4_K_M"));
    let in_ram = |b: u64| mib(b) * 2 <= h.ram_mib;
    if h.vram_mib() > 0 {
        let budget = (h.vram_mib() * 85 / 100).saturating_sub(1536);
        if let Some(q) = m.quants.iter().filter(|q| q.auto).find(|q| mib(q.bytes) <= budget) {
            return q;
        }
    }
    if let Some(q) = q4.filter(|q| in_ram(q.bytes)) {
        return q;
    }
    m.quants.iter().filter(|q| q.auto).find(|q| in_ram(q.bytes)).unwrap_or_else(|| m.quants.last().expect("no quants"))
}

async fn ensure_llama(c: &reqwest::Client, home: &Path, h: &hw::Hw, tag: &str) -> Result<PathBuf> {
    let assets = hw::llama_assets(h, tag);
    let dir = home.join("llama.cpp").join(assets[0].trim_end_matches(".tar.gz").trim_end_matches(".zip"));
    let exe = dir.join(if h.os == "windows" { "llama-server.exe" } else { "llama-server" });
    if exe.exists() {
        return Ok(exe);
    }
    eprintln!("terrabeam: fetching llama.cpp {tag} ({})", assets.join(" + "));
    let api = format!("https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}");
    let digests: HashMap<String, (u64, Option<String>)> = match c.get(&api).send().await.and_then(|r| r.error_for_status()) {
        Ok(r) => r.json::<Value>().await.ok().and_then(|v| v["assets"].as_array().cloned()).unwrap_or_default().iter()
            .map(|x| (x["name"].as_str().unwrap_or_default().to_string(), (x["size"].as_u64().unwrap_or(0), x["digest"].as_str().map(String::from))))
            .collect(),
        Err(_) => HashMap::new(),
    };
    for asset in &assets {
        let url = format!("https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}");
        let (size, dig) = digests.get(asset).cloned().map(|(s, d)| (Some(s).filter(|s| *s > 0), d)).unwrap_or((None, None));
        let arc = fetch::download(c, &url, &home.join("downloads").join(asset), size, dig.as_deref()).await?;
        fetch::extract(&arc, &dir)?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for e in std::fs::read_dir(&dir)? {
            let p = e?.path();
            if p.file_name().and_then(|f| f.to_str()).is_some_and(|f| f.starts_with("llama-")) {
                std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755))?;
            }
        }
    }
    anyhow::ensure!(exe.exists(), "llama-server not found in {}", dir.display());
    Ok(exe)
}

pub async fn fetch_aux(c: &reqwest::Client, home: &Path, (repo, rev): (&str, &str), files: &[&str]) -> Result<PathBuf> {
    let dir = home.join("aux").join(repo.replace('/', "--")).join(rev);
    for f in files {
        fetch::download(c, &fetch::hf_url(repo, false, rev, f), &dir.join(f), None, None).await?;
    }
    Ok(dir)
}

fn free_port() -> Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

pub async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM handler");
        tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = term.recv() => {} }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await.ok();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_quant_names_from_file_names() {
        assert_eq!(quant_of("Qwen3-8B-Q4_K_M").as_deref(), Some("Q4_K_M"));
        assert_eq!(quant_of("gemma-4-26b-UD-Q4_K_XL").as_deref(), Some("UD-Q4_K_XL"));
        assert_eq!(quant_of("Moon-Beam-8B-A1B-IQ2_XXS").as_deref(), Some("IQ2_XXS"));
        assert_eq!(quant_of("model-bf16").as_deref(), Some("BF16"));
    }

    #[test]
    fn expands_split_gguf_parts() {
        assert_eq!(
            split_parts("Q4/m-Q4_K_M-00001-of-00003.gguf"),
            vec!["Q4/m-Q4_K_M-00001-of-00003.gguf", "Q4/m-Q4_K_M-00002-of-00003.gguf", "Q4/m-Q4_K_M-00003-of-00003.gguf"]
        );
        assert_eq!(split_parts("m-Q8_0.gguf"), vec!["m-Q8_0.gguf"]);
    }
}
