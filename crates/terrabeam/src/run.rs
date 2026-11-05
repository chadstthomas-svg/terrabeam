//! `terrabeam run <repo>`: manifest -> hardware -> quant -> llama.cpp release + GGUF (+ MTP draft) + engram pack -> llama-server on
//! an internal port -> OpenAI-compatible proxy on --port that rewrites the last user message with the model's training layout.

use crate::{fetch, hw, inject, manifest::Manifest};
use anyhow::{bail, Context, Result};
use axum::body::Body;
use axum::extract::{Query, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

// Pinned retrieval models (Apache-2.0); the same revisions the tables were built and gated with.
const EMBED_REPO: (&str, &str) = ("sentence-transformers/static-retrieval-mrl-en-v1", "f60985c706f192d45d218078e49e5a8b6f15283a");
const EMBED_FILES: &[&str] = &["0_StaticEmbedding/tokenizer.json", "0_StaticEmbedding/model.safetensors"];
const CE_REPO: (&str, &str) = ("cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a");
const CE_FILES: &[&str] = &["config.json", "tokenizer.json", "onnx/model.onnx"];

#[derive(clap::Parser, Clone, Debug)]
pub struct RunArgs {
    /// Hugging Face model repo with a tablinum.json (e.g. tablinum/Moon-Beam-8B-A1B-GGUF), or a local folder holding one
    pub model: String,
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    /// quant name from the manifest (default: the best one that fits this machine)
    #[arg(long)]
    pub quant: Option<String>,
    /// engram pack (default: the manifest's default_pack)
    #[arg(long)]
    pub pack: Option<String>,
    /// use an existing engram root (wiki at the top, other tables under tables/<name>) instead of downloading a pack
    #[arg(long)]
    pub engram_root: Option<PathBuf>,
    /// tables to load from --engram-root (comma list; default: the pack's tables)
    #[arg(long, value_delimiter = ',')]
    pub tables: Option<Vec<String>>,
    /// run without the engram library (plain llama-server behind the proxy)
    #[arg(long)]
    pub no_engram: bool,
    /// do not use the MTP draft head
    #[arg(long)]
    pub no_mtp: bool,
    /// ignore GPUs: CPU llama.cpp build, quant chosen for RAM
    #[arg(long)]
    pub cpu: bool,
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

struct Px {
    upstream: String,
    http: reqwest::Client,
    eg: Option<Arc<engram::Engram>>,
    tables: Vec<String>,
    format: inject::Format,
    k: usize,
    model: String,
}

pub async fn run(a: RunArgs) -> Result<()> {
    let c = fetch::client();
    let home = fetch::home();
    // 1. manifest
    let local = Path::new(&a.model).is_dir().then(|| PathBuf::from(&a.model));
    let mdir = local.clone().unwrap_or_else(|| home.join("models").join(a.model.replace('/', "--")));
    let mpath = mdir.join("tablinum.json");
    if local.is_none() {
        let url = fetch::hf_url(&a.model, false, &a.revision, "tablinum.json");
        std::fs::create_dir_all(&mdir)?;
        let r = c.get(&url).send().await?.error_for_status().with_context(|| format!("{} has no tablinum.json", a.model))?;
        std::fs::write(&mpath, r.bytes().await?)?;
    }
    let m: Manifest = serde_json::from_str(&std::fs::read_to_string(&mpath)?).with_context(|| format!("parse {}", mpath.display()))?;
    // 2. hardware + quant
    let mut h = hw::detect();
    if a.cpu {
        h.accel = hw::Accel::Cpu;
        h.gpus.clear();
    }
    eprintln!("terrabeam: {} | {}", m.name, h.describe());
    let q = match &a.quant {
        Some(n) => m.quant(n).with_context(|| format!("no quant {n} in the manifest"))?.clone(),
        None => pick_quant(&m, &h).clone(),
    };
    // 3. files
    let gguf = fetch_model_file(&c, &a, local.as_deref(), &mdir, &q.file, Some(q.bytes), q.sha256.as_deref()).await?;
    let gpu = matches!(h.accel, hw::Accel::Cuda { .. } | hw::Accel::Metal | hw::Accel::Vulkan);
    let mtp = match (&m.mtp, gpu && !a.no_mtp) {
        (Some(d), true) => Some((fetch_model_file(&c, &a, local.as_deref(), &mdir, &d.file, Some(d.bytes), d.sha256.as_deref()).await?, d.n_max)),
        _ => None,
    };
    let server = match &a.llama_server {
        Some(p) => p.clone(),
        None => ensure_llama(&c, &home, &h, &m.llama_cpp).await?,
    };
    // 4. engram
    let mut tables = vec![];
    let (mut format, mut k) = (inject::Format::Tablinum, 3);
    let eg = if a.no_engram || m.engram.is_none() && a.engram_root.is_none() {
        None
    } else {
        let spec = m.engram.clone();
        if let Some(s) = &spec {
            format = if s.format == "sources" { inject::Format::Sources } else { inject::Format::Tablinum };
            k = s.k;
        }
        let (root, ts) = match &a.engram_root {
            Some(r) => (r.clone(), a.tables.clone().or_else(|| spec.as_ref().and_then(|s| s.packs.get(&s.default_pack).map(|p| p.tables.clone()))).unwrap_or(vec!["wiki".into()])),
            None => {
                let s = spec.as_ref().unwrap();
                let pn = a.pack.clone().unwrap_or(s.default_pack.clone());
                let p = s.packs.get(&pn).with_context(|| format!("no engram pack {pn}"))?;
                (fetch_pack(&c, &home, &s.repo, &a.revision, &pn, &p.path).await?, a.tables.clone().unwrap_or(p.tables.clone()))
            }
        };
        let (emb, ce) = (fetch_aux(&c, &home, EMBED_REPO, EMBED_FILES).await?, fetch_aux(&c, &home, CE_REPO, CE_FILES).await?);
        let (ncand, nprobe) = spec.as_ref().map(|s| (s.ncand, s.nprobe)).unwrap_or((48, 128));
        let t0 = Instant::now();
        let ts2 = ts.clone();
        let e = tokio::task::spawn_blocking(move || -> Result<engram::Engram> {
            let names: Vec<&str> = ts2.iter().map(|s| s.as_str()).collect();
            let mut e = engram::Engram::open(&root, &names, &emb.join("0_StaticEmbedding"), nprobe, ncand)?;
            let threads = std::thread::available_parallelism().map(|n| n.get().clamp(2, 8)).unwrap_or(4);
            e.rerank = Some(engram::rerank::CrossEncoder::open_with(&ce, true, threads)?);
            e.confident_at = 0.85;
            Ok(e)
        })
        .await??;
        eprintln!("terrabeam: engram ready in {:.1}s ({} tables)", t0.elapsed().as_secs_f32(), ts.len());
        tables = ts;
        Some(Arc::new(e))
    };
    // 5. llama-server on an internal port
    let iport = free_port()?;
    let mut args: Vec<String> = vec![
        "-m".into(), gguf.to_string_lossy().into(), "--jinja".into(), "-c".into(), a.ctx.unwrap_or(m.context).to_string(),
        "--host".into(), "127.0.0.1".into(), "--port".into(), iport.to_string(), "--no-webui".into(), "--alias".into(), m.name.clone(),
    ];
    for (k_, v) in &m.sampling {
        let flag = match k_.as_str() { "temperature" => "--temp", "top_p" => "--top-p", "top_k" => "--top-k", "min_p" => "--min-p", _ => continue };
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
    // 6. proxy
    let px = Arc::new(Px { upstream, http: reqwest::Client::new(), eg, tables: tables.clone(), format, k, model: format!("{} ({})", m.name, q.name) });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/engram/search", get(search))
        .route("/health", get(health))
        .fallback(forward)
        .with_state(px.clone());
    let addr = format!("{}:{}", a.host, a.port);
    let l = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("bind {addr}"))?;
    eprintln!(
        "\nterrabeam: {} ready at http://{addr}/v1  (OpenAI-compatible; model {})\n  engram: {}\n  per request: \"engram\": false in the body or header X-Engram: off; debug: GET /engram/search?q=...\n",
        m.name, px.model,
        if px.eg.is_some() { format!("on, {} tables ({}), {} passages per question", tables.len(), tables.join(","), k) } else { "off".into() }
    );
    tokio::select! {
        r = axum::serve(l, app) => { r?; }
        _ = tokio::signal::ctrl_c() => { eprintln!("terrabeam: stopping"); }
        st = child.wait() => { bail!("llama-server exited: {:?}", st?); }
    }
    child.kill().await.ok();
    Ok(())
}

/// Best quant that fits: on a GPU the best one that fits in VRAM with ~1.5 GB for KV/compute, else Q4_K_M (hybrid/CPU);
/// on CPU Q4_K_M (decode speed tracks bytes) when RAM allows, else the largest that fits half the RAM.
fn pick_quant<'a>(m: &'a Manifest, h: &hw::Hw) -> &'a crate::manifest::Quant {
    let mib = |b: u64| b / (1024 * 1024);
    let q4 = m.quant("Q4_K_M");
    let in_ram = |b: u64| mib(b) * 2 <= h.ram_mib;
    if h.vram_mib() > 0 {
        let budget = (h.vram_mib() * 85 / 100).saturating_sub(1536);
        if let Some(q) = m.quants.iter().find(|q| mib(q.bytes) <= budget) {
            return q;
        }
    }
    if let Some(q) = q4.filter(|q| in_ram(q.bytes)) {
        return q;
    }
    m.quants.iter().find(|q| in_ram(q.bytes)).unwrap_or_else(|| m.quants.last().expect("manifest has no quants"))
}

async fn fetch_model_file(c: &reqwest::Client, a: &RunArgs, local: Option<&Path>, mdir: &Path, file: &str, size: Option<u64>, sha: Option<&str>) -> Result<PathBuf> {
    if let Some(d) = local {
        let p = d.join(file);
        anyhow::ensure!(p.exists(), "{} missing", p.display());
        return Ok(p);
    }
    fetch::download(c, &fetch::hf_url(&a.model, false, &a.revision, file), &mdir.join(file), size, sha).await
}

async fn ensure_llama(c: &reqwest::Client, home: &Path, h: &hw::Hw, tag: &str) -> Result<PathBuf> {
    let assets = hw::llama_assets(h, tag);
    let dir = home.join("llama.cpp").join(assets[0].trim_end_matches(".tar.gz").trim_end_matches(".zip"));
    let exe = dir.join(if h.os == "windows" { "llama-server.exe" } else { "llama-server" });
    if exe.exists() {
        return Ok(exe);
    }
    eprintln!("terrabeam: fetching llama.cpp {tag} ({})", assets.join(" + "));
    // digests from the release API when GitHub provides them
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

async fn fetch_pack(c: &reqwest::Client, home: &Path, repo: &str, rev: &str, name: &str, path: &str) -> Result<PathBuf> {
    let root = home.join("engram").join(repo.replace('/', "--")).join(name);
    let files = fetch::hf_tree(c, repo, true, rev, path).await?;
    let total: u64 = files.iter().map(|f| f.size).sum();
    eprintln!("terrabeam: engram pack {name}: {} files, {:.1} GB", files.len(), total as f64 / 1e9);
    for f in &files {
        let rel = f.path.strip_prefix(path).unwrap_or(&f.path).trim_start_matches('/');
        fetch::download(c, &fetch::hf_url(repo, true, rev, &f.path), &root.join(rel), Some(f.size), f.sha256.as_deref()).await?;
    }
    Ok(root)
}

async fn fetch_aux(c: &reqwest::Client, home: &Path, (repo, rev): (&str, &str), files: &[&str]) -> Result<PathBuf> {
    let dir = home.join("aux").join(repo.replace('/', "--")).join(rev);
    for f in files {
        fetch::download(c, &fetch::hf_url(repo, false, rev, f), &dir.join(f), None, None).await?;
    }
    Ok(dir)
}

fn free_port() -> Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

async fn health(State(px): State<Arc<Px>>) -> Json<Value> {
    Json(json!({"status": "ok", "model": px.model, "engram": px.eg.is_some(), "tables": px.tables}))
}

async fn search(State(px): State<Arc<Px>>, Query(p): Query<HashMap<String, String>>) -> Response {
    let Some(eg) = px.eg.clone() else { return (StatusCode::NOT_FOUND, "engram is off").into_response() };
    let q = p.get("q").cloned().unwrap_or_default();
    let tables = px.tables.clone();
    let k = p.get("k").and_then(|k| k.parse().ok()).unwrap_or(px.k);
    match tokio::task::spawn_blocking(move || eg.retrieve(&q, k, &tables.iter().map(|s| s.as_str()).collect::<Vec<_>>())).await {
        Ok(Ok(h)) => Json(json!({"hits": h.iter().map(|h| json!({"table": h.table, "pid": h.pid, "title": h.passage.title, "score": h.score, "text": h.passage.text})).collect::<Vec<_>>()})).into_response(),
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// POST /v1/chat/completions: rewrite the last user message (unless off for this request), then forward as-is (streams too).
async fn chat(State(px): State<Arc<Px>>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    let Ok(mut v) = serde_json::from_slice::<Value>(&body) else { return send(&px, "/v1/chat/completions", &headers, body.to_vec(), None).await };
    let flag = v.as_object_mut().and_then(|o| o.remove("engram"));
    let hdr = headers.get("x-engram").and_then(|h| h.to_str().ok()).unwrap_or("").to_ascii_lowercase();
    let on = px.eg.is_some() && !matches!(flag, Some(Value::Bool(false))) && !matches!(hdr.as_str(), "off" | "0" | "false" | "no");
    let mut used = None;
    if on {
        if let Some(msgs) = v.get_mut("messages").and_then(|m| m.as_array_mut()) {
            if let Some((i, q)) = inject::last_user(msgs) {
                let (eg, tables, k) = (px.eg.clone().unwrap(), px.tables.clone(), px.k);
                let q2 = q.clone();
                let job = tokio::task::spawn_blocking(move || eg.retrieve(&q2, k, &tables.iter().map(|s| s.as_str()).collect::<Vec<_>>()));
                if let Ok(Ok(Ok(hits))) = tokio::time::timeout(Duration::from_secs(8), job).await {
                    if !hits.is_empty() {
                        let srcs: Vec<inject::Src> = hits.iter().map(|h| inject::Src { table: &h.table, title: &h.passage.title, text: &h.passage.text }).collect();
                        inject::replace_text(msgs, i, inject::render(px.format, &q, &srcs, &inject::new_nonce()));
                        used = Some(hits.iter().map(|h| format!("{}:{}", h.table, h.pid)).collect::<Vec<_>>().join(","));
                    }
                }
            }
        }
    }
    send(&px, "/v1/chat/completions", &headers, serde_json::to_vec(&v).unwrap_or(body.to_vec()), used).await
}

async fn forward(State(px): State<Arc<Px>>, req: Request) -> Response {
    let path = req.uri().path_and_query().map(|p| p.as_str().to_string()).unwrap_or("/".into());
    let (method, headers) = (req.method().clone(), req.headers().clone());
    let body = axum::body::to_bytes(req.into_body(), 64 << 20).await.unwrap_or_default();
    match px.http.request(method, format!("{}{path}", px.upstream)).headers(clean(&headers)).body(body.to_vec()).send().await {
        Ok(r) => relay(r, None),
        Err(e) => (StatusCode::BAD_GATEWAY, format!("llama-server: {e}")).into_response(),
    }
}

async fn send(px: &Px, path: &str, headers: &HeaderMap, body: Vec<u8>, used: Option<String>) -> Response {
    let mut h = clean(headers);
    h.insert("content-type", HeaderValue::from_static("application/json"));
    match px.http.post(format!("{}{path}", px.upstream)).headers(h).body(body).send().await {
        Ok(r) => relay(r, used),
        Err(e) => (StatusCode::BAD_GATEWAY, format!("llama-server: {e}")).into_response(),
    }
}

fn clean(h: &HeaderMap) -> HeaderMap {
    let mut out = h.clone();
    for k in ["host", "content-length", "x-engram", "connection", "transfer-encoding"] {
        out.remove(k);
    }
    out
}

fn relay(r: reqwest::Response, used: Option<String>) -> Response {
    let status = StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut b = Response::builder().status(status);
    for (k, v) in r.headers() {
        if !matches!(k.as_str(), "content-length" | "transfer-encoding" | "connection") {
            b = b.header(k, v);
        }
    }
    if let Some(u) = used.and_then(|u| HeaderValue::from_str(&u).ok()) {
        b = b.header("x-engram-sources", u);
    }
    b.body(Body::from_stream(r.bytes_stream())).unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}
