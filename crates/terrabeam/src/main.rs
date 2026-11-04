//! terrabeam - one-command runner + engram service for Beam models. See DESIGN.md.
//! Implemented so far: `serve` (engram service; drop-in for Rig's router/engram_serve.py + POST /inject).
mod inject;

use anyhow::{Context, Result};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "terrabeam", version, about = "Run Beam models with their engram library attached")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Engram retrieval service: GET /health, GET /search?q=&k=&tables=, POST /inject
    Serve(ServeArgs),
}

#[derive(Parser, Clone)]
struct ServeArgs {
    /// engram root (wiki lives here, other tables under tables/<name>)
    #[arg(long, env = "ENGRAM_ROOT", default_value = "/mnt/optane/engram")]
    root: PathBuf,
    /// tables to load (comma list)
    #[arg(long, env = "TABLES", default_value = "wiki", value_delimiter = ',')]
    tables: Vec<String>,
    #[arg(long, env = "ENGRAM_HOST", default_value = "127.0.0.1")]
    host: String,
    #[arg(long, env = "ENGRAM_PORT", default_value_t = 18395)]
    port: u16,
    #[arg(long, env = "NPROBE", default_value_t = 128)]
    nprobe: usize,
    /// first-stage candidates shared across the queried tables (re-ranked)
    #[arg(long, env = "NCAND", default_value_t = 24)]
    ncand: usize,
    /// cross-encoder threads
    #[arg(long, env = "THREADS", default_value_t = 8)]
    threads: usize,
    /// skip the re-rank when the top vector hit's inner product is at least this (0 = always re-rank)
    #[arg(long, env = "CONFIDENT_AT", default_value_t = 0.85)]
    confident_at: f32,
    /// static-retrieval-mrl-en-v1 0_StaticEmbedding folder
    #[arg(long, env = "EMBED_DIR")]
    embed_dir: PathBuf,
    /// ms-marco-MiniLM-L6-v2 snapshot folder (onnx/model.onnx -> onnxruntime, else candle)
    #[arg(long, env = "CE_DIR")]
    ce_dir: PathBuf,
    /// /search text cap in characters (engram_serve.py TEXT_CAP); /inject always uses full passages
    #[arg(long, default_value_t = 1400)]
    text_cap: usize,
}

struct AppState {
    eg: Mutex<Option<Arc<engram::Engram>>>,
    error: Mutex<Option<String>>,
    tables: Vec<String>,
    warmed: Mutex<Vec<String>>,
    queries: AtomicU64,
    hits: AtomicU64,
    fast: AtomicU64,
    text_cap: usize,
    backend: Mutex<String>,
}

type S = Arc<AppState>;

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(json!({"error": msg.into()}))).into_response()
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Serve(a) => serve(a).await,
    }
}

async fn serve(a: ServeArgs) -> Result<()> {
    let st: S = Arc::new(AppState {
        eg: Mutex::new(None),
        error: Mutex::new(None),
        tables: a.tables.clone(),
        warmed: Mutex::new(vec![]),
        queries: AtomicU64::new(0),
        hits: AtomicU64::new(0),
        fast: AtomicU64::new(0),
        text_cap: a.text_cap,
        backend: Mutex::new(String::new()),
    });
    // load in the background so /health answers (ready: false) during startup, as the Python sidecar
    let (st2, a2) = (st.clone(), a.clone());
    std::thread::spawn(move || {
        let t0 = Instant::now();
        let r = (|| -> Result<engram::Engram> {
            let names: Vec<&str> = a2.tables.iter().map(|s| s.as_str()).collect();
            let mut eg = engram::Engram::open(&a2.root, &names, &a2.embed_dir, a2.nprobe, a2.ncand)?;
            eg.rerank = Some(engram::rerank::CrossEncoder::open_with(&a2.ce_dir, true, a2.threads)?);
            eg.confident_at = a2.confident_at;
            Ok(eg)
        })();
        match r {
            Ok(eg) => {
                *st2.backend.lock().unwrap() = eg.rerank.as_ref().map(|c| c.backend_name().to_string()).unwrap_or_default();
                let eg = Arc::new(eg);
                *st2.eg.lock().unwrap() = Some(eg.clone());
                eprintln!("engram ready in {:?}: tables={:?}", t0.elapsed(), a2.tables);
                for t in &a2.tables {
                    let t1 = Instant::now();
                    match eg.retrieve("warmup", 1, &[t.as_str()]) {
                        Ok(_) => {
                            st2.warmed.lock().unwrap().push(t.clone());
                            eprintln!("warmed {t} in {:?}", t1.elapsed());
                        }
                        Err(e) => eprintln!("warm {t} failed: {e:#}"),
                    }
                }
            }
            Err(e) => {
                eprintln!("engram load failed: {e:#}");
                *st2.error.lock().unwrap() = Some(format!("{e:#}"));
            }
        }
    });
    let app = Router::new()
        .route("/health", get(health))
        .route("/search", get(search))
        .route("/inject", post(inject_h))
        .with_state(st);
    let addr = format!("{}:{}", a.host, a.port);
    let l = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("bind {addr}"))?;
    eprintln!("terrabeam serve on {addr} tables={:?}", a.tables);
    axum::serve(l, app).with_graceful_shutdown(async { tokio::signal::ctrl_c().await.ok(); }).await?;
    Ok(())
}

async fn health(State(st): State<S>) -> Json<Value> {
    Json(json!({
        "ready": st.eg.lock().unwrap().is_some(),
        "error": *st.error.lock().unwrap(),
        "tables": st.tables,
        "warmed": *st.warmed.lock().unwrap(),
        "queries_served": st.queries.load(Ordering::Relaxed),
        "hits_served": st.hits.load(Ordering::Relaxed),
        "fast_served": st.fast.load(Ordering::Relaxed),
        "engine": "terrabeam",
        "version": env!("CARGO_PKG_VERSION"),
        "rerank_backend": *st.backend.lock().unwrap(),
    }))
}

fn engine(st: &S) -> Result<Arc<engram::Engram>, Response> {
    st.eg.lock().unwrap().clone().ok_or_else(|| {
        err(StatusCode::SERVICE_UNAVAILABLE, st.error.lock().unwrap().clone().unwrap_or_else(|| "engram still loading".into()))
    })
}

fn pick_tables(eg: &engram::Engram, req: Option<&str>) -> Result<Vec<String>, Response> {
    let ts: Vec<String> = req.unwrap_or("").split(',').filter(|t| !t.is_empty()).map(String::from).collect();
    if ts.is_empty() {
        return Ok(eg.tables.keys().cloned().collect());
    }
    let unknown: Vec<&String> = ts.iter().filter(|t| !eg.tables.contains_key(*t)).collect();
    if !unknown.is_empty() {
        let known: Vec<&String> = eg.tables.keys().collect();
        return Err((StatusCode::BAD_REQUEST, Json(json!({"error": format!("unknown tables: {}", unknown.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(",")), "tables": known}))).into_response());
    }
    Ok(ts)
}

/// retrieve on a blocking thread; returns (hits, "fast" | "rerank", took_ms)
async fn run_retrieve(st: &S, eg: Arc<engram::Engram>, q: String, k: usize, tables: Vec<String>) -> Result<(Vec<engram::Hit>, &'static str, u64), Response> {
    let t0 = Instant::now();
    let r = tokio::task::spawn_blocking(move || {
        let names: Vec<&str> = tables.iter().map(|s| s.as_str()).collect();
        eg.retrieve(&q, k, &names)
    })
    .await
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("retrieval task: {e}")))?
    .map_err(|e| {
        eprintln!("search failed: {e:#}");
        err(StatusCode::INTERNAL_SERVER_ERROR, format!("retrieval failed: {e:#}"))
    })?;
    let path = if r.iter().any(|h| h.score.is_some()) || r.is_empty() { "rerank" } else { "fast" };
    st.queries.fetch_add(1, Ordering::Relaxed);
    st.hits.fetch_add(r.len() as u64, Ordering::Relaxed);
    if path == "fast" {
        st.fast.fetch_add(1, Ordering::Relaxed);
    }
    Ok((r, path, t0.elapsed().as_millis() as u64))
}

fn hit_json(h: &engram::Hit, cap: usize) -> Value {
    let text: String = if cap > 0 { h.passage.text.chars().take(cap).collect() } else { h.passage.text.clone() };
    let mut v = json!({"table": h.table, "pid": h.pid, "first": h.first, "title": h.passage.title, "section": h.passage.section, "text": text});
    match h.score {
        Some(s) => v["score"] = json!(s),
        None => v["rerank"] = json!("skipped"),
    }
    v
}

async fn search(State(st): State<S>, Query(p): Query<HashMap<String, String>>) -> Response {
    let eg = match engine(&st) {
        Ok(e) => e,
        Err(r) => return r,
    };
    let q = p.get("q").map(|s| s.trim().to_string()).unwrap_or_default();
    if q.is_empty() {
        return err(StatusCode::BAD_REQUEST, "q is required");
    }
    let k = match p.get("k").map(|s| s.parse::<usize>()).unwrap_or(Ok(3)) {
        Ok(k) => k.clamp(1, 8),
        Err(_) => return err(StatusCode::BAD_REQUEST, "k must be an integer"),
    };
    let tables = match pick_tables(&eg, p.get("tables").map(|s| s.as_str())) {
        Ok(t) => t,
        Err(r) => return r,
    };
    match run_retrieve(&st, eg, q, k, tables).await {
        Ok((hits, path, ms)) => Json(json!({
            "hits": hits.iter().map(|h| hit_json(h, st.text_cap)).collect::<Vec<_>>(),
            "path": path, "took_ms": ms, "queries_served": st.queries.load(Ordering::Relaxed),
        }))
        .into_response(),
        Err(r) => r,
    }
}

#[derive(Deserialize)]
struct InjectReq {
    messages: Vec<Value>,
    #[serde(default = "default_format")]
    format: inject::Format,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    tables: Option<String>,
}
fn default_format() -> inject::Format {
    inject::Format::Sources
}
fn default_k() -> usize {
    3
}

/// POST /inject {messages, format: tablinum|sources, k, tables} -> {messages, injected, sources, path, took_ms}
async fn inject_h(State(st): State<S>, Json(mut r): Json<InjectReq>) -> Response {
    let Some(q) = inject::last_user_text(&r.messages) else {
        return Json(json!({"messages": r.messages, "injected": false, "reason": "last message is not an un-injected user turn"})).into_response();
    };
    let eg = match engine(&st) {
        Ok(e) => e,
        Err(resp) => return resp,
    };
    let tables = match pick_tables(&eg, r.tables.as_deref()) {
        Ok(t) => t,
        Err(resp) => return resp,
    };
    let (hits, path, ms) = match run_retrieve(&st, eg, q.clone(), r.k.clamp(1, 8), tables).await {
        Ok(x) => x,
        Err(resp) => return resp,
    };
    if hits.is_empty() {
        return Json(json!({"messages": r.messages, "injected": false, "reason": "no passages", "took_ms": ms})).into_response();
    }
    let srcs: Vec<inject::Src> = hits.iter().map(|h| inject::Src { table: &h.table, title: &h.passage.title, text: &h.passage.text }).collect();
    let text = inject::render(r.format, &q, &srcs, &inject::new_nonce());
    inject::replace_last_user_text(&mut r.messages, text);
    Json(json!({
        "messages": r.messages, "injected": true, "format": r.format, "path": path, "took_ms": ms,
        "sources": hits.iter().map(|h| json!({"table": h.table, "pid": h.pid, "title": h.passage.title, "score": h.score})).collect::<Vec<_>>(),
    }))
    .into_response()
}
