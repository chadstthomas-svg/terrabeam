//! The OpenAI-compatible proxy shared by `run` (upstream = the llama-server it starts) and `proxy` (upstream = any server:
//! Ollama, LM Studio, vLLM, llama.cpp, a cloud API). Per /v1/chat/completions request:
//! 1. canonical history: assistant turns the client trimmed get their reasoning and hidden library steps back (history.rs);
//! 2. recalled memory goes into the system prompt, engram passages into the last user message;
//! 3. with the library tool on, the model may call `search_library` (and `remember` / `recall`) itself; terrabeam runs those
//!    calls and continues, the client only ever sees its own tools;
//! 4. the answer streams back; the exchange is remembered, stored as canonical history and, with --record, written to disk.
//!
//! Everything else is forwarded unchanged.

use crate::assemble::{Assembled, SseReader};
use crate::history::{History, Lru};
use crate::record::{Exchange, Recorder};
use crate::{env, inject, libtools, memory};
use axum::body::Body;
use axum::extract::{Path as AxPath, Query, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct Px {
    pub upstream: String,
    pub http: reqwest::Client,
    pub eg: Option<Arc<engram::Engram>>,
    pub tables: Vec<String>,
    pub format: inject::Format,
    pub k: usize,
    pub model: String,
    pub mem: Arc<memory::Memory>,
    pub mem_k: usize,
    /// put trimmed reasoning and hidden tool steps back into earlier assistant turns
    pub history: Option<History>,
    pub recorder: Option<Recorder>,
    /// offer search_library (+ remember / recall) as tools the model can call; terrabeam runs them
    pub library_tool: bool,
    /// most model calls per request in the tool loop
    pub max_steps: usize,
    /// the harness this proxy implements (see harness_id); sent as x-terrabeam-harness and recorded
    pub harness: String,
    pub cache: Lru<Vec<engram::Hit>>,
}

impl Px {
    pub fn new(upstream: String, eg: Option<Arc<engram::Engram>>, tables: Vec<String>, format: inject::Format, k: usize, model: String, mem: Arc<memory::Memory>) -> Px {
        Px {
            upstream, http: reqwest::Client::new(), eg, tables, format, k, model, mem, mem_k: 3, history: None, recorder: None,
            library_tool: false, max_steps: 6, harness: String::new(), cache: Mutex::new((HashMap::new(), VecDeque::new())),
        }
    }
}

/// Everything about the serving setup a model can be trained to: injection layout, passages per question, library tool,
/// preserved reasoning. `tb1;inject=tablinum;k=3;tool=none;reasoning=keep` = the setup Moon-Beam-8B-A1B was trained on.
pub fn harness_id(format: Option<inject::Format>, k: usize, library_tool: bool, keep_reasoning: bool) -> String {
    format!(
        "tb1;inject={};k={k};tool={};reasoning={}",
        match format {
            Some(inject::Format::Tablinum) => "tablinum",
            Some(inject::Format::Sources) => "sources",
            None => "off",
        },
        if library_tool { "library" } else { "none" },
        if keep_reasoning { "keep" } else { "client" }
    )
}

pub fn router(px: Arc<Px>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/engram/search", get(search))
        .route("/health", get(health))
        .route("/memory", get(mem_list).delete(mem_clear))
        .route("/memory/{id}", delete(mem_delete))
        .route("/env/render", post(env::render_h))
        .route("/env/grade", post(env::grade_h))
        .fallback(forward)
        .with_state(px)
}

fn hdr<'a>(h: &'a HeaderMap, k: &str) -> Option<&'a str> {
    h.get(k).and_then(|v| v.to_str().ok()).filter(|s| !s.is_empty())
}

/// (user, session): explicit headers, OpenWebUI's forwarded user/chat headers, the OpenAI `user` field, else "default" and a
/// hash of the conversation's first user message (stable across the turns of one chat for clients that send no ids).
fn who(h: &HeaderMap, body: &Value) -> (String, String) {
    let user = hdr(h, "x-user-id").or(hdr(h, "x-openwebui-user-id")).or(body.get("user").and_then(|u| u.as_str())).unwrap_or("default").to_string();
    let session = hdr(h, "x-session-id").or(hdr(h, "x-openwebui-chat-id")).map(String::from).unwrap_or_else(|| {
        use sha2::Digest;
        let first = body["messages"].as_array().and_then(|m| m.iter().find(|x| x["role"] == "user")).map(|m| m["content"].to_string()).unwrap_or_default();
        hex::encode(&sha2::Sha256::digest(first.as_bytes())[..8])
    });
    (user, session)
}

fn add_system(msgs: &mut Vec<Value>, text: &str) {
    match msgs.first_mut() {
        Some(m) if m["role"] == "system" && m["content"].is_string() => {
            let c = m["content"].as_str().unwrap_or_default().to_string();
            m["content"] = Value::String(format!("{c}\n\n{text}"));
        }
        _ => msgs.insert(0, json!({"role": "system", "content": text})),
    }
}

async fn health(State(px): State<Arc<Px>>) -> Json<Value> {
    Json(json!({"status": "ok", "model": px.model, "engram": px.eg.is_some(), "tables": px.tables, "memory": px.mem.mode,
        "harness": px.harness, "library_tool": px.library_tool, "keep_reasoning": px.history.is_some(), "record": px.recorder.is_some()}))
}

async fn search(State(px): State<Arc<Px>>, Query(p): Query<HashMap<String, String>>) -> Response {
    let Some(eg) = px.eg.clone() else { return (StatusCode::NOT_FOUND, "engram is off").into_response() };
    let q = p.get("q").cloned().unwrap_or_default();
    let k = p.get("k").and_then(|k| k.parse().ok()).unwrap_or(px.k);
    match retrieve(&px, eg, q, k).await {
        Some(h) => Json(json!({"hits": h.iter().map(|h| json!({"table": h.table, "pid": h.pid, "title": h.passage.title, "score": h.score, "text": h.passage.text})).collect::<Vec<_>>()})).into_response(),
        None => (StatusCode::INTERNAL_SERVER_ERROR, "retrieval failed").into_response(),
    }
}

const CACHE_CAP: usize = 512;

/// Retrieval with a small cache: an agent loop re-sends the same user turn on every call, and identical passages keep the
/// prompt identical (llama.cpp's prompt cache keeps working) without paying the re-rank again.
pub async fn retrieve(px: &Px, eg: Arc<engram::Engram>, q: String, k: usize) -> Option<Vec<engram::Hit>> {
    let key = format!("{k}|{q}");
    if let Some(h) = px.cache.lock().unwrap().0.get(&key) {
        return Some(h.clone());
    }
    let tables = px.tables.clone();
    let job = tokio::task::spawn_blocking(move || eg.retrieve(&q, k, &tables.iter().map(|s| s.as_str()).collect::<Vec<_>>()));
    match tokio::time::timeout(Duration::from_secs(8), job).await {
        Ok(Ok(Ok(h))) => {
            let mut c = px.cache.lock().unwrap();
            if c.0.insert(key.clone(), h.clone()).is_none() {
                c.1.push_back(key);
            }
            while c.1.len() > CACHE_CAP {
                if let Some(o) = c.1.pop_front() {
                    c.0.remove(&o);
                }
            }
            Some(h)
        }
        _ => None,
    }
}

async fn mem_list(State(px): State<Arc<Px>>, headers: HeaderMap) -> Json<Value> {
    let (user, _) = who(&headers, &Value::Null);
    Json(json!({"mode": px.mem.mode, "user": user, "notes": px.mem.list(&user)}))
}

async fn mem_clear(State(px): State<Arc<Px>>, headers: HeaderMap) -> Json<Value> {
    let (user, _) = who(&headers, &Value::Null);
    Json(json!({"deleted": px.mem.delete(&user, None)}))
}

async fn mem_delete(State(px): State<Arc<Px>>, headers: HeaderMap, AxPath(id): AxPath<String>) -> Json<Value> {
    let (user, _) = who(&headers, &Value::Null);
    Json(json!({"deleted": px.mem.delete(&user, Some(&id))}))
}

fn is_off(f: &Option<Value>, h: &str) -> bool {
    matches!(f, Some(Value::Bool(false))) || matches!(h, "off" | "0" | "false" | "no")
}

/// What happens once the answer is known: memory, canonical history, record.
pub struct Fin {
    pub px: Arc<Px>,
    pub user: String,
    pub session: String,
    /// the question to remember (memory on)
    pub remember: Option<String>,
    pub request: Value,
    /// messages the model saw in its last call
    pub sent: Vec<Value>,
    pub tools: Option<Value>,
    pub sources: Vec<String>,
    /// hidden library-tool steps that came before the final answer
    pub hidden: Vec<Value>,
    pub steps: usize,
    pub t0: Instant,
}

impl Fin {
    /// `fin` = the model's final message (with reasoning); `visible` = what the client received.
    pub fn done(&self, fin: &Assembled, visible: &Value) {
        let content = visible["content"].as_str().unwrap_or_default();
        if let Some(q) = &self.remember {
            self.px.mem.add(&self.user, &self.session, "user", q);
            if !content.trim().is_empty() {
                self.px.mem.add(&self.user, &self.session, "assistant", content);
            }
        }
        let fm = fin.message();
        if let Some(h) = &self.px.history {
            let mut canon = self.hidden.clone();
            canon.push(fm.clone());
            h.store(&format!("{}|{}", self.user, self.session), visible, canon);
        }
        if let Some(r) = &self.px.recorder {
            let mut traj = self.sent.clone();
            traj.push(fm);
            r.write(Exchange {
                harness: &self.px.harness, model: &self.px.model, user: &self.user, session: &self.session, request: &self.request,
                trajectory: traj, tools: self.tools.as_ref(), response: visible.clone(), sources: self.sources.clone(),
                finish: fin.finish.clone(), usage: fin.usage.clone(), steps: self.steps, ms: self.t0.elapsed().as_millis() as u64,
            });
        }
    }
}

/// POST /v1/chat/completions
async fn chat(State(px): State<Arc<Px>>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    let t0 = Instant::now();
    let Ok(mut v) = serde_json::from_slice::<Value>(&body) else { return send(&px, &headers, body.to_vec(), vec![], None).await };
    let request = v.clone();
    let flag = v.as_object_mut().and_then(|o| o.remove("engram"));
    let mflag = v.as_object_mut().and_then(|o| o.remove("memory"));
    let tflag = v.as_object_mut().and_then(|o| o.remove("library_tool"));
    let eh = hdr(&headers, "x-engram").unwrap_or("").to_ascii_lowercase();
    let mh = hdr(&headers, "x-memory").unwrap_or("").to_ascii_lowercase();
    let th = hdr(&headers, "x-library-tool").unwrap_or("").to_ascii_lowercase();
    let (user, session) = who(&headers, &v);
    let mut extra = vec![("x-terrabeam-harness", px.harness.clone())];
    let mut sources: Vec<String> = vec![];
    if let (Some(h), Some(msgs)) = (&px.history, v.get_mut("messages").and_then(|m| m.as_array_mut())) {
        let n = h.restore(&format!("{user}|{session}"), msgs);
        if n > 0 {
            extra.push(("x-terrabeam-restored", n.to_string()));
        }
    }
    let question = v["messages"].as_array().and_then(|m| inject::last_user(m)).map(|(_, q)| q);
    let mem_on = px.mem.mode != memory::Mode::Off && !is_off(&mflag, &mh);
    if let (Some(q), Some(msgs)) = (question.clone(), v.get_mut("messages").and_then(|m| m.as_array_mut())) {
        // memory is recalled first (it never sees retrieved passages). With the Tablinum layout the notes go in front of the
        // library passages, where the model looks for answers; otherwise into the system prompt.
        let notes = if mem_on {
            let ctx: Vec<String> = msgs.iter().map(|m| m["content"].to_string()).collect();
            px.mem.recall(&user, &session, &q, &ctx, px.mem_k)
        } else {
            vec![]
        };
        if !notes.is_empty() {
            extra.push(("x-terrabeam-memory", notes.len().to_string()));
        }
        let mut notes_placed = false;
        if let (Some(eg), false) = (px.eg.clone(), is_off(&flag, &eh)) {
            if let Some((i, _)) = inject::last_user(msgs) {
                if let Some(hits) = retrieve(&px, eg, q.clone(), px.k).await.filter(|h| !h.is_empty()) {
                    let note_texts = if px.format == inject::Format::Tablinum { memory::as_passages(&notes) } else { vec![] };
                    let mut srcs: Vec<inject::Src> = note_texts.iter().map(|t| inject::Src { table: "memory", title: "", text: t }).collect();
                    srcs.extend(hits.iter().map(|h| inject::Src { table: &h.table, title: &h.passage.title, text: &h.passage.text }));
                    inject::replace_text(msgs, i, inject::render(px.format, &q, &srcs, &inject::new_nonce()));
                    sources = hits.iter().map(|h| format!("{}:{}", h.table, h.pid)).collect();
                    notes_placed = !note_texts.is_empty();
                }
            }
        }
        if !notes.is_empty() && !notes_placed {
            add_system(msgs, &memory::block(px.mem.mode, &notes));
        }
    }
    if !sources.is_empty() {
        extra.push(("x-engram-sources", sources.join(",")));
    }
    let tool_loop = px.library_tool && px.eg.is_some() && !is_off(&flag, &eh) && !is_off(&tflag, &th);
    let mem_mode = if mem_on { px.mem.mode } else { memory::Mode::Off };
    if tool_loop {
        let ours = libtools::openai_tools(mem_mode);
        let have: Vec<String> = v["tools"].as_array().into_iter().flatten().filter_map(|t| t["function"]["name"].as_str().map(String::from)).collect();
        let mut tools = v["tools"].as_array().cloned().unwrap_or_default();
        tools.extend(ours.into_iter().filter(|t| !have.iter().any(|h| Some(h.as_str()) == t["function"]["name"].as_str())));
        v["tools"] = Value::Array(tools);
    }
    let fin = Fin {
        px: px.clone(), user, session, remember: if mem_on { question } else { None }, request, sent: v["messages"].as_array().cloned().unwrap_or_default(),
        tools: v.get("tools").cloned(), sources, hidden: vec![], steps: 1, t0,
    };
    if tool_loop {
        let stream = v["stream"].as_bool().unwrap_or(false);
        return tool_loop_resp(px, headers, v, fin, mem_mode, stream, extra).await;
    }
    let needs_fin = fin.remember.is_some() || px.history.is_some() || px.recorder.is_some();
    send(&px, &headers, serde_json::to_vec(&v).unwrap_or(body.to_vec()), extra, needs_fin.then_some(fin)).await
}

async fn forward(State(px): State<Arc<Px>>, req: Request) -> Response {
    let path = req.uri().path_and_query().map(|p| p.as_str().to_string()).unwrap_or("/".into());
    let (method, headers) = (req.method().clone(), req.headers().clone());
    let body = axum::body::to_bytes(req.into_body(), 64 << 20).await.unwrap_or_default();
    match px.http.request(method, format!("{}{path}", px.upstream)).headers(clean(&headers)).body(body.to_vec()).send().await {
        Ok(r) => relay(r, vec![], None),
        Err(e) => (StatusCode::BAD_GATEWAY, format!("upstream: {e}")).into_response(),
    }
}

async fn send(px: &Px, headers: &HeaderMap, body: Vec<u8>, extra: Vec<(&'static str, String)>, fin: Option<Fin>) -> Response {
    match post_upstream(px, headers, body).await {
        Ok(r) => relay(r, extra, fin),
        Err(e) => (StatusCode::BAD_GATEWAY, format!("upstream: {e}")).into_response(),
    }
}

async fn post_upstream(px: &Px, headers: &HeaderMap, body: Vec<u8>) -> reqwest::Result<reqwest::Response> {
    let mut h = clean(headers);
    h.insert("content-type", HeaderValue::from_static("application/json"));
    px.http.post(format!("{}/v1/chat/completions", px.upstream)).headers(h).body(body).send().await
}

fn clean(h: &HeaderMap) -> HeaderMap {
    let mut out = h.clone();
    for k in ["host", "content-length", "x-engram", "x-memory", "x-library-tool", "connection", "transfer-encoding", "accept-encoding"] {
        out.remove(k);
    }
    out
}

/// Pass the upstream response through. With `fin`, the answer is assembled on the way (JSON body or SSE deltas) and handed
/// to [`Fin::done`] once the response ends.
fn relay(r: reqwest::Response, extra: Vec<(&'static str, String)>, fin: Option<Fin>) -> Response {
    let status = StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let ok = r.status().is_success();
    let mut b = Response::builder().status(status);
    for (k, v) in r.headers() {
        if !matches!(k.as_str(), "content-length" | "transfer-encoding" | "connection") {
            b = b.header(k, v);
        }
    }
    for (k, v) in extra {
        if let Ok(v) = HeaderValue::from_str(&v) {
            b = b.header(k, v);
        }
    }
    let Some(fin) = fin.filter(|_| ok) else {
        return b.body(Body::from_stream(r.bytes_stream())).unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response());
    };
    let tee = Tee { inner: Box::pin(r.bytes_stream()), rd: SseReader::default(), fin: Some(fin) };
    b.body(Body::from_stream(tee)).unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}

struct Tee<S> {
    inner: std::pin::Pin<Box<S>>,
    rd: SseReader,
    fin: Option<Fin>,
}

impl<S> Tee<S> {
    fn finish(&mut self) {
        if let Some(fin) = self.fin.take() {
            let a = self.rd.finish().clone();
            let mut visible = a.message();
            if let Some(o) = visible.as_object_mut() {
                o.remove("reasoning_content");
            }
            fin.done(&a, &visible);
        }
    }
}

impl<S, E> futures_util::Stream for Tee<S>
where
    S: futures_util::Stream<Item = Result<Bytes, E>> + Send,
{
    type Item = Result<Bytes, E>;
    fn poll_next(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
        let p = self.inner.as_mut().poll_next(cx);
        match &p {
            std::task::Poll::Ready(Some(Ok(c))) => {
                let c = c.clone();
                self.rd.feed(&c);
            }
            std::task::Poll::Ready(None) | std::task::Poll::Ready(Some(Err(_))) => self.finish(),
            _ => {}
        }
        p
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// library tool loop

/// Split the model's tool calls into terrabeam's own and the client's.
fn split_calls(calls: &[Value], mem: memory::Mode) -> (Vec<Value>, Vec<Value>) {
    calls.iter().cloned().partition(|tc| libtools::is_ours(tc["function"]["name"].as_str().unwrap_or(""), mem))
}

/// Run terrabeam's calls; returns the tool messages, sources and a one-line note per call (shown in the reasoning stream).
async fn run_calls(px: &Px, calls: &[Value], fin: &Fin, mem: memory::Mode) -> (Vec<Value>, Vec<String>, Vec<String>) {
    let ctx = libtools::Ctx { eg: px.eg.clone(), tables: &px.tables, mem: &px.mem, user: &fin.user, session: &fin.session, k: px.k };
    let (mut msgs, mut srcs, mut notes) = (vec![], vec![], vec![]);
    for tc in calls {
        let name = tc["function"]["name"].as_str().unwrap_or("");
        let args = libtools::args_of(tc);
        let (out, _err, s) = if libtools::is_ours(name, mem) { libtools::exec(name, &args, &ctx).await } else { (format!("unknown tool: {name}"), true, vec![]) };
        notes.push(format!("[{name}: {}]", args.get("query").or(args.get("text")).and_then(|q| q.as_str()).unwrap_or("")));
        msgs.push(json!({"role": "tool", "tool_call_id": tc["id"], "name": name, "content": out}));
        srcs.extend(s);
    }
    (msgs, srcs, notes)
}

fn strip_our_tools(v: &mut Value, mem: memory::Mode) {
    let empty = match v.get_mut("tools").and_then(|t| t.as_array_mut()) {
        Some(ts) => {
            ts.retain(|t| !libtools::is_ours(t["function"]["name"].as_str().unwrap_or(""), mem));
            ts.is_empty()
        }
        None => false,
    };
    if empty {
        if let Some(o) = v.as_object_mut() {
            o.remove("tools");
            o.remove("tool_choice");
        }
    }
}

fn sse(v: &Value) -> Bytes {
    Bytes::from(format!("data: {v}\n\n"))
}

/// The client's tool calls, re-indexed from 0, as one SSE delta.
fn client_calls_event(base: &Value, calls: &[Value]) -> Value {
    let tcs: Vec<Value> = calls.iter().enumerate().map(|(i, c)| json!({"index": i, "id": c["id"], "type": "function", "function": c["function"]})).collect();
    let mut e = base.clone();
    e["choices"] = json!([{"index": 0, "delta": {"tool_calls": tcs}, "finish_reason": null}]);
    e
}

async fn tool_loop_resp(px: Arc<Px>, headers: HeaderMap, mut v: Value, mut fin: Fin, mem: memory::Mode, stream: bool, extra: Vec<(&'static str, String)>) -> Response {
    let mut b = Response::builder().status(StatusCode::OK);
    for (k, val) in &extra {
        if let Ok(val) = HeaderValue::from_str(val) {
            b = b.header(*k, val);
        }
    }
    if !stream {
        return match run_loop(&px, &headers, &mut v, &mut fin, mem, None).await {
            Ok((body, a, visible)) => {
                fin.done(&a, &visible);
                b.header("content-type", "application/json").body(Body::from(body.to_string())).unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
            }
            Err((code, msg)) => (code, msg).into_response(),
        };
    }
    let (tx, rx) = tokio::sync::mpsc::channel::<Bytes>(64);
    tokio::spawn(async move {
        match run_loop(&px, &headers, &mut v, &mut fin, mem, Some(&tx)).await {
            Ok((_, a, visible)) => fin.done(&a, &visible),
            Err((_, msg)) => {
                let _ = tx.send(sse(&json!({"error": {"message": msg, "type": "upstream"}}))).await;
            }
        }
        let _ = tx.send(Bytes::from_static(b"data: [DONE]\n\n")).await;
    });
    let body = futures_util::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|b| (Ok::<_, std::io::Error>(b), rx)) });
    b.header("content-type", "text/event-stream").header("cache-control", "no-cache").body(Body::from_stream(body)).unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}

/// The loop: call the model; if it only calls terrabeam's tools, run them and call again (at most `max_steps` calls; the last
/// call is made without terrabeam's tools so it has to answer). Streaming: content and reasoning deltas go to the client as
/// they come, tool-call deltas are held back. Returns (final JSON body for non-streaming, final assembled step, visible msg).
async fn run_loop(px: &Px, headers: &HeaderMap, v: &mut Value, fin: &mut Fin, mem: memory::Mode, tx: Option<&tokio::sync::mpsc::Sender<Bytes>>) -> Result<(Value, Assembled, Value), (StatusCode, String)> {
    let stream = tx.is_some();
    v["stream"] = json!(stream);
    let max = px.max_steps.max(1);
    let (mut shown_content, mut shown_reasoning) = (String::new(), String::new());
    let mut base = json!({"object": "chat.completion.chunk"});
    for step in 1..=max {
        if step == max {
            strip_our_tools(v, mem);
        }
        fin.sent = v["messages"].as_array().cloned().unwrap_or_default();
        fin.steps = step;
        let r = post_upstream(px, headers, serde_json::to_vec(&v).unwrap_or_default()).await.map_err(|e| (StatusCode::BAD_GATEWAY, format!("upstream: {e}")))?;
        if !r.status().is_success() {
            let code = StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            return Err((code, r.text().await.unwrap_or_default()));
        }
        let mut rd = SseReader::default();
        let (mut held_finish, mut held_usage) = (None::<Value>, None::<Value>);
        let mut body = r.bytes_stream();
        while let Some(c) = body.next().await {
            let c = c.map_err(|e| (StatusCode::BAD_GATEWAY, format!("upstream: {e}")))?;
            let events = rd.feed(&c);
            let Some(tx) = tx else { continue };
            for d in events {
                let Ok(mut ev) = serde_json::from_str::<Value>(&d) else { continue };
                for k in ["id", "created", "model", "system_fingerprint"] {
                    if let Some(x) = ev.get(k) {
                        base[k] = x.clone();
                    }
                }
                if ev["choices"].as_array().is_none_or(|c| c.is_empty()) {
                    held_usage = Some(ev);
                    continue;
                }
                let ch = &mut ev["choices"][0];
                if let Some(d) = ch.get_mut("delta").and_then(|d| d.as_object_mut()) {
                    d.remove("tool_calls");
                }
                if ch["finish_reason"].is_string() {
                    held_finish = Some(ev);
                    continue;
                }
                let d = &ch["delta"];
                let has = ["content", "reasoning_content", "reasoning"].iter().any(|k| d.get(*k).and_then(|x| x.as_str()).is_some_and(|s| !s.is_empty()))
                    || (step == 1 && d.get("role").is_some());
                if has {
                    let _ = tx.send(sse(&ev)).await;
                }
            }
        }
        let a = rd.finish().clone();
        let last_json: Value = if stream { Value::Null } else { serde_json::from_slice(&rd.raw).unwrap_or(Value::Null) };
        shown_content.push_str(&a.content);
        if !a.reasoning.is_empty() {
            shown_reasoning.push_str(&a.reasoning);
        }
        let (ours, client) = split_calls(&a.tool_calls, mem);
        if !ours.is_empty() && client.is_empty() && step < max {
            let (tool_msgs, srcs, notes) = run_calls(px, &ours, fin, mem).await;
            let mut am = a.message();
            am["tool_calls"] = json!(ours);
            let msgs = v["messages"].as_array_mut().ok_or((StatusCode::BAD_REQUEST, "messages missing".to_string()))?;
            msgs.push(am.clone());
            msgs.extend(tool_msgs.iter().cloned());
            fin.hidden.push(am);
            fin.hidden.extend(tool_msgs);
            fin.sources.extend(srcs);
            let note = format!("\n\n{}\n\n", notes.join(" "));
            shown_reasoning.push_str(&note);
            if let Some(tx) = tx {
                let mut e = base.clone();
                e["choices"] = json!([{"index": 0, "delta": {"reasoning_content": note}, "finish_reason": null}]);
                let _ = tx.send(sse(&e)).await;
            }
            continue;
        }
        // final step: only the client's tool calls are visible (terrabeam's own are dropped when mixed with them)
        let mut fa = a.clone();
        fa.tool_calls = client.clone();
        let finish = if !client.is_empty() {
            "tool_calls"
        } else if a.finish.as_deref() == Some("tool_calls") {
            "stop"
        } else {
            a.finish.as_deref().unwrap_or("stop")
        };
        fa.finish = Some(finish.to_string());
        let mut visible = json!({"role": "assistant", "content": if shown_content.is_empty() && !client.is_empty() { Value::Null } else { json!(shown_content) }});
        if !client.is_empty() {
            visible["tool_calls"] = json!(client);
        }
        if let Some(tx) = tx {
            if !client.is_empty() {
                let _ = tx.send(sse(&client_calls_event(&base, &client))).await;
            }
            let mut e = held_finish.unwrap_or_else(|| base.clone());
            e["choices"] = json!([{"index": 0, "delta": {}, "finish_reason": finish}]);
            let _ = tx.send(sse(&e)).await;
            if let Some(u) = held_usage {
                let _ = tx.send(sse(&u)).await;
            }
            return Ok((Value::Null, fa, visible));
        }
        let mut out = if last_json.is_object() { last_json } else { json!({"object": "chat.completion"}) };
        let mut msg = visible.clone();
        if !shown_reasoning.is_empty() {
            msg["reasoning_content"] = json!(shown_reasoning);
        }
        out["choices"] = json!([{"index": 0, "message": msg, "finish_reason": finish}]);
        return Ok((out, fa, visible));
    }
    Err((StatusCode::INTERNAL_SERVER_ERROR, "tool loop ended without an answer".into()))
}
