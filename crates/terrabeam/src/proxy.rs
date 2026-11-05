//! The OpenAI-compatible proxy shared by `run` (upstream = the llama-server it starts) and `proxy` (upstream = any server:
//! Ollama, LM Studio, vLLM, llama.cpp, a cloud API). Per /v1/chat/completions request: recalled memory goes into the system
//! prompt, engram passages into the last user message, then the request is forwarded (streams pass through) and the exchange
//! is remembered. Everything else is forwarded unchanged.

use crate::{inject, memory};
use axum::body::Body;
use axum::extract::{Path as AxPath, Query, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

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
}

pub fn router(px: Arc<Px>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/engram/search", get(search))
        .route("/health", get(health))
        .route("/memory", get(mem_list).delete(mem_clear))
        .route("/memory/{id}", delete(mem_delete))
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
    Json(json!({"status": "ok", "model": px.model, "engram": px.eg.is_some(), "tables": px.tables, "memory": px.mem.mode}))
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

async fn retrieve(px: &Px, eg: Arc<engram::Engram>, q: String, k: usize) -> Option<Vec<engram::Hit>> {
    let tables = px.tables.clone();
    let job = tokio::task::spawn_blocking(move || eg.retrieve(&q, k, &tables.iter().map(|s| s.as_str()).collect::<Vec<_>>()));
    match tokio::time::timeout(Duration::from_secs(8), job).await {
        Ok(Ok(Ok(h))) => Some(h),
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

/// POST /v1/chat/completions
async fn chat(State(px): State<Arc<Px>>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    let Ok(mut v) = serde_json::from_slice::<Value>(&body) else { return send(&px, &headers, body.to_vec(), vec![], None).await };
    let flag = v.as_object_mut().and_then(|o| o.remove("engram"));
    let mflag = v.as_object_mut().and_then(|o| o.remove("memory"));
    let eh = hdr(&headers, "x-engram").unwrap_or("").to_ascii_lowercase();
    let mh = hdr(&headers, "x-memory").unwrap_or("").to_ascii_lowercase();
    let off = |f: &Option<Value>, h: &str| matches!(f, Some(Value::Bool(false))) || matches!(h, "off" | "0" | "false" | "no");
    let (user, session) = who(&headers, &v);
    let mut extra = vec![];
    let mut used = None;
    let question = v["messages"].as_array().and_then(|m| inject::last_user(m)).map(|(_, q)| q);
    if let (Some(q), Some(msgs)) = (question.clone(), v.get_mut("messages").and_then(|m| m.as_array_mut())) {
        // memory first (system prompt), then the engram (user message) - the memory never sees retrieved passages
        if px.mem.mode != memory::Mode::Off && !off(&mflag, &mh) {
            let ctx: Vec<String> = msgs.iter().map(|m| m["content"].to_string()).collect();
            let notes = px.mem.recall(&user, &session, &q, &ctx, px.mem_k);
            if !notes.is_empty() {
                add_system(msgs, &memory::block(px.mem.mode, &notes));
                extra.push(("x-terrabeam-memory", notes.len().to_string()));
            }
        }
        if let (Some(eg), false) = (px.eg.clone(), off(&flag, &eh)) {
            if let Some((i, _)) = inject::last_user(msgs) {
                if let Some(hits) = retrieve(&px, eg, q.clone(), px.k).await.filter(|h| !h.is_empty()) {
                    let srcs: Vec<inject::Src> = hits.iter().map(|h| inject::Src { table: &h.table, title: &h.passage.title, text: &h.passage.text }).collect();
                    inject::replace_text(msgs, i, inject::render(px.format, &q, &srcs, &inject::new_nonce()));
                    used = Some(hits.iter().map(|h| format!("{}:{}", h.table, h.pid)).collect::<Vec<_>>().join(","));
                }
            }
        }
    }
    if let Some(u) = used {
        extra.push(("x-engram-sources", u));
    }
    let remember = (px.mem.mode != memory::Mode::Off && !off(&mflag, &mh)).then(|| question).flatten().map(|q| (user, session, q));
    send(&px, &headers, serde_json::to_vec(&v).unwrap_or(body.to_vec()), extra, remember).await
}

async fn forward(State(px): State<Arc<Px>>, req: Request) -> Response {
    let path = req.uri().path_and_query().map(|p| p.as_str().to_string()).unwrap_or("/".into());
    let (method, headers) = (req.method().clone(), req.headers().clone());
    let body = axum::body::to_bytes(req.into_body(), 64 << 20).await.unwrap_or_default();
    match px.http.request(method, format!("{}{path}", px.upstream)).headers(clean(&headers)).body(body.to_vec()).send().await {
        Ok(r) => relay(r, vec![], None, px.mem.clone()),
        Err(e) => (StatusCode::BAD_GATEWAY, format!("upstream: {e}")).into_response(),
    }
}

type Remember = Option<(String, String, String)>;

async fn send(px: &Px, headers: &HeaderMap, body: Vec<u8>, extra: Vec<(&'static str, String)>, remember: Remember) -> Response {
    let mut h = clean(headers);
    h.insert("content-type", HeaderValue::from_static("application/json"));
    match px.http.post(format!("{}/v1/chat/completions", px.upstream)).headers(h).body(body).send().await {
        Ok(r) => relay(r, extra, remember, px.mem.clone()),
        Err(e) => (StatusCode::BAD_GATEWAY, format!("upstream: {e}")).into_response(),
    }
}

fn clean(h: &HeaderMap) -> HeaderMap {
    let mut out = h.clone();
    for k in ["host", "content-length", "x-engram", "x-memory", "connection", "transfer-encoding", "accept-encoding"] {
        out.remove(k);
    }
    out
}

/// Pass the upstream response through. When `remember` is set, the answer text is collected on the way (JSON body, or the
/// `delta.content` of each SSE event) and stored with the question once the response ends.
fn relay(r: reqwest::Response, extra: Vec<(&'static str, String)>, remember: Remember, mem: Arc<memory::Memory>) -> Response {
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
    let Some((user, session, q)) = remember.filter(|_| ok) else {
        return b.body(Body::from_stream(r.bytes_stream())).unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response());
    };
    let tap = Tap { mem, user, session, q, answer: String::new(), raw: vec![], line: vec![], sse: false, stored: false };
    b.body(Body::from_stream(Tapped { inner: Box::pin(r.bytes_stream()), tap })).unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}

/// Collects the answer while the body passes through: `delta.content` of SSE events, or the JSON body's message content.
struct Tap {
    mem: Arc<memory::Memory>,
    user: String,
    session: String,
    q: String,
    answer: String,
    raw: Vec<u8>,
    line: Vec<u8>,
    sse: bool,
    stored: bool,
}

impl Tap {
    fn feed(&mut self, c: &[u8]) {
        if self.raw.len() < (8 << 20) {
            self.raw.extend_from_slice(c);
        }
        self.line.extend_from_slice(c);
        while let Some(i) = self.line.iter().position(|&x| x == b'\n') {
            let l: Vec<u8> = self.line.drain(..=i).collect();
            let l = String::from_utf8_lossy(&l);
            if let Some(d) = l.trim().strip_prefix("data:") {
                self.sse = true;
                if let Ok(v) = serde_json::from_str::<Value>(d.trim()) {
                    if let Some(t) = v["choices"][0]["delta"]["content"].as_str() {
                        self.answer.push_str(t);
                    }
                }
            }
        }
    }
    fn finish(&mut self) {
        if self.stored {
            return;
        }
        self.stored = true;
        if !self.sse {
            if let Ok(v) = serde_json::from_slice::<Value>(&self.raw) {
                self.answer = v["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_string();
            }
        }
        self.mem.add(&self.user, &self.session, "user", &self.q);
        if !self.answer.trim().is_empty() {
            self.mem.add(&self.user, &self.session, "assistant", &self.answer);
        }
    }
}

struct Tapped<S> {
    inner: std::pin::Pin<Box<S>>,
    tap: Tap,
}

impl<S, E> futures_util::Stream for Tapped<S>
where
    S: futures_util::Stream<Item = Result<bytes::Bytes, E>> + Send,
{
    type Item = Result<bytes::Bytes, E>;
    fn poll_next(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
        let p = self.inner.as_mut().poll_next(cx);
        match &p {
            std::task::Poll::Ready(Some(Ok(c))) => {
                let c = c.clone();
                self.tap.feed(&c);
            }
            std::task::Poll::Ready(None) | std::task::Poll::Ready(Some(Err(_))) => self.tap.finish(),
            _ => {}
        }
        p
    }
}
