//! `terrabeam mcp`: the engram (and memory) as an MCP server on stdio (newline-delimited JSON-RPC 2.0), for agent hosts such as
//! Claude Code, Claude Desktop, Cursor, Cline or opencode. Tools: `search_library` (+ `remember` / `recall` with --memory).
//! The handshake is answered at once; the library loads in the background so a first-run pack download cannot time it out.

use crate::{fetch, libtools, manifest::EngramSpec, memory, run::{load_library, LibArgs}};
use anyhow::Result;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::OnceCell;

#[derive(clap::Args, Clone, Debug)]
pub struct McpArgs {
    #[command(flatten)]
    pub lib: LibArgs,
}

type Lib = Option<(Arc<engram::Engram>, Vec<String>)>;

pub async fn mcp(a: McpArgs) -> Result<()> {
    let home = fetch::home();
    let lib: Arc<OnceCell<Result<Lib, String>>> = Arc::new(OnceCell::new());
    let (l2, args2, h2) = (lib.clone(), a.lib.clone(), home.clone());
    tokio::spawn(async move {
        let c = fetch::client();
        let r = load_library(&c, &h2, &args2, &EngramSpec::default()).await.map_err(|e| format!("{e:#}"));
        let _ = l2.set(r);
    });
    let mem = Arc::new(memory::Memory::open(&home, a.lib.memory));
    let k_default = a.lib.k.unwrap_or(3);
    let mut out = tokio::io::stdout();
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Some(line) = lines.next_line().await? {
        let Ok(req) = serde_json::from_str::<Value>(&line) else { continue };
        let id = req.get("id").cloned();
        let method = req["method"].as_str().unwrap_or("");
        let result: Option<Result<Value, (i64, String)>> = match method {
            "initialize" => Some(Ok(json!({
                "protocolVersion": req["params"]["protocolVersion"].as_str().unwrap_or("2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "terrabeam", "version": env!("CARGO_PKG_VERSION")},
                "instructions": "Tablinum knowledge library. Call search_library before answering factual questions; cite the passages you use and say when they do not answer the question."
            }))),
            "ping" => Some(Ok(json!({}))),
            "tools/list" => Some(Ok(json!({"tools": tools(mem.mode)}))),
            "tools/call" => Some(call(&req["params"], &lib, &mem, k_default).await),
            m if m.starts_with("notifications/") => None,
            _ => Some(Err((-32601, format!("method not found: {method}")))),
        };
        if let (Some(id), Some(r)) = (id, result) {
            let msg = match r {
                Ok(v) => json!({"jsonrpc": "2.0", "id": id, "result": v}),
                Err((code, m)) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": m}}),
            };
            out.write_all(format!("{msg}\n").as_bytes()).await?;
            out.flush().await?;
        }
    }
    Ok(())
}

fn tools(mode: memory::Mode) -> Value {
    Value::Array(libtools::specs(mode).into_iter().map(|(n, d, sch)| json!({"name": n, "description": d, "inputSchema": sch})).collect())
}

fn text(s: String, is_error: bool) -> Value {
    json!({"content": [{"type": "text", "text": s}], "isError": is_error})
}

async fn call(p: &Value, lib: &Arc<OnceCell<Result<Lib, String>>>, mem: &memory::Memory, k_default: usize) -> Result<Value, (i64, String)> {
    let name = p["name"].as_str().unwrap_or("");
    if !libtools::is_ours(name, mem.mode) {
        return Err((-32602, format!("unknown tool: {name}")));
    }
    let (eg, tables) = if name == libtools::SEARCH {
        // wait for the background load (first run may be downloading the pack)
        let loaded = loop {
            if let Some(r) = lib.get() {
                break r.clone();
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        };
        match loaded {
            Ok(Some((e, t))) => (Some(e), t),
            Ok(None) => return Ok(text("the library is off (--no-engram)".into(), true)),
            Err(e) => return Ok(text(format!("the library failed to load: {e}"), true)),
        }
    } else {
        (None, vec![])
    };
    let ctx = libtools::Ctx { eg, tables: &tables, mem, user: "default", session: "mcp", k: k_default };
    let (out, err, _) = libtools::exec(name, &p["arguments"], &ctx).await;
    Ok(text(out, err))
}
