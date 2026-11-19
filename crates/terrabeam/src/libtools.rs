//! The library as tools the model can call itself: `search_library` (and `remember` / `recall` when memory is on).
//! One definition and one executor shared by the MCP server and the proxy's tool loop, so a model trained on either sees the
//! same tool names, schemas and result layout.

use crate::memory;
use serde_json::{json, Value};
use std::sync::Arc;

pub const SEARCH: &str = "search_library";
pub const REMEMBER: &str = "remember";
pub const RECALL: &str = "recall";

pub const SEARCH_DESC: &str = "Search the Tablinum knowledge library (Wikipedia-style reference, law, software docs, research papers, Stack Exchange, open textbooks) and return the best passages for a question. Use a passage only if it states the answer.";

/// (name, description, JSON schema) of each tool for this memory mode.
pub fn specs(mem: memory::Mode) -> Vec<(&'static str, &'static str, Value)> {
    let mut t = vec![(
        SEARCH,
        SEARCH_DESC,
        json!({"type": "object", "properties": {"query": {"type": "string", "description": "the question, in plain words"}, "k": {"type": "integer", "minimum": 1, "maximum": 8, "description": "passages to return (default 3)"}}, "required": ["query"]}),
    )];
    if mem != memory::Mode::Off {
        t.push((REMEMBER, "Save a short note to the user's persistent memory (credentials are redacted).", json!({"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})));
        t.push((RECALL, "Find notes in the user's persistent memory that relate to a question.", json!({"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})));
    }
    t
}

/// OpenAI `tools` entries.
pub fn openai_tools(mem: memory::Mode) -> Vec<Value> {
    specs(mem).into_iter().map(|(n, d, s)| json!({"type": "function", "function": {"name": n, "description": d, "parameters": s}})).collect()
}

pub fn is_ours(name: &str, mem: memory::Mode) -> bool {
    name == SEARCH || (mem != memory::Mode::Off && (name == REMEMBER || name == RECALL))
}

/// Passages as a tool result: `[1] (table: title) text`, blank line between.
pub fn format_hits(hits: &[engram::Hit]) -> String {
    hits.iter()
        .enumerate()
        .map(|(i, h)| format!("[{}] ({}{}) {}", i + 1, h.table, if h.passage.title.is_empty() { String::new() } else { format!(": {}", h.passage.title) }, h.passage.text))
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub struct Ctx<'a> {
    pub eg: Option<Arc<engram::Engram>>,
    pub tables: &'a [String],
    pub mem: &'a memory::Memory,
    pub user: &'a str,
    pub session: &'a str,
    pub k: usize,
}

/// Run one tool call. Returns (result text, is_error, sources used as "table:pid").
pub async fn exec(name: &str, args: &Value, c: &Ctx<'_>) -> (String, bool, Vec<String>) {
    match name {
        SEARCH => {
            let q = args["query"].as_str().unwrap_or("").trim().to_string();
            if q.is_empty() {
                return ("query is required".into(), true, vec![]);
            }
            let Some(eg) = c.eg.clone() else { return ("the library is off".into(), true, vec![]) };
            let k = args["k"].as_u64().map(|k| k.clamp(1, 8) as usize).unwrap_or(c.k);
            let tables = c.tables.to_vec();
            let r = tokio::task::spawn_blocking(move || eg.retrieve(&q, k, &tables.iter().map(|s| s.as_str()).collect::<Vec<_>>())).await;
            match r {
                Ok(Ok(hits)) if !hits.is_empty() => (format_hits(&hits), false, hits.iter().map(|h| format!("{}:{}", h.table, h.pid)).collect()),
                Ok(Ok(_)) => ("no passages found".into(), false, vec![]),
                Ok(Err(e)) => (format!("search failed: {e:#}"), true, vec![]),
                Err(e) => (format!("search failed: {e}"), true, vec![]),
            }
        }
        REMEMBER => {
            let t = args["text"].as_str().unwrap_or("");
            if t.trim().is_empty() {
                return ("text is required".into(), true, vec![]);
            }
            c.mem.add(c.user, c.session, "user", t);
            ("saved".into(), false, vec![])
        }
        RECALL => {
            let notes = c.mem.recall(c.user, c.session, args["query"].as_str().unwrap_or(""), &[], 8);
            (if notes.is_empty() { "no related notes".into() } else { notes.iter().map(|n| format!("- {}", n.text)).collect::<Vec<_>>().join("\n") }, false, vec![])
        }
        other => (format!("unknown tool: {other}"), true, vec![]),
    }
}

/// Tool-call arguments arrive as a JSON string (OpenAI) or, from some servers, as an object.
pub fn args_of(tc: &Value) -> Value {
    match &tc["function"]["arguments"] {
        Value::String(s) => serde_json::from_str(s).unwrap_or(json!({})),
        v @ Value::Object(_) => v.clone(),
        _ => json!({}),
    }
}
