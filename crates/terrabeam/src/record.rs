//! `--record DIR`: every chat exchange written as one JSON line, exactly as the model saw and produced it, for building
//! training data in the same harness the model is served in. Opt-in, local files only; terrabeam never uploads them.
//! Exchanges whose passages come from a health table are not written. User and conversation ids are stored as hashes.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

/// Tables whose content must never leave the machine in any derived form (Tablinum policy).
pub const HEALTH_TABLES: &[&str] = &["health", "healthp", "psych", "fdalabel", "cdcj"];

pub struct Recorder {
    dir: PathBuf,
    lock: Mutex<()>,
}

pub struct Exchange<'a> {
    pub harness: &'a str,
    pub model: &'a str,
    pub user: &'a str,
    pub session: &'a str,
    /// the request body as the client sent it
    pub request: &'a Value,
    /// every message the model saw in its last step, plus what it produced (hidden tool steps included)
    pub trajectory: Vec<Value>,
    pub tools: Option<&'a Value>,
    /// the assistant message the client received
    pub response: Value,
    pub sources: Vec<String>,
    pub finish: Option<String>,
    pub usage: Option<Value>,
    pub steps: usize,
    pub ms: u64,
}

fn hash(s: &str) -> String {
    hex::encode(&Sha256::digest(s.as_bytes())[..8])
}

pub fn is_health(source: &str) -> bool {
    let t = source.split(':').next().unwrap_or("");
    HEALTH_TABLES.contains(&t)
}

impl Recorder {
    pub fn open(dir: PathBuf) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&dir)?;
        Ok(Recorder { dir, lock: Mutex::new(()) })
    }

    /// Returns false when the exchange was skipped (health passages) or could not be written.
    pub fn write(&self, x: Exchange) -> bool {
        if x.sources.iter().any(|s| is_health(s)) {
            return false;
        }
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let day = ts / 86400;
        let row = json!({
            "ts": ts, "harness": x.harness, "model": x.model, "user": hash(x.user), "session": hash(&format!("{}|{}", x.user, x.session)),
            "request": x.request, "trajectory": x.trajectory, "tools": x.tools, "response": x.response, "sources": x.sources,
            "finish": x.finish, "usage": x.usage, "steps": x.steps, "ms": x.ms,
        });
        let _g = self.lock.lock().unwrap();
        let path = self.dir.join(format!("exchanges-{day}.jsonl"));
        let r = std::fs::OpenOptions::new().create(true).append(true).open(&path).and_then(|mut f| writeln!(f, "{row}"));
        if let Err(e) = &r {
            eprintln!("terrabeam: record {}: {e}", path.display());
        }
        r.is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_health_and_hashes_ids() {
        let d = std::env::temp_dir().join(format!("tb-rec-{}", std::process::id()));
        let r = Recorder::open(d.clone()).unwrap();
        let req = json!({"messages": []});
        let ex = |src: Vec<String>| Exchange {
            harness: "h", model: "m", user: "alice", session: "s1", request: &req, trajectory: vec![], tools: None,
            response: json!({}), sources: src, finish: None, usage: None, steps: 1, ms: 1,
        };
        assert!(!r.write(ex(vec!["healthp:12".into()])));
        assert!(r.write(ex(vec!["wiki:1".into()])));
        let body: String = std::fs::read_dir(&d).unwrap().map(|e| std::fs::read_to_string(e.unwrap().path()).unwrap()).collect();
        assert_eq!(body.lines().count(), 1);
        assert!(!body.contains("alice"));
        std::fs::remove_dir_all(d).ok();
    }
}
