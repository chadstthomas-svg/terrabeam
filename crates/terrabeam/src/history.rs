//! Canonical history: what the model actually produced, put back when a client sends a trimmed copy.
//!
//! Many clients drop `reasoning_content` from earlier assistant turns, and none of them can know about the library tool steps
//! terrabeam ran on their behalf. Beam models are trained with preserved thinking (Ling-3.0-tiny's template keeps earlier
//! reasoning), and without it multi-turn agent work collapses (Moon-Beam tau2 airline: 0.711 with reasoning kept, 0.045 as
//! clients send it). For every answer the proxy stores the canonical messages (hidden library-tool steps + the final assistant
//! message with its reasoning) under keys taken from what the client received: each tool-call id and a hash of the visible
//! content. On the next request every assistant message that matches a key is replaced by its canonical messages.
//! In memory only, per user + conversation, bounded.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

const CAP: usize = 4096;

/// insertion-ordered map with a size cap: (entries, keys oldest first)
pub type Lru<V> = Mutex<(HashMap<String, V>, VecDeque<String>)>;

pub struct History {
    map: Lru<Vec<Value>>,
}

impl Default for History {
    fn default() -> Self {
        History { map: Mutex::new((HashMap::new(), VecDeque::new())) }
    }
}

fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn h(s: &str) -> String {
    hex::encode(&Sha256::digest(s.as_bytes())[..12])
}

fn text_of(m: &Value) -> String {
    match m.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(p)) => p.iter().filter_map(|x| x.get("text").and_then(|t| t.as_str())).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

/// Keys of an assistant message as a client would send it back.
pub fn keys(scope: &str, m: &Value) -> Vec<String> {
    let mut k = vec![];
    for tc in m.get("tool_calls").and_then(|t| t.as_array()).into_iter().flatten() {
        if let Some(id) = tc.get("id").and_then(|i| i.as_str()).filter(|s| !s.is_empty()) {
            k.push(format!("{scope}|id|{id}"));
        }
    }
    let c = norm(&text_of(m));
    if !c.is_empty() {
        k.push(format!("{scope}|c|{}", h(&c)));
    }
    k
}

impl History {
    /// `visible` = the assistant message the client received; `canonical` = the messages that really happened (hidden tool
    /// steps first, the final assistant message with reasoning last).
    pub fn store(&self, scope: &str, visible: &Value, canonical: Vec<Value>) {
        let ks = keys(scope, visible);
        if ks.is_empty() || canonical.is_empty() {
            return;
        }
        let mut g = self.map.lock().unwrap();
        for k in ks {
            if g.0.insert(k.clone(), canonical.clone()).is_none() {
                g.1.push_back(k);
            }
        }
        while g.1.len() > CAP {
            if let Some(old) = g.1.pop_front() {
                g.0.remove(&old);
            }
        }
    }

    /// Replace each matching assistant message with its canonical messages. Returns how many were restored.
    pub fn restore(&self, scope: &str, msgs: &mut Vec<Value>) -> usize {
        let g = self.map.lock().unwrap();
        let mut out = Vec::with_capacity(msgs.len());
        let mut n = 0;
        for m in msgs.drain(..) {
            let is_asst = m.get("role").and_then(|r| r.as_str()) == Some("assistant");
            let found = if is_asst { keys(scope, &m).iter().find_map(|k| g.0.get(k)).cloned() } else { None };
            match found {
                Some(canon) => {
                    n += 1;
                    out.extend(canon);
                }
                None => out.push(m),
            }
        }
        *msgs = out;
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn restores_reasoning_and_hidden_steps() {
        let hst = History::default();
        let visible = json!({"role": "assistant", "content": "Paris."});
        let canon = vec![
            json!({"role": "assistant", "content": null, "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "search_library", "arguments": "{}"}}]}),
            json!({"role": "tool", "tool_call_id": "t1", "content": "[1] ..."}),
            json!({"role": "assistant", "content": "Paris.", "reasoning_content": "The passage says Paris."}),
        ];
        hst.store("u|s", &visible, canon.clone());
        // client sends the answer back with different whitespace and no reasoning
        let mut msgs = vec![json!({"role": "user", "content": "Capital?"}), json!({"role": "assistant", "content": " Paris. "}), json!({"role": "user", "content": "And Spain?"})];
        assert_eq!(hst.restore("u|s", &mut msgs), 1);
        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[3]["reasoning_content"], "The passage says Paris.");
        // other conversations never see it
        let mut other = vec![json!({"role": "assistant", "content": "Paris."})];
        assert_eq!(hst.restore("u|other", &mut other), 0);
    }

    #[test]
    fn matches_tool_calls_by_id() {
        let hst = History::default();
        let v = json!({"role": "assistant", "content": null, "tool_calls": [{"id": "abc", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]});
        let mut c = v.clone();
        c["reasoning_content"] = json!("need weather");
        hst.store("s", &v, vec![c]);
        let mut msgs = vec![json!({"role": "assistant", "tool_calls": [{"id": "abc", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]})];
        assert_eq!(hst.restore("s", &mut msgs), 1);
        assert_eq!(msgs[0]["reasoning_content"], "need weather");
    }
}
