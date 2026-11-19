//! Rebuild the assistant message from an upstream chat-completions response: a JSON body or an SSE stream of deltas.
//! Content, reasoning (`reasoning_content`, or `reasoning` as some servers name it), tool calls (merged by index), the finish
//! reason and usage. Used by reasoning restore, recording, memory and the library tool loop.

use serde_json::{json, Map, Value};

#[derive(Clone, Debug, Default)]
pub struct Assembled {
    pub content: String,
    pub reasoning: String,
    /// OpenAI tool calls: {"id", "type": "function", "function": {"name", "arguments"}}
    pub tool_calls: Vec<Value>,
    pub finish: Option<String>,
    pub usage: Option<Value>,
    pub model: Option<String>,
    /// a JSON or SSE body was seen and parsed
    pub parsed: bool,
}

impl Assembled {
    /// One SSE event payload (the text after `data:`), or a whole JSON body.
    pub fn feed_event(&mut self, v: &Value) {
        self.parsed = true;
        if let Some(m) = v.get("model").and_then(|m| m.as_str()) {
            self.model.get_or_insert_with(|| m.to_string());
        }
        if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(u.clone());
        }
        let Some(ch) = v.get("choices").and_then(|c| c.get(0)) else { return };
        if let Some(f) = ch.get("finish_reason").and_then(|f| f.as_str()) {
            self.finish = Some(f.to_string());
        }
        if let Some(d) = ch.get("delta") {
            self.add_delta(d);
        } else if let Some(m) = ch.get("message") {
            self.content = m.get("content").and_then(|c| c.as_str()).unwrap_or_default().to_string();
            self.reasoning = reasoning_of(m).unwrap_or_default();
            self.tool_calls = m.get("tool_calls").and_then(|t| t.as_array()).cloned().unwrap_or_default();
        }
    }

    fn add_delta(&mut self, d: &Value) {
        if let Some(t) = d.get("content").and_then(|c| c.as_str()) {
            self.content.push_str(t);
        }
        if let Some(t) = d.get("reasoning_content").or_else(|| d.get("reasoning")).and_then(|c| c.as_str()) {
            self.reasoning.push_str(t);
        }
        for tc in d.get("tool_calls").and_then(|t| t.as_array()).into_iter().flatten() {
            let idx = tc.get("index").and_then(|i| i.as_u64()).map(|i| i as usize).unwrap_or(self.tool_calls.len());
            while self.tool_calls.len() <= idx {
                self.tool_calls.push(json!({"id": "", "type": "function", "function": {"name": "", "arguments": ""}}));
            }
            let slot = &mut self.tool_calls[idx];
            if let Some(id) = tc.get("id").and_then(|x| x.as_str()).filter(|s| !s.is_empty()) {
                slot["id"] = json!(id);
            }
            if let Some(f) = tc.get("function") {
                if let Some(n) = f.get("name").and_then(|x| x.as_str()) {
                    let cur = slot["function"]["name"].as_str().unwrap_or_default().to_string();
                    slot["function"]["name"] = json!(cur + n);
                }
                if let Some(a) = f.get("arguments").and_then(|x| x.as_str()) {
                    let cur = slot["function"]["arguments"].as_str().unwrap_or_default().to_string();
                    slot["function"]["arguments"] = json!(cur + a);
                }
            }
        }
    }

    /// The assistant message as it goes back into the conversation (reasoning kept as `reasoning_content`).
    pub fn message(&self) -> Value {
        let mut m = Map::new();
        m.insert("role".into(), json!("assistant"));
        m.insert("content".into(), if self.content.is_empty() && !self.tool_calls.is_empty() { Value::Null } else { json!(self.content) });
        if !self.reasoning.is_empty() {
            m.insert("reasoning_content".into(), json!(self.reasoning));
        }
        if !self.tool_calls.is_empty() {
            m.insert("tool_calls".into(), json!(self.tool_calls));
        }
        Value::Object(m)
    }
}

pub fn reasoning_of(m: &Value) -> Option<String> {
    m.get("reasoning_content").or_else(|| m.get("reasoning")).and_then(|r| r.as_str()).filter(|s| !s.is_empty()).map(String::from)
}

/// Splits a byte stream into SSE lines and feeds `data:` events to an [`Assembled`].
#[derive(Default)]
pub struct SseReader {
    line: Vec<u8>,
    pub raw: Vec<u8>,
    pub sse: bool,
    pub asm: Assembled,
}

impl SseReader {
    /// Feed a chunk; returns the complete `data:` payloads seen (parsed), for callers that rewrite the stream.
    pub fn feed(&mut self, c: &[u8]) -> Vec<String> {
        if self.raw.len() < (16 << 20) {
            self.raw.extend_from_slice(c);
        }
        self.line.extend_from_slice(c);
        let mut out = vec![];
        while let Some(i) = self.line.iter().position(|&x| x == b'\n') {
            let l: Vec<u8> = self.line.drain(..=i).collect();
            let l = String::from_utf8_lossy(&l);
            if let Some(d) = l.trim().strip_prefix("data:") {
                self.sse = true;
                let d = d.trim().to_string();
                if let Ok(v) = serde_json::from_str::<Value>(&d) {
                    self.asm.feed_event(&v);
                }
                out.push(d);
            }
        }
        out
    }

    /// End of body: a non-SSE body is parsed as one JSON response.
    pub fn finish(&mut self) -> &Assembled {
        if !self.sse && !self.asm.parsed {
            if let Ok(v) = serde_json::from_slice::<Value>(&self.raw) {
                self.asm.feed_event(&v);
            }
        }
        &self.asm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_stream_deltas() {
        let mut r = SseReader::default();
        let ev = [
            r#"{"model":"m","choices":[{"delta":{"role":"assistant","reasoning_content":"think "}}]}"#,
            r#"{"choices":[{"delta":{"reasoning_content":"more"}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"search_","arguments":"{\"q"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"library","arguments":"uery\":\"x\"}"}}]}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        ];
        let body: String = ev.iter().map(|e| format!("data: {e}\n\n")).collect::<String>() + "data: [DONE]\n\n";
        // split mid-line to check buffering
        let (a, b) = body.as_bytes().split_at(37);
        r.feed(a);
        r.feed(b);
        let m = r.finish().message();
        assert_eq!(m["reasoning_content"], "think more");
        assert_eq!(m["tool_calls"][0]["id"], "c1");
        assert_eq!(m["tool_calls"][0]["function"]["name"], "search_library");
        assert_eq!(m["tool_calls"][0]["function"]["arguments"], "{\"query\":\"x\"}");
        assert!(m["content"].is_null());
        assert_eq!(r.asm.finish.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn reads_a_json_body() {
        let mut r = SseReader::default();
        r.feed(br#"{"choices":[{"message":{"role":"assistant","content":"hi","reasoning_content":"r"},"finish_reason":"stop"}],"usage":{"total_tokens":3}}"#);
        let a = r.finish();
        assert_eq!(a.content, "hi");
        assert_eq!(a.reasoning, "r");
        assert_eq!(a.usage.as_ref().unwrap()["total_tokens"], 3);
    }
}
