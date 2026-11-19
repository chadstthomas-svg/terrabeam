//! Tasks run through the harness itself: the same proxy, injection, tools and history users get, so a model can be scored,
//! calibrated and trained in exactly the setup it is served in.
//!
//! A task (one JSON line):
//!   {"id", "kind": "fact"|"multihop"|"memory", "split": "train"|"eval", "question", "gold", "aliases": [..],
//!    "passages": {"oracle": [..], "distract": [..]}, "prior": [[messages of earlier conversations]] (memory tasks)}
//! Controls (which variant of the task the model sees):
//!   live     - the harness as served: library retrieval (+ the library tool when on)
//!   oracle   - the passages that hold the answer, in the trained injection layout (can the model read?)
//!   distract - related passages that do NOT hold the answer (does it say it doesn't know, or guess?)
//!   closed   - no library at all (what it knows on its own)
//! Grading: correct +1, abstain 0, incorrect -1 (the "index" the Beam cards report).
//!
//! HTTP (on every proxy): POST /env/render {task, control} -> the request to send to /v1/chat/completions;
//! POST /env/grade {task, answer} -> {verdict, reward}. CLI: `terrabeam env run --tasks t.jsonl --upstream URL --out r.jsonl`.

use crate::proxy::Px;
use crate::{inject, memory, run};
use anyhow::{Context, Result};
use axum::extract::State;
use axum::Json;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use unicode_normalization::UnicodeNormalization;

pub const CONTROLS: &[&str] = &["live", "oracle", "distract", "closed"];

/// lowercase, accents folded, punctuation to spaces, single spaces, padded with spaces for whole-word matching
pub fn norm(s: &str) -> String {
    let folded: String = s.nfkd().filter(|c| !unicode_normalization::char::is_combining_mark(*c)).collect::<String>().to_lowercase();
    let cleaned: String = folded.chars().map(|c| if c.is_alphanumeric() { c } else if c == ',' || c == '\'' { '\u{0}' } else { ' ' }).filter(|c| *c != '\u{0}').collect();
    let words: Vec<&str> = cleaned.split_whitespace().collect();
    let words: Vec<&str> = match words.first() {
        Some(&"the") | Some(&"a") | Some(&"an") if words.len() > 1 => words[1..].to_vec(),
        _ => words,
    };
    format!(" {} ", words.join(" "))
}

const ABSTAIN: &[&str] = &[
    " i don t know", " i dont know", " i do not know", " don t know", " not sure", " cannot determine", " can t determine", " cannot answer",
    " can t answer", " unable to", " no information", " not stated", " does not say", " doesn t say", " do not have", " don t have",
    " not mentioned", " not mention", " no mention", " not specified", " not specify", " not provided", " no passage", " none of the passages", " not in the passages",
];

fn strip_think(s: &str) -> String {
    match (s.find("<think>"), s.rfind("</think>")) {
        (Some(a), Some(b)) if b > a => format!("{}{}", &s[..a], &s[b + 8..]),
        (None, Some(b)) => s[b + 8..].to_string(),
        _ => s.to_string(),
    }
}

/// (verdict, reward)
pub fn grade(task: &Value, answer: &str) -> (&'static str, f64) {
    let a = norm(&strip_think(answer));
    if a.trim().is_empty() {
        return ("abstain", 0.0);
    }
    let mut golds: Vec<String> = vec![task["gold"].as_str().unwrap_or_default().to_string()];
    golds.extend(task["aliases"].as_array().into_iter().flatten().filter_map(|x| x.as_str().map(String::from)));
    if golds.iter().map(|g| norm(g)).any(|g| !g.trim().is_empty() && a.contains(&g)) {
        return ("correct", 1.0);
    }
    // "don t know" etc. after normalisation (apostrophes dropped: "don't" -> "dont"; both spellings listed)
    let a2 = a.replace(" dont ", " don t ").replace(" cant ", " can t ").replace(" doesnt ", " doesn t ");
    if ABSTAIN.iter().any(|p| a2.contains(p)) {
        return ("abstain", 0.0);
    }
    ("incorrect", -1.0)
}

fn question_turn(px: &Px, task: &Value, control: &str) -> Value {
    let q = task["question"].as_str().unwrap_or_default();
    let ps = |k: &str| -> Vec<String> { task["passages"][k].as_array().into_iter().flatten().filter_map(|p| p.as_str().map(String::from)).collect() };
    let text = match control {
        "oracle" | "distract" => {
            let p = ps(control);
            let srcs: Vec<inject::Src> = p.iter().map(|t| inject::Src { table: "env", title: "", text: t }).collect();
            inject::render(px.format, q, &srcs, &inject::new_nonce())
        }
        _ => q.to_string(),
    };
    json!({"role": "user", "content": text})
}

/// The request body (and the conversations to replay first, for memory tasks) for one task variant.
pub fn render(px: &Px, task: &Value, control: &str) -> Value {
    let live = control == "live";
    let body = json!({
        "messages": [question_turn(px, task, control)],
        "engram": live, "library_tool": live && px.library_tool, "memory": task["kind"] == "memory",
    });
    json!({"body": body, "prior": task.get("prior").cloned().unwrap_or(json!([]))})
}

pub async fn render_h(State(px): State<Arc<Px>>, Json(r): Json<Value>) -> Json<Value> {
    let control = r["control"].as_str().unwrap_or("live");
    Json(render(&px, &r["task"], control))
}

pub async fn grade_h(Json(r): Json<Value>) -> Json<Value> {
    let (verdict, reward) = grade(&r["task"], r["answer"].as_str().unwrap_or_default());
    Json(json!({"verdict": verdict, "reward": reward}))
}

#[derive(clap::Args, Clone, Debug)]
pub struct EnvRunArgs {
    /// task file (JSON lines)
    #[arg(long)]
    pub tasks: PathBuf,
    /// results file (JSON lines, one per attempt) ; a summary goes to <out>.summary.json
    #[arg(long)]
    pub out: PathBuf,
    /// the model server (OpenAI-compatible), e.g. a llama-server started with the model's own settings
    #[arg(long)]
    pub upstream: String,
    #[arg(long, value_delimiter = ',', default_value = "live,oracle,distract,closed")]
    pub controls: Vec<String>,
    #[arg(long, default_value_t = 1)]
    pub attempts: usize,
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,
    /// only tasks of this split (train / eval)
    #[arg(long)]
    pub split: Option<String>,
    /// at most this many tasks
    #[arg(long)]
    pub limit: Option<usize>,
    /// extra fields merged into every request (sampling, max_tokens, chat_template_kwargs ...), as JSON
    #[arg(long, default_value = "{}")]
    pub request: String,
    #[command(flatten)]
    pub lib: run::LibArgs,
    #[command(flatten)]
    pub harness: run::HarnessArgs,
}

fn merge(dst: &mut Value, src: &Value) {
    if let (Some(d), Some(s)) = (dst.as_object_mut(), src.as_object()) {
        for (k, v) in s {
            d.insert(k.clone(), v.clone());
        }
    }
}

pub async fn env_run(a: EnvRunArgs) -> Result<()> {
    let extra: Value = serde_json::from_str(&a.request).context("--request is not JSON")?;
    let mut tasks: Vec<Value> = vec![];
    for l in std::io::BufReader::new(std::fs::File::open(&a.tasks).with_context(|| a.tasks.display().to_string())?).lines() {
        let l = l?;
        if l.trim().is_empty() {
            continue;
        }
        let t: Value = serde_json::from_str(&l)?;
        if a.split.as_deref().is_some_and(|s| t["split"].as_str() != Some(s)) {
            continue;
        }
        tasks.push(t);
    }
    if let Some(n) = a.limit {
        tasks.truncate(n);
    }
    let mut lib = a.lib.clone();
    if tasks.iter().any(|t| t["kind"] == "memory") && lib.memory == memory::Mode::Off {
        lib.memory = memory::Mode::LongTerm;
    }
    // the env's memory lives in its own home so it never touches the user's notes
    let mem_home = std::env::temp_dir().join(format!("terrabeam-env-{}", std::process::id()));
    let (px, addr) = run::start_proxy_for_env(&a.upstream, &lib, &a.harness, &mem_home).await?;
    eprintln!("terrabeam env: {} tasks x {:?} x {} attempts via {} (harness {})", tasks.len(), a.controls, a.attempts, a.upstream, px.harness);
    let http = reqwest::Client::new();
    let base = format!("http://{addr}");
    // retrieval recall for the live control: is the gold in the top 3 / top 10 the library returns for the question?
    let mut recall: BTreeMap<String, (bool, bool)> = BTreeMap::new();
    if a.controls.iter().any(|c| c == "live") && px.eg.is_some() {
        for t in tasks.iter().filter(|t| t["kind"] != "memory") {
            let q = t["question"].as_str().unwrap_or_default();
            let r: Value = http.get(format!("{base}/engram/search")).query(&[("q", q), ("k", "10")]).send().await?.json().await.unwrap_or(json!({}));
            let hits: Vec<String> = r["hits"].as_array().into_iter().flatten().map(|h| norm(h["text"].as_str().unwrap_or_default())).collect();
            let g = norm(t["gold"].as_str().unwrap_or_default());
            let has = |n: usize| hits.iter().take(n).any(|h| h.contains(&g));
            recall.insert(t["id"].as_str().unwrap_or_default().to_string(), (has(3), has(10)));
        }
    }
    let mut jobs = vec![];
    for t in &tasks {
        for c in &a.controls {
            if !CONTROLS.contains(&c.as_str()) {
                anyhow::bail!("unknown control {c} (have {CONTROLS:?})");
            }
            if (c.as_str() == "oracle" || c.as_str() == "distract") && t["passages"][c.as_str()].as_array().is_none_or(|p| p.is_empty()) {
                continue;
            }
            for n in 0..a.attempts {
                jobs.push((t.clone(), c.clone(), n));
            }
        }
    }
    let total = jobs.len();
    let out = std::sync::Mutex::new(std::io::BufWriter::new(std::fs::File::create(&a.out)?));
    let results = std::sync::Mutex::new(Vec::<Value>::new());
    let done = std::sync::atomic::AtomicUsize::new(0);
    let t0 = std::time::Instant::now();
    futures_util::stream::iter(jobs)
        .for_each_concurrent(a.concurrency.max(1), |(t, c, n)| {
            let (http, base, px, extra, out, results, done, recall) = (&http, &base, &px, &extra, &out, &results, &done, &recall);
            async move {
                let id = t["id"].as_str().unwrap_or_default().to_string();
                let user = format!("env-{id}-{c}-{n}");
                let r = render(px, &t, &c);
                let started = std::time::Instant::now();
                // memory tasks: replay the earlier conversations first, each in its own session
                for (i, conv) in r["prior"].as_array().into_iter().flatten().enumerate() {
                    let mut b = json!({"messages": conv, "engram": false, "library_tool": false, "memory": true, "max_tokens": 64});
                    merge(&mut b, &json!({"chat_template_kwargs": extra.get("chat_template_kwargs").cloned().unwrap_or(json!({}))}));
                    let _ = http.post(format!("{base}/v1/chat/completions")).header("x-user-id", &user).header("x-session-id", format!("prior{i}")).json(&b).send().await;
                }
                let mut body = r["body"].clone();
                merge(&mut body, extra);
                body["stream"] = json!(false);
                let resp = http.post(format!("{base}/v1/chat/completions")).header("x-user-id", &user).header("x-session-id", "q").json(&body).send().await;
                let (answer, reasoning, finish, usage, sources, err) = match resp {
                    Ok(r) => {
                        let src = r.headers().get("x-engram-sources").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
                        match r.json::<Value>().await {
                            Ok(v) => {
                                let m = &v["choices"][0]["message"];
                                (m["content"].as_str().unwrap_or_default().to_string(), crate::assemble::reasoning_of(m).unwrap_or_default(),
                                    v["choices"][0]["finish_reason"].as_str().unwrap_or_default().to_string(), v["usage"].clone(), src,
                                    v.get("error").map(|e| e.to_string()))
                            }
                            Err(e) => (String::new(), String::new(), String::new(), Value::Null, src, Some(e.to_string())),
                        }
                    }
                    Err(e) => (String::new(), String::new(), String::new(), Value::Null, String::new(), Some(e.to_string())),
                };
                let (verdict, reward) = if err.is_some() { ("error", 0.0) } else { grade(&t, &answer) };
                let mut row = json!({
                    "id": id, "kind": t["kind"], "split": t["split"], "table": t["table"], "control": c, "attempt": n, "verdict": verdict,
                    "reward": reward, "answer": answer, "reasoning_chars": reasoning.chars().count(), "finish": finish, "usage": usage,
                    "sources": sources, "ms": started.elapsed().as_millis() as u64, "error": err, "harness": px.harness,
                });
                if c.as_str() == "live" {
                    if let Some((r3, r10)) = recall.get(&id) {
                        row["gold_in_top3"] = json!(r3);
                        row["gold_in_top10"] = json!(r10);
                    }
                }
                px.mem.delete(&user, None);
                writeln!(out.lock().unwrap(), "{row}").ok();
                results.lock().unwrap().push(row);
                let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if d % 20 == 0 || d == total {
                    let el = t0.elapsed().as_secs_f64();
                    eprintln!("terrabeam env: {d}/{total} ({:.1} min, ~{:.0} min left)", el / 60.0, el / d as f64 * (total - d) as f64 / 60.0);
                }
            }
        })
        .await;
    out.lock().unwrap().flush()?;
    let rows = results.into_inner().unwrap();
    let s = summarize(&rows, &px.harness);
    let sp = PathBuf::from(format!("{}.summary.json", a.out.display()));
    std::fs::write(&sp, serde_json::to_string_pretty(&s)?)?;
    println!("{}", serde_json::to_string_pretty(&s["controls"])?);
    eprintln!("terrabeam env: results {} ; summary {}", a.out.display(), sp.display());
    std::fs::remove_dir_all(&mem_home).ok();
    Ok(())
}

/// Per control: correct / abstain / incorrect rates and the index; per task: solve rate per control (calibration).
pub fn summarize(rows: &[Value], harness: &str) -> Value {
    let mut by: BTreeMap<String, [usize; 4]> = BTreeMap::new();
    let mut per_task: BTreeMap<String, BTreeMap<String, (usize, usize)>> = BTreeMap::new();
    let (mut r3, mut r10, mut nlive) = (0usize, 0usize, 0usize);
    let mut seen_live = std::collections::HashSet::new();
    for r in rows {
        let c = r["control"].as_str().unwrap_or("").to_string();
        let e = by.entry(c.clone()).or_default();
        match r["verdict"].as_str() {
            Some("correct") => e[0] += 1,
            Some("abstain") => e[1] += 1,
            Some("incorrect") => e[2] += 1,
            _ => e[3] += 1,
        }
        let id = r["id"].as_str().unwrap_or("").to_string();
        let pt = per_task.entry(id.clone()).or_default().entry(c.clone()).or_default();
        pt.1 += 1;
        if r["verdict"] == "correct" {
            pt.0 += 1;
        }
        if c == "live" && r.get("gold_in_top3").is_some() && seen_live.insert(id) {
            nlive += 1;
            r3 += r["gold_in_top3"].as_bool().unwrap_or(false) as usize;
            r10 += r["gold_in_top10"].as_bool().unwrap_or(false) as usize;
        }
    }
    let controls: BTreeMap<String, Value> = by
        .iter()
        .map(|(c, [ok, ab, bad, er])| {
            let n = (ok + ab + bad).max(1) as f64;
            (c.clone(), json!({"n": ok + ab + bad, "errors": er, "correct": *ok as f64 / n * 100.0, "abstain": *ab as f64 / n * 100.0,
                "incorrect": *bad as f64 / n * 100.0, "index": (*ok as f64 - *bad as f64) / n * 100.0}))
        })
        .collect();
    let calib: BTreeMap<String, Value> = per_task
        .iter()
        .map(|(id, m)| (id.clone(), Value::Object(m.iter().map(|(c, (s, n))| (c.clone(), json!({"solved": s, "attempts": n}))).collect())))
        .collect();
    json!({"harness": harness, "controls": controls,
        "retrieval": {"tasks": nlive, "gold_in_top3": r3 as f64 / nlive.max(1) as f64 * 100.0, "gold_in_top10": r10 as f64 / nlive.max(1) as f64 * 100.0},
        "calibration": calib})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grades_with_normalisation_and_abstain() {
        let t = json!({"gold": "Émile Zola", "aliases": ["Zola"]});
        assert_eq!(grade(&t, "It was emile zola.").0, "correct");
        assert_eq!(grade(&t, "<think>maybe Hugo</think>The author is Zola").0, "correct");
        assert_eq!(grade(&t, "I don't know.").0, "abstain");
        assert_eq!(grade(&t, "The passages do not mention it.").0, "abstain");
        assert_eq!(grade(&t, "Victor Hugo").0, "incorrect");
        let n = json!({"gold": "1,250"});
        assert_eq!(grade(&n, "about 1250 people").0, "correct");
        // whole words only
        let y = json!({"gold": "Ra"});
        assert_eq!(grade(&y, "Ramses II").0, "incorrect");
    }

    #[test]
    fn summary_counts_index_and_calibration() {
        let rows = vec![
            json!({"id": "a", "control": "live", "verdict": "correct", "gold_in_top3": true, "gold_in_top10": true}),
            json!({"id": "a", "control": "live", "verdict": "incorrect", "gold_in_top3": true, "gold_in_top10": true}),
            json!({"id": "a", "control": "oracle", "verdict": "correct"}),
            json!({"id": "b", "control": "live", "verdict": "abstain", "gold_in_top3": false, "gold_in_top10": true}),
        ];
        let s = summarize(&rows, "h");
        assert_eq!(s["controls"]["live"]["n"], 3);
        assert!((s["controls"]["live"]["index"].as_f64().unwrap() - 0.0).abs() < 1e-9);
        assert_eq!(s["calibration"]["a"]["live"]["solved"], 1);
        assert_eq!(s["calibration"]["a"]["live"]["attempts"], 2);
        assert!((s["retrieval"]["gold_in_top3"].as_f64().unwrap() - 50.0).abs() < 1e-9);
    }
}
