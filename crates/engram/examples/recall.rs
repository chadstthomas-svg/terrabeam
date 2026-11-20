//! recall ROOT EMBED_DIR CE_DIR TASKS.jsonl TABLES > report
//! First-stage + re-rank recall of the gold answer (normalized substring) at 3 and 10, for several candidate configurations,
//! on tablinum-envs tasks (memory tasks skipped). The cross-encoder score is pointwise, so every passage in the union of all
//! configurations is scored once per question and each configuration ranks its own candidates by those scores.
use std::collections::{BTreeMap, HashMap};
use std::io::BufRead;
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

fn norm(s: &str) -> String {
    let f: String = s.nfkd().filter(|c| !unicode_normalization::char::is_combining_mark(*c)).collect::<String>().to_lowercase();
    let c: String = f.chars().filter(|c| *c != ',' && *c != '\'').map(|c| if c.is_alphanumeric() { c } else { ' ' }).collect();
    let w: Vec<&str> = c.split_whitespace().collect();
    let w = match w.first() {
        Some(&"the") | Some(&"a") | Some(&"an") if w.len() > 1 => &w[1..],
        _ => &w[..],
    };
    format!(" {} ", w.join(" "))
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let tables: Vec<String> = a[5].split(',').map(String::from).collect();
    let tn: Vec<&str> = tables.iter().map(|s| s.as_str()).collect();
    let mut eg = engram::Engram::open(Path::new(&a[1]), &tn, Path::new(&a[2]), 128, 48)?;
    let threads = std::env::var("THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(8);
    eg.rerank = Some(engram::rerank::CrossEncoder::open_with(Path::new(&a[3]), true, threads)?);
    let limit: usize = std::env::var("LIMIT").ok().and_then(|t| t.parse().ok()).unwrap_or(usize::MAX);
    let mut tasks = vec![];
    for l in std::io::BufReader::new(std::fs::File::open(&a[4])?).lines() {
        let v: serde_json::Value = serde_json::from_str(&l?)?;
        if v["kind"] != "memory" {
            tasks.push(v);
        }
    }
    tasks.truncate(limit);
    // (name, kind): split ncand | global (vec_per_table, vec_keep, lex_per_table)
    let cfgs: Vec<(&str, i64, usize, usize, usize)> = vec![
        ("split48", 0, 48, 0, 0), ("split192", 0, 192, 0, 0), ("g8-32-l1", 1, 8, 32, 1), ("g16-64-l2", 1, 16, 64, 2),
    ];
    let mut rec: BTreeMap<&str, [usize; 3]> = BTreeMap::new(); // r@3, r@10, total candidates
    let mut fast_hits = [0usize; 2]; // fast lane taken, and gold in its top 3
    let mut by_table: BTreeMap<(String, &str), [usize; 2]> = BTreeMap::new();
    let t0 = std::time::Instant::now();
    for (i, t) in tasks.iter().enumerate() {
        let q = t["question"].as_str().unwrap_or_default();
        let gold = norm(t["gold"].as_str().unwrap_or_default());
        let table = t["table"].as_str().unwrap_or_default().to_string();
        let mut sets: Vec<Vec<engram::Candidate>> = vec![];
        for &(_, kind, x, y, z) in &cfgs {
            sets.push(if kind == 0 { eg.candidates(q, x, &tn)? } else { eg.candidates_global(q, &tn, x, y, z)? });
        }
        let mut uniq: HashMap<(String, i64), usize> = HashMap::new();
        let mut hits = vec![];
        for s in &sets {
            for c in s {
                uniq.entry((c.table.clone(), c.pid)).or_insert_with(|| {
                    hits.push(eg.fetch(c).expect("fetch"));
                    hits.len() - 1
                });
            }
        }
        let texts: Vec<&str> = hits.iter().map(|h| h.passage.text.as_str()).collect();
        let sc = eg.rerank.as_ref().unwrap().score(q, &texts)?;
        let has = |h: &engram::Hit| norm(&h.passage.text).contains(&gold);
        for (ci, s) in sets.iter().enumerate() {
            let mut idx: Vec<usize> = s.iter().map(|c| uniq[&(c.table.clone(), c.pid)]).collect();
            idx.sort_by(|a, b| sc[*b].total_cmp(&sc[*a]));
            let r3 = idx.iter().take(3).any(|&j| has(&hits[j]));
            let r10 = idx.iter().take(10).any(|&j| has(&hits[j]));
            let e = rec.entry(cfgs[ci].0).or_default();
            e[0] += r3 as usize;
            e[1] += r10 as usize;
            e[2] += s.len();
            let bt = by_table.entry((table.clone(), cfgs[ci].0)).or_default();
            bt[0] += r3 as usize;
            bt[1] += 1;
        }
        // the confident fast lane as terrabeam ships it (0.85)
        eg.confident_at = 0.85;
        if let Some(fh) = eg.retrieve_fast(q, 3, &tn)? {
            fast_hits[0] += 1;
            fast_hits[1] += fh.iter().any(|h| has(h)) as usize;
        }
        eg.confident_at = 0.0;
        if (i + 1) % 10 == 0 {
            eprintln!("{}/{} ({:.0}s, {:.1}s/query, union {} passages)", i + 1, tasks.len(), t0.elapsed().as_secs_f32(), t0.elapsed().as_secs_f32() / (i + 1) as f32, hits.len());
        }
    }
    let n = tasks.len().max(1) as f64;
    println!("tasks {} tables {}", tasks.len(), tables.len());
    for (c, [r3, r10, nc]) in &rec {
        println!("{c:12} recall@3 {:5.1}  recall@10 {:5.1}  candidates/query {:6.1}", *r3 as f64 / n * 100.0, *r10 as f64 / n * 100.0, *nc as f64 / n);
    }
    println!("fast lane (confident 0.85): taken {} of {} queries, gold in its top 3: {}", fast_hits[0], tasks.len(), fast_hits[1]);
    println!("\nrecall@3 by source table:");
    let mut tabs: Vec<String> = by_table.keys().map(|(t, _)| t.clone()).collect();
    tabs.dedup();
    for t in tabs {
        let row: Vec<String> = cfgs.iter().map(|c| by_table.get(&(t.clone(), c.0)).map(|[a, b]| format!("{}={:.0}", c.0, *a as f64 / *b as f64 * 100.0)).unwrap_or_default()).collect();
        println!("{t:12} n={:3} {}", by_table.get(&(t.clone(), cfgs[0].0)).map(|x| x[1]).unwrap_or(0), row.join(" "));
    }
    Ok(())
}
