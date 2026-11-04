//! ce_probe CE_DIR < jsonl {"q":..., "p":[...]} -> one line per item: space-separated scores (parity vs torch)
use std::io::BufRead;
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let mut ce = engram::rerank::CrossEncoder::open_with(std::path::Path::new(&a[1]), std::env::var("CE_BACKEND").map(|b| b != "candle").unwrap_or(true), std::env::var("CE_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(4))?; if let Ok(b) = std::env::var("CE_BATCH") { ce.batch = b.parse()?; } eprintln!("backend {}", ce.backend_name());
    let t = std::time::Instant::now(); let mut n = 0; let mut np = 0;
    for l in std::io::stdin().lock().lines() {
        let v: serde_json::Value = serde_json::from_str(&l?)?;
        let ps: Vec<&str> = v["p"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
        let s = ce.score(v["q"].as_str().unwrap(), &ps)?; n += 1; np += ps.len();
        println!("{}", s.iter().map(|x| format!("{x:.5}")).collect::<Vec<_>>().join(" "));
    }
    eprintln!("{n} queries, {np} pairs, {:?}/query", t.elapsed() / n.max(1) as u32);
    Ok(())
}
