//! cand_probe ROOT TABLES NCAND NPROBE < questions (one per line) -> per query: "table:pid ..." then the first passage title
use std::io::{BufRead, Write};
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let tables: Vec<&str> = a[2].split(',').collect();
    let emb = std::path::PathBuf::from(std::env::var("EMBED_DIR")?);
    let t0 = std::time::Instant::now();
    let eg = engram::Engram::open(std::path::Path::new(&a[1]), &tables, &emb, a[4].parse()?, a[3].parse()?)?;
    eprintln!("open {:?}", t0.elapsed());
    let mut out = std::io::stdout().lock();
    let t1 = std::time::Instant::now(); let mut n = 0;
    for line in std::io::stdin().lock().lines() {
        let q = line?; n += 1;
        let c = eg.candidates(&q, eg.ncand, &tables)?;
        let h = eg.fetch(&c[0])?;
        writeln!(out, "{}\t{}", c.iter().map(|c| format!("{}:{}", c.table, c.pid)).collect::<Vec<_>>().join(" "), h.passage.title)?;
    }
    eprintln!("{n} queries {:?}/q", t1.elapsed() / n.max(1));
    Ok(())
}
