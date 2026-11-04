//! ivf_probe INDEX QUERIES.f32 K NPROBE -> one line per query: "id:score id:score ..." (parity test vs faiss)
use std::io::Write;
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let t0 = std::time::Instant::now();
    let ix = engram::ivf::IvfIndex::open(std::path::Path::new(&a[1]))?;
    eprintln!("open {:?}: d {} nlist {} ntotal {}", t0.elapsed(), ix.d, ix.nlist, ix.ntotal);
    let raw = std::fs::read(&a[2])?;
    let q: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let (k, np): (usize, usize) = (a[3].parse()?, a[4].parse()?);
    let mut out = std::io::stdout().lock();
    let t1 = std::time::Instant::now();
    let nq = q.len() / ix.d;
    for qi in q.chunks_exact(ix.d) {
        let r = ix.search(qi, k, np);
        writeln!(out, "{}", r.iter().map(|(s, i)| format!("{}:{:.4}", i, s)).collect::<Vec<_>>().join(" "))?;
    }
    eprintln!("{} queries in {:?} ({:?}/query)", nq, t1.elapsed(), t1.elapsed() / nq.max(1) as u32);
    Ok(())
}
