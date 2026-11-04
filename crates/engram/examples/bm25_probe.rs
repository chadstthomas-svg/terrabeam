use std::io::BufRead;
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let b = engram::bm25::Bm25::open(std::path::Path::new(&a[1]))?.unwrap();
    for l in std::io::stdin().lock().lines() {
        let r = b.search(&l?, 12)?;
        println!("{}", r.iter().map(|(s, p)| format!("{p}:{s:.4}")).collect::<Vec<_>>().join(" "));
    }
    Ok(())
}
