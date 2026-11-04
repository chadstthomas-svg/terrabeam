//! Engram retrieval in pure Rust (terrabeam). See DESIGN.md.
//! First stage per table = half static-key vector hits (faiss IVF-SQ8) + half BM25 (when the table has an index), as engram.py.
pub mod bm25;
pub mod embed;
pub mod ivf;
pub mod passages;
pub mod rerank;

use anyhow::{Context, Result};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

pub struct Table {
    pub name: String,
    pub ivf: ivf::IvfIndex,
    pub bm25: Option<bm25::Bm25>,
    pub passages: passages::Passages,
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub table: String,
    pub pid: i64,
    /// first-stage score: inner product for vector hits, BM25 score for lexical hits (not comparable)
    pub first: f32,
    pub lexical: bool,
}

#[derive(Clone, Debug)]
pub struct Hit {
    pub table: String,
    pub pid: i64,
    pub first: f32,
    pub lexical: bool,
    pub score: Option<f32>,
    pub passage: passages::Passage,
}

pub struct Engram {
    pub tables: BTreeMap<String, Table>,
    pub embed: embed::StaticEmbed,
    pub nprobe: usize,
    pub ncand: usize,
    pub rerank: Option<rerank::CrossEncoder>,
    /// skip the cross-encoder when the best vector hit is a near-duplicate of the query (Rig sidecar fast lane); 0 = off
    pub confident_at: f32,
}

/// `<root>` for a table name: the engram root itself for "wiki", else `<root>/tables/<name>`.
pub fn table_root(engram_root: &Path, name: &str) -> PathBuf {
    if name == "wiki" { engram_root.to_path_buf() } else { engram_root.join("tables").join(name) }
}

impl Engram {
    pub fn open(engram_root: &Path, tables: &[&str], embed_dir: &Path, nprobe: usize, ncand: usize) -> Result<Self> {
        let mut out = BTreeMap::new();
        for &t in tables {
            let r = table_root(engram_root, t);
            let ivf = ivf::IvfIndex::open(&r.join("index/engram.index")).with_context(|| format!("table {t}"))?;
            out.insert(
                t.to_string(),
                Table { name: t.to_string(), ivf, bm25: bm25::Bm25::open(&r)?, passages: passages::Passages::open(&r)? },
            );
        }
        let embed = embed::StaticEmbed::open(embed_dir, 256)?;
        Ok(Engram { tables: out, embed, nprobe, ncand, rerank: None, confident_at: 0.0 })
    }

    /// engram.py `candidates`: per table m = ncand / n_tables, of which m/2 from BM25 (de-duplicated against the vector hits).
    pub fn candidates(&self, q: &str, n: usize, tables: &[&str]) -> Result<Vec<Candidate>> {
        let qv = self.embed.embed(q)?;
        let ts: Vec<&Table> = tables.iter().filter_map(|t| self.tables.get(*t)).collect();
        let mut out = Vec::new();
        for t in &ts {
            let m = (n / ts.len()).max(1);
            let m_lex = if t.bm25.is_some() { m / 2 } else { 0 };
            let mut got: Vec<Candidate> = t
                .ivf
                .search(&qv, m - m_lex, self.nprobe)
                .into_iter()
                .filter(|(_, id)| *id >= 0)
                .map(|(s, id)| Candidate { table: t.name.clone(), pid: id, first: s, lexical: false })
                .collect();
            if m_lex > 0 && !q.trim().is_empty() {
                let mut seen: HashSet<i64> = got.iter().map(|c| c.pid).collect();
                for (s, pid) in t.bm25.as_ref().unwrap().search(q, m_lex + got.len())? {
                    if got.len() >= m {
                        break;
                    }
                    if seen.insert(pid) {
                        got.push(Candidate { table: t.name.clone(), pid, first: s, lexical: true });
                    }
                }
            }
            out.extend(got);
        }
        Ok(out)
    }

    pub fn fetch(&self, c: &Candidate) -> Result<Hit> {
        let p = self.tables[&c.table].passages.get(c.pid)?;
        Ok(Hit { table: c.table.clone(), pid: c.pid, first: c.first, lexical: c.lexical, score: None, passage: p })
    }

    /// engram.py `retrieve`: hybrid candidates -> passages -> cross-encoder re-rank against q -> top k.
    pub fn retrieve(&self, q: &str, k: usize, tables: &[&str]) -> Result<Vec<Hit>> {
        let cand = self.candidates(q, self.ncand, tables)?;
        let mut hits = cand.iter().map(|c| self.fetch(c)).collect::<Result<Vec<_>>>()?;
        if let Some(ce) = &self.rerank {
            let best_vec = cand.iter().filter(|c| !c.lexical).map(|c| c.first).fold(f32::MIN, f32::max);
            if self.confident_at > 0.0 && best_vec >= self.confident_at {
                hits.retain(|h| !h.lexical);
                hits.sort_by(|a, b| b.first.total_cmp(&a.first));
            } else if !hits.is_empty() {
                let texts: Vec<&str> = hits.iter().map(|h| h.passage.text.as_str()).collect();
                let sc = ce.score(q, &texts)?;
                for (h, s) in hits.iter_mut().zip(sc) {
                    h.score = Some(s);
                }
                hits.sort_by(|a, b| b.score.unwrap().total_cmp(&a.score.unwrap()));
            }
        }
        hits.truncate(k);
        Ok(hits)
    }
}
