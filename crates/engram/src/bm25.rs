//! Lexical half of the hybrid first stage: the tantivy index tools/bm25.py builds in `<table>/bm25/` (tantivy-py 0.26 =
//! tantivy 0.26; field `text` en_stem with frequencies, no positions; stored i64 `pid`). Query = one term per lowercased
//! word, OR-combined (no phrase queries: there are no positions), as engram.py does.

use anyhow::{Context, Result};
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Value};
use tantivy::{Index, IndexReader, TantivyDocument};

pub struct Bm25 {
    index: Index,
    reader: IndexReader,
    text: Field,
    pid: Field,
}

impl Bm25 {
    pub fn open(table_root: &Path) -> Result<Option<Self>> {
        let dir = table_root.join("bm25");
        if !dir.join("meta.json").exists() {
            return Ok(None);
        }
        let index = Index::open_in_dir(&dir).with_context(|| format!("tantivy {}", dir.display()))?;
        let reader = index.reader()?;
        let s = index.schema();
        Ok(Some(Bm25 { text: s.get_field("text")?, pid: s.get_field("pid")?, reader, index }))
    }

    pub fn search(&self, q: &str, n: usize) -> Result<Vec<(f32, i64)>> {
        let terms: Vec<String> = q
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_string())
            .collect();
        let qs = if terms.is_empty() { "the".to_string() } else { terms.join(" ") };
        let parser = QueryParser::for_index(&self.index, vec![self.text]);
        let (query, _errs) = parser.parse_query_lenient(&qs);
        let se = self.reader.searcher();
        let mut out = Vec::new();
        for (score, addr) in se.search(&query, &TopDocs::with_limit(n).order_by_score())? {
            let d: TantivyDocument = se.doc(addr)?;
            if let Some(p) = d.get_first(self.pid).and_then(|v| v.as_i64()) {
                out.push((score, p));
            }
        }
        Ok(out)
    }
}
