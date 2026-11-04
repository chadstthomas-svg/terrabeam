//! Passage store: `<table>/passages/<a>_<b>.parquet`, pid = file_idx << 32 | row. A file is keyed by `b + 100 * a`
//! (the engram.py convention) and, when its stored `pid` column says otherwise (pes2o_full.py: 900000 + idx), by that too.
//! Rows are read one row group at a time (4096 rows) with a small LRU cache, columns title / section / text only.

use anyhow::{Context, Result};
use arrow_array::{Array, RecordBatch, StringArray};
use lru::LruCache;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::statistics::Statistics;
use std::collections::HashMap;
use std::fs::File;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Default)]
pub struct Passage {
    pub title: String,
    pub section: String,
    pub text: String,
}

struct PFile {
    path: PathBuf,
    meta: Arc<ParquetMetaData>,
    starts: Vec<usize>, // row-group start rows (+ total at the end)
}

type Group = Arc<Vec<Passage>>;

pub struct Passages {
    files: HashMap<i64, Arc<PFile>>,
    cache: Mutex<LruCache<(i64, usize), Group>>,
}

impl Passages {
    pub fn open(table_root: &Path) -> Result<Self> {
        let mut files = HashMap::new();
        let dir = table_root.join("passages");
        for e in std::fs::read_dir(&dir).with_context(|| format!("read {}", dir.display()))? {
            let p = e?.path();
            if p.extension().and_then(|x| x.to_str()) != Some("parquet") {
                continue;
            }
            let stem = p.file_stem().unwrap().to_string_lossy().to_string();
            let Some((a, b)) = stem.split_once('_') else { continue };
            let (Ok(a), Ok(b)) = (a.parse::<i64>(), b.parse::<i64>()) else { continue };
            let r = SerializedFileReader::new(File::open(&p)?).with_context(|| format!("parquet {}", p.display()))?;
            let meta = Arc::new(r.metadata().clone());
            let mut starts = vec![0usize];
            for rg in meta.row_groups() {
                starts.push(starts.last().unwrap() + rg.num_rows() as usize);
            }
            // stored pid file index from the row-group-0 statistics, when present
            let stored = meta.file_metadata().schema_descr().columns().iter().position(|c| c.name() == "pid").and_then(|ci| {
                meta.row_groups().first().and_then(|rg| match rg.column(ci).statistics() {
                    Some(Statistics::Int64(s)) => s.min_opt().map(|m| *m >> 32),
                    _ => None,
                })
            });
            let f = Arc::new(PFile { path: p.clone(), meta, starts });
            files.insert(b + 100 * a, f.clone());
            if let Some(k) = stored {
                files.entry(k).or_insert(f);
            }
        }
        Ok(Passages { files, cache: Mutex::new(LruCache::new(NonZeroUsize::new(512).unwrap())) })
    }

    pub fn len_files(&self) -> usize {
        self.files.len()
    }

    pub fn get(&self, pid: i64) -> Result<Passage> {
        let (fi, n) = (pid >> 32, (pid & 0xFFFF_FFFF) as usize);
        let f = self.files.get(&fi).with_context(|| format!("pid {pid}: no passage file with index {fi}"))?;
        let rg = f.starts.partition_point(|&s| s <= n) - 1;
        let key = (fi, rg);
        if let Some(g) = self.cache.lock().unwrap().get(&key) {
            return Ok(g[n - f.starts[rg]].clone());
        }
        let g = Arc::new(read_group(f, rg)?);
        let out = g[n - f.starts[rg]].clone();
        self.cache.lock().unwrap().put(key, g);
        Ok(out)
    }
}

fn read_group(f: &PFile, rg: usize) -> Result<Vec<Passage>> {
    let b = ParquetRecordBatchReaderBuilder::try_new(File::open(&f.path)?)?;
    let schema = f.meta.file_metadata().schema_descr();
    let want = ["title", "section", "text"];
    let idx: Vec<usize> = schema.columns().iter().enumerate().filter(|(_, c)| want.contains(&c.name())).map(|(i, _)| i).collect();
    let mask = ProjectionMask::leaves(schema, idx);
    let rdr = b.with_row_groups(vec![rg]).with_projection(mask).with_batch_size(8192).build()?;
    let mut out = Vec::new();
    for batch in rdr {
        let batch: RecordBatch = batch?;
        let col = |n: &str| -> Option<StringArray> {
            batch.column_by_name(n).and_then(|c| arrow_array::cast::AsArray::as_string_opt::<i32>(c.as_ref()).cloned())
        };
        let (t, s, x) = (col("title"), col("section"), col("text"));
        let get = |a: &Option<StringArray>, i: usize| a.as_ref().map(|a| if a.is_null(i) { String::new() } else { a.value(i).to_string() }).unwrap_or_default();
        for i in 0..batch.num_rows() {
            out.push(Passage { title: get(&t, i), section: get(&s, i), text: get(&x, i) });
        }
    }
    Ok(out)
}
