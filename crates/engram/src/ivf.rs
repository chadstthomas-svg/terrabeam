//! Reader for the engram key tables as faiss writes them: `IVF{nlist}_HNSW32,SQ8`, inner product, with
//! `OnDiskInvertedLists` (codes + ids in `lists.ivfdata`, mmapped). Format = faiss v1.15 `index_write.cpp`.
//!
//! Search differs from faiss in one place on purpose: the coarse step is an exact inner product over all
//! centroids (16k x 256 floats, ~1 ms) instead of the HNSW graph, so the probed lists are the true top-`nprobe`.
//! SQ8 with `by_residual`: x = centroid + vmin + (code + 0.5) / 255 * vdiff, so
//! <q, x> = <q, c> + sum(q * vmin) + sum(q * vdiff / 255 * (code + 0.5)) = base + sum(w * code) with per-query weights w.

use anyhow::{bail, ensure, Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::path::{Path, PathBuf};

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.p + n <= self.b.len(), "index truncated at byte {}", self.p);
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn fourcc(&mut self) -> Result<[u8; 4]> {
        Ok(self.take(4)?.try_into()?)
    }
    /// WRITEVECTOR: size_t count, then count * elem bytes
    fn vec_bytes(&mut self, elem: usize) -> Result<&'a [u8]> {
        let n = self.u64()? as usize;
        self.take(n * elem)
    }
    fn vec_f32(&mut self) -> Result<Vec<f32>> {
        Ok(self.vec_bytes(4)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

struct Header {
    d: usize,
    ntotal: u64,
    metric: i32,
}

fn header(r: &mut Rd) -> Result<Header> {
    let d = r.i32()? as usize;
    let ntotal = r.u64()?;
    r.u64()?; // dummy
    r.u64()?; // dummy
    r.u8()?; // is_trained
    let metric = r.i32()?;
    if metric > 1 {
        r.f32()?; // metric_arg
    }
    Ok(Header { d, ntotal, metric })
}

/// Coarse quantizer: returns the centroids (nlist x d) whatever the wrapper (IndexFlat or HNSW-over-flat storage).
fn quantizer(r: &mut Rd) -> Result<Vec<f32>> {
    let h = r.fourcc()?;
    match &h {
        b"IxFI" | b"IxF2" | b"IxFl" => {
            header(r)?;
            // WRITEXBVECTOR: count of 4-byte units, then the bytes
            let n = r.u64()? as usize;
            let b = r.take(n * 4)?;
            Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
        }
        b"IHNf" => {
            header(r)?;
            // write_HNSW: assign_probas<f64>, cum_nneighbor_per_level<i32>, levels<i32>, offsets<u64>, neighbors<i32>,
            // entry_point i32, max_level i32, efConstruction i32, efSearch i32, upper_beam i32
            r.vec_bytes(8)?;
            r.vec_bytes(4)?;
            r.vec_bytes(4)?;
            r.vec_bytes(8)?;
            r.vec_bytes(4)?;
            for _ in 0..5 {
                r.i32()?;
            }
            quantizer(r) // storage = IndexFlat
        }
        other => bail!("unsupported coarse quantizer {:?}", String::from_utf8_lossy(other)),
    }
}

#[derive(Clone, Copy)]
struct List {
    size: usize,
    capacity: usize,
    offset: usize,
}

pub struct IvfIndex {
    pub d: usize,
    pub nlist: usize,
    pub ntotal: u64,
    pub nprobe: usize,
    centroids: Vec<f32>,
    vmin: Vec<f32>,
    vdiff: Vec<f32>,
    by_residual: bool,
    code_size: usize,
    lists: Vec<List>,
    data: Mmap,
}

impl IvfIndex {
    /// `path` = <table>/index/engram.index; the list data file is the one the index names, re-rooted next to it
    /// (tables move between disks, the absolute path stored inside may be stale).
    pub fn open(path: &Path) -> Result<Self> {
        let raw = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let mut r = Rd { b: &raw, p: 0 };
        let h = r.fourcc()?;
        ensure!(&h == b"IwSq", "{}: not an IVF scalar-quantizer index ({:?})", path.display(), String::from_utf8_lossy(&h));
        let hd = header(&mut r)?;
        ensure!(hd.metric == 0, "expected inner-product metric, got {}", hd.metric);
        let nlist = r.u64()? as usize;
        let nprobe = r.u64()? as usize;
        let centroids = quantizer(&mut r)?;
        ensure!(centroids.len() == nlist * hd.d, "centroids {} != nlist {} x d {}", centroids.len(), nlist, hd.d);
        let dm_type = r.u8()?; // direct map
        r.vec_bytes(8)?;
        if dm_type == 2 {
            r.vec_bytes(16)?;
        }
        // ScalarQuantizer
        let qtype = r.i32()?;
        ensure!(qtype == 0, "only QT_8bit (non-uniform) is supported, got qtype {}", qtype);
        r.i32()?; // rangestat
        r.f32()?; // rangestat_arg
        let sq_d = r.u64()? as usize;
        let sq_cs = r.u64()? as usize;
        ensure!(sq_d == hd.d && sq_cs == hd.d, "SQ8 d/code_size mismatch");
        let trained = r.vec_f32()?;
        ensure!(trained.len() == 2 * hd.d, "SQ8 trained params {} != 2d", trained.len());
        let code_size = r.u64()? as usize;
        let by_residual = r.u8()? != 0;
        // OnDiskInvertedLists
        let ilh = r.fourcc()?;
        ensure!(&ilh == b"ilod", "expected on-disk inverted lists, got {:?}", String::from_utf8_lossy(&ilh));
        let il_nlist = r.u64()? as usize;
        let il_cs = r.u64()? as usize;
        ensure!(il_nlist == nlist && il_cs == code_size, "inverted lists header mismatch");
        let lb = r.vec_bytes(24)?;
        let lists: Vec<List> = lb
            .chunks_exact(24)
            .map(|c| List {
                size: u64::from_le_bytes(c[0..8].try_into().unwrap()) as usize,
                capacity: u64::from_le_bytes(c[8..16].try_into().unwrap()) as usize,
                offset: u64::from_le_bytes(c[16..24].try_into().unwrap()) as usize,
            })
            .collect();
        r.vec_bytes(16)?; // free slots
        let fname = String::from_utf8_lossy(r.vec_bytes(1)?).into_owned();
        let stored = PathBuf::from(&fname);
        let local = path.with_file_name(stored.file_name().context("list file name")?);
        let lf = if local.exists() { local } else { stored };
        let data = unsafe { Mmap::map(&File::open(&lf).with_context(|| format!("open {}", lf.display()))?)? };
        let (vmin, vdiff) = (trained[..hd.d].to_vec(), trained[hd.d..].to_vec());
        Ok(IvfIndex { d: hd.d, nlist, ntotal: hd.ntotal, nprobe, centroids, vmin, vdiff, by_residual, code_size, lists, data })
    }

    /// Top-`k` (score, id) for one L2-normalised query, probing `nprobe` lists. Ids are the passage pids.
    pub fn search(&self, q: &[f32], k: usize, nprobe: usize) -> Vec<(f32, i64)> {
        let d = self.d;
        assert_eq!(q.len(), d);
        let mut cs: Vec<(f32, usize)> = self.centroids.chunks_exact(d).enumerate().map(|(i, c)| (dot(q, c), i)).collect();
        let np = nprobe.min(self.nlist).max(1);
        cs.select_nth_unstable_by(np - 1, |a, b| b.0.total_cmp(&a.0));
        cs.truncate(np);
        // per-query SQ8 weights: w_j = q_j * vdiff_j / 255, base = sum q_j * (vmin_j + 0.5 * vdiff_j / 255)
        let w: Vec<f32> = (0..d).map(|j| q[j] * self.vdiff[j] / 255.0).collect();
        let base: f32 = (0..d).map(|j| q[j] * (self.vmin[j] + 0.5 * self.vdiff[j] / 255.0)).sum();
        let mut all: Vec<(f32, i64)> = cs
            .par_iter()
            .flat_map_iter(|&(cscore, li)| {
                let l = self.lists[li];
                let mut out: Vec<(f32, i64)> = Vec::new();
                if l.size == 0 || l.offset == usize::MAX {
                    return out.into_iter();
                }
                let codes = &self.data[l.offset..l.offset + l.size * self.code_size];
                let ids_at = l.offset + l.capacity * self.code_size;
                let ids = &self.data[ids_at..ids_at + l.size * 8];
                let off = base + if self.by_residual { cscore } else { 0.0 };
                out.reserve(l.size);
                for (i, code) in codes.chunks_exact(self.code_size).enumerate() {
                    let s = off + dot_u8(&w, code);
                    out.push((s, i64::from_le_bytes(ids[i * 8..i * 8 + 8].try_into().unwrap())));
                }
                // keep only this list's top-k to bound memory
                if out.len() > k {
                    out.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
                    out.truncate(k);
                }
                out.into_iter()
            })
            .collect();
        let k = k.min(all.len());
        if k == 0 {
            return all;
        }
        all.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
        all.truncate(k);
        all.sort_by(|a, b| b.0.total_cmp(&a.0));
        all
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    let r = a.len() / 8 * 8;
    for i in r..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline]
fn dot_u8(w: &[f32], c: &[u8]) -> f32 {
    let mut acc = [0f32; 8];
    for (x, y) in w.chunks_exact(8).zip(c.chunks_exact(8)) {
        for i in 0..8 {
            acc[i] += x[i] * y[i] as f32;
        }
    }
    let mut s: f32 = acc.iter().sum();
    let r = w.len() / 8 * 8;
    for i in r..w.len() {
        s += w[i] * c[i] as f32;
    }
    s
}
