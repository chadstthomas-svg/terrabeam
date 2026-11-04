//! First-stage query keys: sentence-transformers/static-retrieval-mrl-en-v1 (Apache-2.0). Mean of the static token
//! vectors (no special tokens, first 512 tokens), Matryoshka-truncated to `dim`, L2-normalised (= static_embed.py).

use anyhow::{Context, Result};
use safetensors::{Dtype, SafeTensors};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct StaticEmbed {
    tok: Tokenizer,
    w: Vec<f32>, // vocab x dim
    dim: usize,
}

impl StaticEmbed {
    /// `dir` = the model's `0_StaticEmbedding` folder (tokenizer.json + model.safetensors)
    pub fn open(dir: &Path, dim: usize) -> Result<Self> {
        let mut tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        tok.with_padding(None);
        tok.with_truncation(None).map_err(|e| anyhow::anyhow!("{e}"))?;
        let raw = std::fs::read(dir.join("model.safetensors")).context("read static embedding weights")?;
        let st = SafeTensors::deserialize(&raw)?;
        let t = st.tensor("embedding.weight")?;
        let (vocab, full) = (t.shape()[0], t.shape()[1]);
        anyhow::ensure!(dim <= full, "dim {dim} > {full}");
        let d = t.data();
        let val = |i: usize| -> f32 {
            match t.dtype() {
                Dtype::F32 => f32::from_le_bytes(d[i * 4..i * 4 + 4].try_into().unwrap()),
                Dtype::F16 => half_to_f32(u16::from_le_bytes(d[i * 2..i * 2 + 2].try_into().unwrap())),
                other => panic!("unsupported dtype {other:?}"),
            }
        };
        let mut w = Vec::with_capacity(vocab * dim);
        for r in 0..vocab {
            for c in 0..dim {
                w.push(val(r * full + c));
            }
        }
        Ok(StaticEmbed { tok, w, dim })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let enc = self.tok.encode(text, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let ids = &enc.get_ids()[..enc.get_ids().len().min(512)];
        let mut v = vec![0f32; self.dim];
        if !ids.is_empty() {
            for &id in ids {
                let row = &self.w[id as usize * self.dim..(id as usize + 1) * self.dim];
                for (a, b) in v.iter_mut().zip(row) {
                    *a += b;
                }
            }
            let n = ids.len() as f32;
            v.iter_mut().for_each(|x| *x /= n);
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-12;
        v.iter_mut().for_each(|x| *x /= norm);
        Ok(v)
    }
}

fn half_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            s | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}
