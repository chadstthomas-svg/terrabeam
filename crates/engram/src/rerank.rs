//! Second stage: cross-encoder/ms-marco-MiniLM-L6-v2 (Apache-2.0) in candle, CPU fp32 (fp32 = the torch ranking; 11-03 found
//! int8 ONNX loses top-1 on 5/60). BertForSequenceClassification = BERT -> pooler (dense + tanh on [CLS]) -> classifier (1 logit).
//! Pairs (query, passage) truncated longest-first to 256 tokens, as the Python `ct(..., truncation=True, max_length=256)`.
//! Speed: pairs are sorted by length and batched so each batch pads only to its own longest pair.
//! Two backends with the same tokenisation/batching: onnxruntime (`ort`, fp32 model.onnx; default, 11-04: ~2.6x faster than
//! candle on this CPU) and candle (pure Rust, no native lib; fallback). Both match torch: top-1 60/60, |diff| ~1e-5.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use std::path::Path;
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

enum Backend {
    Candle { bert: BertModel, pooler: Linear, classifier: Linear, dev: Device },
    #[cfg(feature = "ort")]
    Ort(std::sync::Mutex<ort::session::Session>, #[allow(dead_code)] ort::environment::Environment), // env kept alive with the session
}

pub struct CrossEncoder {
    backend: Backend,
    tok: Tokenizer,
    pub batch: usize,
}

impl CrossEncoder {
    /// `dir` = the HF snapshot folder (config.json, model.safetensors | onnx/model.onnx, tokenizer.json).
    /// Uses onnxruntime when `onnx/model.onnx` exists (threads = `threads`), else candle.
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_with(dir, true, 4)
    }

    pub fn open_with(dir: &Path, prefer_ort: bool, threads: usize) -> Result<Self> {
        let tok = Self::tokenizer(dir)?;
        let onnx = dir.join("onnx/model.onnx");
        #[cfg(not(feature = "ort"))]
        let _ = (prefer_ort, threads, &onnx);
        #[cfg(feature = "ort")]
        if prefer_ort && onnx.exists() {
            use ort::session::builder::GraphOptimizationLevel;
            let env = ort::init().with_name("terrabeam").with_telemetry(false).build().map_err(|e| anyhow::anyhow!("ort env: {e}"))?;
            let sess = ort::session::Session::builder(&env)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .with_intra_threads(threads)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .commit_from_file(&onnx)
                .map_err(|e| anyhow::anyhow!("onnx {}: {e}", onnx.display()))?;
            return Ok(CrossEncoder { backend: Backend::Ort(std::sync::Mutex::new(sess), env), tok, batch: 16 });
        }
        let dev = Device::Cpu;
        let cfg: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &dev)? };
        let bert = BertModel::load(vb.clone(), &cfg).context("bert")?;
        let pooler = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("bert.pooler.dense")).context("pooler")?;
        let classifier = candle_nn::linear(cfg.hidden_size, 1, vb.pp("classifier")).context("classifier")?;
        Ok(CrossEncoder { backend: Backend::Candle { bert, pooler, classifier, dev }, tok, batch: 16 })
    }

    pub fn backend_name(&self) -> &'static str {
        match self.backend {
            Backend::Candle { .. } => "candle",
            #[cfg(feature = "ort")]
            Backend::Ort(..) => "onnxruntime",
        }
    }

    fn tokenizer(dir: &Path) -> Result<Tokenizer> {
        let mut tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
        tok.with_padding(None);
        tok.with_truncation(Some(TruncationParams {
            max_length: 256,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
            direction: TruncationDirection::Right,
        }))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(tok)
    }

    fn logits(&self, ids: Vec<i64>, tt: Vec<i64>, am: Vec<i64>, b: usize, l: usize) -> Result<Vec<f32>> {
        match &self.backend {
            Backend::Candle { bert, pooler, classifier, dev } => {
                let ids = Tensor::from_vec(ids, (b, l), dev)?;
                let tt = Tensor::from_vec(tt, (b, l), dev)?;
                let am = Tensor::from_vec(am, (b, l), dev)?;
                let h = bert.forward(&ids, &tt, Some(&am))?;
                let pooled = pooler.forward(&h.i((.., 0, ..))?)?.tanh()?;
                Ok(classifier.forward(&pooled)?.squeeze(1)?.to_vec1::<f32>()?)
            }
            #[cfg(feature = "ort")]
            Backend::Ort(sess, _) => {
                use ort::value::Tensor as OT;
                let mk = |v: Vec<i64>| OT::from_array(([b, l], v)).map_err(|e| anyhow::anyhow!("{e}"));
                let mut s = sess.lock().unwrap();
                let out = s
                    .run(ort::inputs!["input_ids" => mk(ids)?, "attention_mask" => mk(am)?, "token_type_ids" => mk(tt)?])
                    .map_err(|e| anyhow::anyhow!("onnx run: {e}"))?;
                let (_, data) = out[0].try_extract_tensor::<f32>().map_err(|e| anyhow::anyhow!("{e}"))?;
                Ok(data.iter().step_by(data.len() / b).copied().collect())
            }
        }
    }

    /// One relevance logit per passage for `q` (higher = more relevant), in input order.
    pub fn score(&self, q: &str, passages: &[&str]) -> Result<Vec<f32>> {
        let encs = passages
            .iter()
            .map(|p| self.tok.encode((q, *p), true).map_err(|e| anyhow::anyhow!("{e}")))
            .collect::<Result<Vec<_>>>()?;
        let mut order: Vec<usize> = (0..encs.len()).collect();
        order.sort_by_key(|&i| encs[i].get_ids().len());
        let mut out = vec![0f32; encs.len()];
        for chunk in order.chunks(self.batch.max(1)) {
            let l = chunk.iter().map(|&i| encs[i].get_ids().len()).max().unwrap_or(1);
            let (mut ids, mut tt, mut am) = (Vec::new(), Vec::new(), Vec::new());
            for &i in chunk {
                let e = &encs[i];
                let n = e.get_ids().len();
                ids.extend(e.get_ids().iter().map(|&x| x as i64).chain(std::iter::repeat(0).take(l - n)));
                tt.extend(e.get_type_ids().iter().map(|&x| x as i64).chain(std::iter::repeat(0).take(l - n)));
                am.extend(std::iter::repeat(1i64).take(n).chain(std::iter::repeat(0).take(l - n)));
            }
            let logits = self.logits(ids, tt, am, chunk.len(), l)?;
            for (&i, v) in chunk.iter().zip(logits) {
                out[i] = v;
            }
        }
        Ok(out)
    }
}
