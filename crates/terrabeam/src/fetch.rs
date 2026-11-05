//! Downloads into the user cache (TERRABEAM_HOME, else the OS cache dir /terrabeam): resumable, size- and SHA256-checked,
//! with one progress line on stderr. Hugging Face (models, datasets) and GitHub release assets; HF_TOKEN for private repos.

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub fn home() -> PathBuf {
    std::env::var_os("TERRABEAM_HOME").map(PathBuf::from).unwrap_or_else(|| dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join("terrabeam"))
}

pub fn client() -> reqwest::Client {
    reqwest::Client::builder().user_agent(concat!("terrabeam/", env!("CARGO_PKG_VERSION"))).build().expect("http client")
}

fn auth(rb: reqwest::RequestBuilder, url: &str) -> reqwest::RequestBuilder {
    match std::env::var("HF_TOKEN") {
        Ok(t) if url.starts_with("https://huggingface.co/") && !t.is_empty() => rb.bearer_auth(t),
        _ => rb,
    }
}

pub fn hf_url(repo: &str, dataset: bool, revision: &str, path: &str) -> String {
    let kind = if dataset { "datasets/" } else { "" };
    format!("https://huggingface.co/{kind}{repo}/resolve/{revision}/{path}")
}

#[derive(Debug, Clone)]
pub struct RemoteFile {
    pub path: String,
    pub size: u64,
    pub sha256: Option<String>,
}

/// Files under `prefix` in an HF repo (recursive), with their LFS sha256 when the file is in LFS.
pub async fn hf_tree(c: &reqwest::Client, repo: &str, dataset: bool, revision: &str, prefix: &str) -> Result<Vec<RemoteFile>> {
    let kind = if dataset { "datasets" } else { "models" };
    let url = format!("https://huggingface.co/api/{kind}/{repo}/tree/{revision}/{prefix}?recursive=true");
    let r = auth(c.get(&url), &url).send().await?.error_for_status().with_context(|| format!("list {url}"))?;
    let items: Vec<serde_json::Value> = r.json().await?;
    Ok(items
        .into_iter()
        .filter(|i| i["type"] == "file")
        .map(|i| RemoteFile {
            path: i["path"].as_str().unwrap_or_default().to_string(),
            size: i["lfs"]["size"].as_u64().or_else(|| i["size"].as_u64()).unwrap_or(0),
            sha256: i["lfs"]["oid"].as_str().map(String::from),
        })
        .collect())
}

/// Fetch `url` to `dest` unless a verified copy is there. Resumes a `.part` file with an HTTP Range request.
pub async fn download(c: &reqwest::Client, url: &str, dest: &Path, size: Option<u64>, sha256: Option<&str>) -> Result<PathBuf> {
    let ok = dest.with_extension(format!("{}.ok", dest.extension().and_then(|e| e.to_str()).unwrap_or("")));
    if dest.exists() && ok.exists() && size.is_none_or(|s| std::fs::metadata(dest).map(|m| m.len() == s).unwrap_or(false)) {
        return Ok(dest.to_path_buf());
    }
    if let Some(p) = dest.parent() {
        std::fs::create_dir_all(p)?;
    }
    let part = dest.with_extension(format!("{}.part", dest.extension().and_then(|e| e.to_str()).unwrap_or("")));
    let mut have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let mut hasher = Sha256::new();
    if have > 0 {
        let mut f = std::fs::File::open(&part)?;
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
    }
    let mut rb = auth(c.get(url), url);
    if have > 0 {
        rb = rb.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }
    let r = rb.send().await.with_context(|| format!("GET {url}"))?;
    if have > 0 && r.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        have = 0; // server ignored the range: start over
        hasher = Sha256::new();
    }
    let r = r.error_for_status().with_context(|| format!("GET {url}"))?;
    let total = size.or(r.content_length().map(|l| l + have));
    let mut f = std::fs::OpenOptions::new().create(true).append(have > 0).write(true).truncate(have == 0).open(&part)?;
    let name = dest.file_name().unwrap().to_string_lossy().to_string();
    let (t0, mut last, mut got) = (Instant::now(), Instant::now(), have);
    let mut stream = r.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        f.write_all(&chunk)?;
        hasher.update(&chunk);
        got += chunk.len() as u64;
        if last.elapsed().as_secs_f32() > 2.0 {
            last = Instant::now();
            let rate = (got - have) as f64 / t0.elapsed().as_secs_f64() / 1e6;
            match total {
                Some(t) => eprint!("\r  {name}: {:.0}% of {:.2} GB, {rate:.1} MB/s   ", 100.0 * got as f64 / t as f64, t as f64 / 1e9),
                None => eprint!("\r  {name}: {:.2} GB, {rate:.1} MB/s   ", got as f64 / 1e9),
            }
        }
    }
    f.flush()?;
    drop(f);
    eprintln!("\r  {name}: done ({:.2} GB)                       ", got as f64 / 1e9);
    if let Some(s) = size {
        if got != s {
            bail!("{name}: got {got} bytes, expected {s} (kept {} to resume)", part.display());
        }
    }
    let digest = hex::encode(hasher.finalize());
    if let Some(want) = sha256 {
        if !digest.eq_ignore_ascii_case(want.trim_start_matches("sha256:")) {
            std::fs::remove_file(&part).ok();
            bail!("{name}: SHA256 mismatch (got {digest}, expected {want}); removed the download");
        }
    }
    std::fs::rename(&part, dest)?;
    std::fs::write(&ok, digest)?;
    Ok(dest.to_path_buf())
}

/// Unpack a .tar.gz or .zip into `dir` (flattened: llama.cpp archives keep everything in one folder level).
pub fn extract(archive: &Path, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let name = archive.to_string_lossy();
    if name.ends_with(".zip") {
        let mut z = zip::ZipArchive::new(std::fs::File::open(archive)?)?;
        for i in 0..z.len() {
            let mut e = z.by_index(i)?;
            if e.is_dir() {
                continue;
            }
            let Some(fname) = e.enclosed_name().and_then(|p| p.file_name().map(|f| f.to_owned())) else { continue };
            let mut out = std::fs::File::create(dir.join(fname))?;
            std::io::copy(&mut e, &mut out)?;
        }
    } else {
        let gz = flate2::read::GzDecoder::new(std::fs::File::open(archive)?);
        let mut t = tar::Archive::new(gz);
        for e in t.entries()? {
            let mut e = e?;
            let p = e.path()?.into_owned();
            let Some(fname) = p.file_name().map(|f| f.to_owned()) else { continue };
            match e.header().entry_type() {
                tar::EntryType::Regular => {
                    e.unpack(dir.join(&fname))?;
                }
                tar::EntryType::Symlink => {
                    if let Some(target) = e.link_name()? {
                        let link = dir.join(&fname);
                        let _ = std::fs::remove_file(&link);
                        #[cfg(unix)]
                        std::os::unix::fs::symlink(target.file_name().unwrap_or(target.as_os_str()), &link)?;
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}
