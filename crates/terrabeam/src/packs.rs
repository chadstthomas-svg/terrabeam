//! Engram packs: named sets of the published Tablinum tables, downloaded as one tar per table from the HF datasets and unpacked
//! into `<TERRABEAM_HOME>/engram/tables/<name>/` (the engram root layout; `wiki` lives at the root itself).
//! The registry here is the fallback; a dataset can override it with `packs.json` (same shape) so packs change without a release.

use crate::fetch;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TableSrc {
    pub repo: String,
    pub path: String,
    #[serde(default)]
    pub sha256: Option<String>,
    pub license: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Registry {
    pub tables: BTreeMap<String, TableSrc>,
    pub packs: BTreeMap<String, Vec<String>>,
    pub default_pack: String,
}

const T2: &str = "tablinum/tablinum-2";
const BET: &str = "tablinum/beam-engram-tables";
const T3: &str = "tablinum/tablinum-3";

pub fn builtin() -> Registry {
    let t = |repo: &str, path: &str, sha: &str, lic: &str| TableSrc {
        repo: repo.into(),
        path: path.into(),
        sha256: (!sha.is_empty()).then(|| sha.to_string()),
        license: lic.into(),
    };
    let mut tables = BTreeMap::new();
    tables.insert("software".into(), t(BET, "software-engram-table.tar", "25a005ac75044d86c369b9edb1c423de3a6aa039abe2d4265fffbbd9e245613c", "per-doc, see docs.json (permissive docs)"));
    tables.insert("law".into(), t(BET, "law-engram-table.tar", "b94d2badd48848d5871a585c7c9c5e3b405c5dc09768ce1211666d47be1ecc43", "public domain / CC BY (see SOURCES)"));
    tables.insert("oer".into(), t(BET, "oer-engram-table.tar", "308efed14a76869ed6504555bf05b3dcb0a961f2f0e4c50b08347a92022f2ef9", "CC BY open textbooks (see files.txt)"));
    for (n, sha) in [
        ("ecfr", "cc17b8338b04f01523cff11b11cdcd4443cd17e2c13507830b1dacbc6e1e169c"),
        ("eulex", "7f5e9cfa794dc66345a432a9bcba61b1c82882703a380ad76f55363f8fc0ecae"),
        ("finreg", "494379af31bcbcd457e1214f2f4cb32e510325fd2eaad460781a01e0e5874579"),
        ("humref", "198acff9ed109d76ca8ac39d6aba602e6dff439acdf8e445eb06663e54eb4e74"),
        ("sefin", "bdd0ef73d5bfcb8f9d01f1f73ee4e5b6054b2f279e368a29d1b1b9f9f334eb35"),
        ("sehum", "3a8e82f9b95982016a6fef1478b73093e7b341e06d68e40ed98b49a8b41c65c5"),
        ("selaw", "004483be4636e8cb059647577a5b91ee68f7f0afa45191d06e7bf2d2895a41c6"),
        ("sestem", "91402d8e3366ca522109f5dd5417971c2d8750a5da8671f3d48bf739ac5e78a4"),
        ("uscode", "c68243296b9c418e475a1d16adc92768d1524bfdc6e2d4314f82861ee3d758c3"),
    ] {
        tables.insert(n.into(), t(T2, &format!("tables/{n}.tar"), sha, "CC BY 4.0 compilation (Tablinum-2; sources in SOURCES.md)"));
    }
    tables.insert("wiki".into(), t(T2, "tables/wiki.tar", "5df562a53e73e0b30da2958bfd3388970d9f8655656bedc2af214cedafeab6eb", "CC BY-SA 4.0 (English Wikipedia)"));
    for (n, sha, lic) in [
        ("apps", "3f99ae18ded1acf48dbd059265655e1f14ddde8688b1804370af07d375b8ba97", "per-source: CC BY 3.0/4.0, CC BY-SA 4.0, MPL 2.0, GFDL (see SOURCES.md)"),
        ("web", "610ec7da34c8e8b7846fc1ec8f996946cd9553729d413b00fc22a31c3c1e097f", "per-source: W3C document/software licenses, Apache 2.0, MIT, CC0 (see SOURCES.md)"),
        ("scicode", "5fa83af8fc85010d2ea09c3ca0fa3bdf5128904523b897c55c82d43ef7649015", "per-source: GFDL 1.3 (Rosetta Code), BSD-3 (SciPy docs) (see SOURCES.md)"),
        ("homerepair", "4ce7eb729d0b93e519d53f2abcbb7758956d628efed504bc4187ffba7fa7e20b", "CC BY-SA 4.0 (Stack Exchange)"),
        ("sedev", "2ada535c30fcbbd947cf954962ad6104ff52898a8a6a8ea79eea5163d536e6d0", "CC BY-SA 4.0 (Stack Exchange)"),
        ("sesys", "6e8c5135d0151b1faa41b49b3e10b41094bd7446791bd5c725674dca86aea898", "CC BY-SA 4.0 (Stack Exchange)"),
    ] {
        tables.insert(n.into(), t(T3, &format!("tables/{n}.tar"), sha, lic));
    }
    // the Tablinum-3 data set's packs.json, mirrored as the offline fallback; packs nest: lite < standard < full
    let v = |names: &[&str]| -> Vec<String> { names.iter().map(|s| s.to_string()).collect() };
    let mut packs = BTreeMap::new();
    packs.insert("lite".into(), v(&["software", "apps", "web", "scicode", "sestem", "sehum", "humref", "sefin", "selaw", "uscode", "ecfr", "homerepair"]));
    packs.insert("agent".into(), v(&["software", "apps", "web", "scicode", "sedev", "sesys"]));
    packs.insert("standard".into(), v(&["software", "apps", "web", "scicode", "sestem", "sehum", "humref", "sefin", "selaw", "uscode", "ecfr", "homerepair", "law", "finreg", "eulex", "sedev", "sesys"]));
    packs.insert("full".into(), v(&["software", "apps", "web", "scicode", "sestem", "sehum", "humref", "sefin", "selaw", "uscode", "ecfr", "homerepair", "law", "finreg", "eulex", "sedev", "sesys", "oer", "wiki"]));
    packs.insert("wiki".into(), v(&["wiki"]));
    Registry { tables, packs, default_pack: "lite".into() }
}

/// The registry: the dataset's packs.json when it is reachable (so new tables, e.g. wiki, need no new binary), else builtin().
pub async fn registry(c: &reqwest::Client) -> Registry {
    let url = fetch::hf_url(T2, true, "main", "packs.json");
    if let Ok(r) = c.get(&url).send().await.and_then(|r| r.error_for_status()) {
        if let Ok(reg) = r.json::<Registry>().await {
            return reg;
        }
    }
    builtin()
}

pub fn engram_root(home: &Path) -> PathBuf {
    home.join("engram")
}

/// Download + unpack every table of `pack` that is not installed yet; returns (engram root, table names).
pub async fn ensure_pack(c: &reqwest::Client, home: &Path, reg: &Registry, pack: &str) -> Result<(PathBuf, Vec<String>)> {
    let names = reg.packs.get(pack).with_context(|| format!("no engram pack '{pack}' (have: {})", reg.packs.keys().cloned().collect::<Vec<_>>().join(", ")))?;
    let root = engram_root(home);
    for n in names {
        ensure_table(c, home, reg, n).await?;
    }
    Ok((root, names.clone()))
}

pub async fn ensure_table(c: &reqwest::Client, home: &Path, reg: &Registry, name: &str) -> Result<()> {
    let root = engram_root(home);
    let dir = if name == "wiki" { root.clone() } else { root.join("tables").join(name) };
    if dir.join("index/engram.index").exists() && dir.join(".installed").exists() {
        return Ok(());
    }
    let src = reg.tables.get(name).with_context(|| format!("table '{name}' is not in the registry"))?;
    eprintln!("terrabeam: engram table {name} ({})", src.license);
    let tar = home.join("downloads").join(format!("{}--{}", src.repo.replace('/', "--"), src.path.replace('/', "--")));
    fetch::download(c, &fetch::hf_url(&src.repo, true, "main", &src.path), &tar, None, src.sha256.as_deref()).await?;
    let (t2, d2, n2) = (tar.clone(), dir.clone(), name.to_string());
    tokio::task::spawn_blocking(move || untar_table(&t2, &d2, &n2)).await??;
    std::fs::write(dir.join(".installed"), &src.path)?;
    // the archive is not needed once unpacked (it is as large as the table)
    let _ = std::fs::remove_file(&tar);
    let _ = std::fs::remove_file(tar.with_extension("tar.ok"));
    Ok(())
}

/// Unpack a table tar into `dir`. Archives come in two layouts: `<name>/index/...` (Tablinum-2) or `index/...` (beam-engram-tables).
fn untar_table(tar: &Path, dir: &Path, name: &str) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut a = tar::Archive::new(std::fs::File::open(tar)?);
    for e in a.entries()? {
        let mut e = e?;
        let p = e.path()?.into_owned();
        let rel = p.strip_prefix(name).unwrap_or(&p).to_path_buf();
        if rel.as_os_str().is_empty() || rel.components().any(|c| matches!(c, std::path::Component::ParentDir | std::path::Component::RootDir)) {
            continue;
        }
        let out = dir.join(&rel);
        if e.header().entry_type().is_dir() {
            std::fs::create_dir_all(&out)?;
        } else if e.header().entry_type().is_file() {
            if let Some(par) = out.parent() {
                std::fs::create_dir_all(par)?;
            }
            e.unpack(&out)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_packs_only_name_known_tables() {
        let r = builtin();
        for (p, ts) in &r.packs {
            for t in ts {
                assert!(r.tables.contains_key(t), "pack {p} names unknown table {t}");
            }
        }
        assert!(r.packs.contains_key(&r.default_pack));
        assert!(!r.tables.contains_key("health"), "health is not redistributable");
    }

    #[test]
    fn packs_nest_and_never_ship_health() {
        let r = builtin();
        let has = |p: &str, q: &str| r.packs[p].iter().all(|t| r.packs[q].contains(t));
        assert!(has("lite", "standard") && has("standard", "full") && has("agent", "standard"));
        for t in ["health", "healthp", "fdalabel", "cdcj", "psych"] {
            assert!(!r.tables.contains_key(t), "{t}: health data stays local");
        }
    }

    #[test]
    fn untar_strips_a_leading_table_folder() {
        let d = std::env::temp_dir().join(format!("tb-untar-{}", std::process::id()));
        let tarp = d.join("t.tar");
        std::fs::create_dir_all(&d).unwrap();
        {
            let mut b = tar::Builder::new(std::fs::File::create(&tarp).unwrap());
            let data = b"x";
            let mut h = tar::Header::new_gnu();
            h.set_size(1);
            h.set_mode(0o644);
            h.set_cksum();
            b.append_data(&mut h, "selaw/index/engram.index", &data[..]).unwrap();
            b.finish().unwrap();
        }
        let out = d.join("out");
        untar_table(&tarp, &out, "selaw").unwrap();
        assert!(out.join("index/engram.index").exists());
        std::fs::remove_dir_all(&d).ok();
    }
}
