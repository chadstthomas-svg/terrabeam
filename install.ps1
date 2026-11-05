# terrabeam installer for Windows: downloads the latest release into %LOCALAPPDATA%\terrabeam and adds it to the user PATH.
# Usage: irm https://raw.githubusercontent.com/chadstthomas-svg/terrabeam/master/install.ps1 | iex
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Windows PowerShell 5.1's progress bar slows downloads to a crawl
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$repo = "chadstthomas-svg/terrabeam"
$t = "x86_64-pc-windows-msvc"   # also runs on Windows on ARM (x64 emulation)
$tag = if ($env:TERRABEAM_VERSION) { $env:TERRABEAM_VERSION } else { (Invoke-RestMethod "https://api.github.com/repos/$repo/releases?per_page=1")[0].tag_name }
$f = "terrabeam-$tag-$t.zip"
$dir = Join-Path $env:LOCALAPPDATA "terrabeam"; New-Item -ItemType Directory -Force -Path $dir | Out-Null
$zip = Join-Path $env:TEMP $f
Write-Host "terrabeam: downloading $f"
Invoke-WebRequest "https://github.com/$repo/releases/download/$tag/$f" -OutFile $zip
# GitHub serves .sha256 as application/octet-stream, so .Content would be a byte array: save it and read it as text
$sha = "$zip.sha256"
Invoke-WebRequest "https://github.com/$repo/releases/download/$tag/$f.sha256" -OutFile $sha
$want = ((Get-Content $sha -Raw).Trim() -split '\s+')[0]
$got = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
if ($want.ToLower() -ne $got) { throw "terrabeam: checksum mismatch (expected $want, got $got)" }
Expand-Archive -Force $zip $dir
$p = [Environment]::GetEnvironmentVariable("Path", "User")
if ($p -notlike "*$dir*") { [Environment]::SetEnvironmentVariable("Path", "$p;$dir", "User"); Write-Host "added $dir to your PATH (open a new terminal)" }
Write-Host "terrabeam $tag installed. Next:  terrabeam run tablinum/Moon-Beam-8B-A1B-GGUF-Preview"
