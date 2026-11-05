#!/bin/sh
# terrabeam installer: downloads the latest release binary for this machine into ~/.local/bin (override with TERRABEAM_BIN).
# Usage: curl -fsSL https://raw.githubusercontent.com/chadstthomas-svg/terrabeam/master/install.sh | sh
set -eu
REPO=chadstthomas-svg/terrabeam
BIN=${TERRABEAM_BIN:-$HOME/.local/bin}
os=$(uname -s); arch=$(uname -m)
case "$os-$arch" in
  Linux-x86_64)  t=x86_64-unknown-linux-gnu ;;
  Linux-aarch64|Linux-arm64) t=aarch64-unknown-linux-gnu ;;
  Darwin-arm64)  t=aarch64-apple-darwin ;;
  Darwin-x86_64) t=x86_64-apple-darwin ;;
  *) echo "terrabeam: no prebuilt binary for $os-$arch; try: cargo install --git https://github.com/$REPO terrabeam" >&2; exit 1 ;;
esac
tag=${TERRABEAM_VERSION:-$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)}
[ -n "$tag" ] || { echo "terrabeam: could not find the latest release" >&2; exit 1; }
f="terrabeam-$tag-$t.tar.gz"; tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
echo "terrabeam: downloading $f"
curl -fsSL -o "$tmp/$f" "https://github.com/$REPO/releases/download/$tag/$f"
curl -fsSL -o "$tmp/$f.sha256" "https://github.com/$REPO/releases/download/$tag/$f.sha256"
(cd "$tmp" && if command -v sha256sum >/dev/null; then sha256sum -c "$f.sha256"; else shasum -a 256 -c "$f.sha256"; fi) >/dev/null
tar -xzf "$tmp/$f" -C "$tmp"
mkdir -p "$BIN" && mv "$tmp/terrabeam" "$BIN/terrabeam" && chmod +x "$BIN/terrabeam"
echo "terrabeam $tag installed to $BIN/terrabeam"
case ":$PATH:" in *":$BIN:"*) ;; *) echo "add it to your PATH:  export PATH=\"$BIN:\$PATH\"" ;; esac
echo "next:  terrabeam run tablinum/Moon-Beam-8B-A1B-GGUF-Preview"
