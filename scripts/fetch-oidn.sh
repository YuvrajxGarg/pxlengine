#!/bin/sh
# Downloads Intel Open Image Denoise (Apache-2.0) into vendor/oidn (not committed).
set -e
cd "$(dirname "$0")/.."
V=2.5.1
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) PKG=oidn-$V.x64.windows.zip ;;
  Darwin) PKG=oidn-$V.arm64.macos.tar.gz ;;
  *) PKG=oidn-$V.x86_64.linux.tar.gz ;;
esac
mkdir -p vendor && cd vendor
[ -s "$PKG" ] || curl -fL "https://github.com/RenderKit/oidn/releases/download/v$V/$PKG" -o "$PKG"
rm -rf oidn "oidn-$V"*/
case "$PKG" in *.zip) unzip -q "$PKG" ;; *) tar -xzf "$PKG" ;; esac
mv "oidn-$V".*/ oidn
ls oidn/bin 2>/dev/null || ls oidn/lib
