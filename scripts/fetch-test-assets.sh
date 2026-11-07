#!/bin/sh
# Downloads third-party test assets into assets/ (not committed).
#  - DamagedHelmet.glb: Khronos glTF Sample Assets, CC BY 4.0 (theblueturtle_, Battle Damaged Sci-fi Helmet - PBR)
#  - AlphaBlendModeTest.glb: Khronos glTF Sample Assets, CC0
#  - studio_small_09_2k.hdr, kloofendal_48d_partly_cloudy_puresky_2k.hdr: Poly Haven, CC0
set -e
cd "$(dirname "$0")/.."
mkdir -p assets
KH=https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models
PH=https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k
get() { [ -s "assets/$2" ] || curl -fsSL "$1" -o "assets/$2"; }
get "$KH/DamagedHelmet/glTF-Binary/DamagedHelmet.glb" DamagedHelmet.glb
get "$KH/AlphaBlendModeTest/glTF-Binary/AlphaBlendModeTest.glb" AlphaBlendModeTest.glb
get "$PH/studio_small_09_2k.hdr" studio_small_09_2k.hdr
get "$PH/kloofendal_48d_partly_cloudy_puresky_2k.hdr" kloofendal_48d_partly_cloudy_puresky_2k.hdr
ls -la assets
