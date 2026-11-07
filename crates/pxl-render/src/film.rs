//! Film response: exposure, view transform (AgX / ACES / standard), sRGB encoding.

use crate::Image;
use pxl_scene::View;

fn srgb_oetf(x: f32) -> f32 {
    if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

fn aces(x: f32) -> f32 {
    let (a, b, c, d, e) = (2.51, 0.03, 2.43, 0.59, 0.14);
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

/// AgX base (Troy Sobotka) with the polynomial contrast fit; output is display-encoded.
fn agx(c: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = c.map(|v| v.max(1e-10));
    let v = [
        0.842479 * r + 0.0784336 * g + 0.0792237 * b,
        0.0423282 * r + 0.878469 * g + 0.0791661 * b,
        0.0423757 * r + 0.0784336 * g + 0.879143 * b,
    ];
    const MIN_EV: f32 = -12.47393;
    const MAX_EV: f32 = 4.026069;
    let v = v.map(|x| {
        let x = ((x.max(1e-10).log2() - MIN_EV) / (MAX_EV - MIN_EV)).clamp(0.0, 1.0);
        let (x2, x4) = (x * x, x * x * x * x);
        15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232
    });
    [
        1.196879 * v[0] - 0.0980209 * v[1] - 0.0990297 * v[2],
        -0.0528969 * v[0] + 1.151903 * v[1] - 0.0989612 * v[2],
        -0.0529716 * v[0] - 0.0980435 * v[1] + 1.151074 * v[2],
    ]
    .map(|x| x.clamp(0.0, 1.0))
}

/// Linear HDR -> 8-bit display RGB.
pub fn to_rgb8(img: &Image, exposure: f32, view: View) -> Vec<u8> {
    let k = 2f32.powf(exposure);
    let q = |x: f32| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let mut out = Vec::with_capacity(img.rgb.len());
    for px in img.rgb.chunks_exact(3) {
        let c = [px[0] * k, px[1] * k, px[2] * k];
        match view {
            View::Agx => out.extend(agx(c).map(q)),
            View::Aces => out.extend(c.map(|x| q(srgb_oetf(aces(x))))),
            View::Standard => out.extend(c.map(|x| q(srgb_oetf(x)))),
        }
    }
    out
}
