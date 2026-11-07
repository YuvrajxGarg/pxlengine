//! Camera tracks in Nuke `.chan` format: one line per frame,
//! `frame tx ty tz rx ry rz [vertical_fov_degrees]`, rotations in degrees, order ZXY
//! (Z applied first). Blender, SynthEyes, 3DEqualizer and PFTrack can export it.

use crate::CameraNode;
use anyhow::{bail, Context, Result};
use glam::{Quat, Vec3};
use std::fmt::Write as _;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct TrackKey {
    pub frame: f64,
    pub translation: Vec3,
    pub rotation: Quat,
    pub fov: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct CameraTrack {
    pub keys: Vec<TrackKey>,
}

impl CameraTrack {
    pub fn parse(text: &str) -> Result<CameraTrack> {
        let mut keys = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let v: Vec<f64> = line
                .split_whitespace()
                .map(str::parse)
                .collect::<std::result::Result<_, _>>()
                .with_context(|| format!("line {}: expected numbers", n + 1))?;
            if v.len() < 7 {
                bail!("line {}: expected `frame tx ty tz rx ry rz [fov]`, got {} values", n + 1, v.len());
            }
            keys.push(TrackKey {
                frame: v[0],
                translation: Vec3::new(v[1] as f32, v[2] as f32, v[3] as f32),
                rotation: CameraNode::rotation_quat([v[4] as f32, v[5] as f32, v[6] as f32]),
                fov: v.get(7).map(|f| *f as f32),
            });
        }
        if keys.is_empty() {
            bail!("camera track has no keys");
        }
        keys.sort_by(|a, b| a.frame.total_cmp(&b.frame));
        Ok(CameraTrack { keys })
    }

    pub fn load(path: &Path) -> Result<CameraTrack> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading camera track {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("in {}", path.display()))
    }

    /// Camera pose at `frame` (linear translation, slerped rotation; held outside the range).
    pub fn sample(&self, frame: f64) -> TrackKey {
        let k = &self.keys;
        if frame <= k[0].frame {
            return k[0];
        }
        if frame >= k[k.len() - 1].frame {
            return k[k.len() - 1];
        }
        let i = k.partition_point(|x| x.frame <= frame) - 1;
        let (a, b) = (k[i], k[i + 1]);
        let t = ((frame - a.frame) / (b.frame - a.frame)) as f32;
        TrackKey {
            frame,
            translation: a.translation.lerp(b.translation, t),
            rotation: a.rotation.slerp(b.rotation, t),
            fov: match (a.fov, b.fov) {
                (Some(x), Some(y)) => Some(x + (y - x) * t),
                (x, y) => x.or(y),
            },
        }
    }

    /// Drive `cam` from the track at `frame`.
    pub fn apply(&self, cam: &mut CameraNode, frame: f64) {
        let k = self.sample(frame);
        cam.translation = k.translation.to_array();
        cam.rotation = Some(CameraNode::rotation_degrees(k.rotation));
        if let Some(f) = k.fov {
            cam.fov = f;
        }
    }

    /// One `.chan` line for a camera at `frame`.
    pub fn chan_line(cam: &CameraNode, frame: f64) -> String {
        let (eye, fwd, right, up) = cam.basis();
        let q = Quat::from_mat3(&glam::Mat3::from_cols(right, up, -fwd)).normalize();
        let [rx, ry, rz] = CameraNode::rotation_degrees(q);
        let mut s = String::new();
        let _ = write!(s, "{frame} {} {} {} {rx} {ry} {rz} {}", eye.x, eye.y, eye.z, cam.fov);
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam(json: &str) -> CameraNode {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn chan_round_trip_matches_look_at() {
        let c = cam(r#"{"translation":[3,2,5],"target":[0.5,0.2,-1],"fov":35}"#);
        let line = CameraTrack::chan_line(&c, 4.0);
        let track = CameraTrack::parse(&line).unwrap();
        let mut c2 = cam(r#"{}"#);
        track.apply(&mut c2, 4.0);
        let (e1, f1, r1, u1) = c.basis();
        let (e2, f2, r2, u2) = c2.basis();
        assert!((e1 - e2).length() < 1e-4);
        assert!((f1 - f2).length() < 1e-4 && (r1 - r2).length() < 1e-4 && (u1 - u2).length() < 1e-4, "{f1} {f2}");
        assert_eq!(c2.fov, 35.0);
    }

    #[test]
    fn zxy_order() {
        // Pure yaw of +90° about Y turns the -Z view direction to -X.
        let c = cam(r#"{"rotation":[0,90,0]}"#);
        let (_, f, _, _) = c.basis();
        assert!((f - Vec3::NEG_X).length() < 1e-5, "{f}");
        // Pitch +90° about X looks up (+Y).
        let c = cam(r#"{"rotation":[90,0,0]}"#);
        let (_, f, _, _) = c.basis();
        assert!((f - Vec3::Y).length() < 1e-5, "{f}");
    }

    #[test]
    fn interpolates() {
        let t = CameraTrack::parse("0 0 0 0 0 0 0 40\n10 10 0 0 0 90 0 60\n").unwrap();
        let k = t.sample(5.0);
        assert!((k.translation.x - 5.0).abs() < 1e-5);
        assert!((k.fov.unwrap() - 50.0).abs() < 1e-4);
    }
}
