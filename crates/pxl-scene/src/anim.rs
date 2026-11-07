//! Animation: keyframed tracks that write any numeric field of the scene JSON.
//!
//! A track addresses `nodes[name=<node>]/<path>` (or the scene root when `node` is omitted),
//! e.g. `{"node":"cam","path":"translation","keys":[[0,[0,1,5]],[48,[4,1,3]]],"interp":"spline"}`.

use anyhow::{anyhow, bail, Result};
use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Animation {
    #[serde(default)]
    pub frame_start: i32,
    /// Defaults to the last key of any track.
    #[serde(default)]
    pub frame_end: Option<i32>,
    #[serde(default)]
    pub tracks: Vec<Track>,
}

impl Animation {
    pub fn end(&self) -> i32 {
        self.frame_end.unwrap_or_else(|| {
            self.tracks
                .iter()
                .map(|t| match &t.orbit {
                    Some(o) => o.frames[1] as f64,
                    None => t.keys.last().map(|k| k.0).unwrap_or(0.0),
                })
                .fold(self.frame_start as f64, f64::max)
                .ceil() as i32
        })
    }

    pub fn validate(&self, raw: &Value) -> Result<()> {
        for (i, t) in self.tracks.iter().enumerate() {
            let ctx = |m: String| anyhow!("animation.tracks[{i}]: {m}");
            if t.orbit.is_none() && t.keys.is_empty() {
                return Err(ctx("needs `keys` ([[frame, value], ...]) or `orbit`".into()));
            }
            if t.orbit.is_some() && !t.keys.is_empty() {
                return Err(ctx("use either `keys` or `orbit`, not both".into()));
            }
            if t.keys.windows(2).any(|w| w[1].0 <= w[0].0) {
                return Err(ctx("key frames must be strictly increasing".into()));
            }
            for (f, v) in &t.keys {
                if as_numbers(v).is_none() {
                    return Err(ctx(format!("key at frame {f}: value must be a number or an array of numbers")));
                }
            }
            if t.path.is_empty() {
                return Err(ctx("empty `path`".into()));
            }
            if let Some(n) = &t.node {
                find_node(raw, n).map_err(|e| ctx(e.to_string()))?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Interp {
    #[default]
    Linear,
    /// Ease in/out within each pair of keys.
    Smooth,
    /// Catmull-Rom through all keys: smooth camera paths.
    Spline,
    /// Hold each key until the next.
    Step,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Orbit {
    #[serde(default)]
    pub center: [f32; 3],
    pub radius: f32,
    /// Height above `center`.
    #[serde(default)]
    pub height: f32,
    /// Angle at the first frame; 0 = +Z side of center.
    #[serde(default)]
    pub start_deg: f32,
    #[serde(default = "full_turn")]
    pub degrees: f32,
    pub frames: [i32; 2],
}

fn full_turn() -> f32 {
    360.0
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Track {
    /// Node `name`; omit to address the scene root (e.g. "world/strength").
    #[serde(default)]
    pub node: Option<String>,
    /// Slash-separated field path, e.g. "translation" or "materials/0/inputs/base_color".
    pub path: String,
    #[serde(default)]
    pub keys: Vec<(f64, Value)>,
    #[serde(default)]
    pub interp: Interp,
    #[serde(default)]
    pub orbit: Option<Orbit>,
}

fn as_numbers(v: &Value) -> Option<Vec<f64>> {
    match v {
        Value::Number(n) => Some(vec![n.as_f64()?]),
        Value::Array(a) => a.iter().map(|x| x.as_f64()).collect(),
        _ => None,
    }
}

fn to_value(nums: Vec<f64>, scalar: bool) -> Value {
    if scalar {
        serde_json::json!(nums[0])
    } else {
        Value::Array(nums.into_iter().map(|x| serde_json::json!(x)).collect())
    }
}

impl Track {
    pub fn eval(&self, frame: f64) -> Result<Value> {
        if let Some(o) = &self.orbit {
            let [f0, f1] = o.frames;
            let t = if f1 == f0 { 0.0 } else { ((frame - f0 as f64) / (f1 - f0) as f64).clamp(0.0, 1.0) };
            let a = (o.start_deg as f64 + o.degrees as f64 * t).to_radians();
            let c = o.center.map(|x| x as f64);
            return Ok(to_value(
                vec![c[0] + o.radius as f64 * a.sin(), c[1] + o.height as f64, c[2] + o.radius as f64 * a.cos()],
                false,
            ));
        }
        let keys = &self.keys;
        let scalar = matches!(keys[0].1, Value::Number(_));
        let vals: Vec<Vec<f64>> = keys.iter().map(|k| as_numbers(&k.1).unwrap()).collect();
        if vals.iter().any(|v| v.len() != vals[0].len()) {
            bail!("all keys of a track need the same number of components");
        }
        if frame <= keys[0].0 {
            return Ok(to_value(vals[0].clone(), scalar));
        }
        let last = keys.len() - 1;
        if frame >= keys[last].0 {
            return Ok(to_value(vals[last].clone(), scalar));
        }
        let i = keys.iter().rposition(|k| k.0 <= frame).unwrap();
        let t = (frame - keys[i].0) / (keys[i + 1].0 - keys[i].0);
        let out: Vec<f64> = match self.interp {
            Interp::Step => vals[i].clone(),
            Interp::Linear | Interp::Smooth => {
                let t = if self.interp == Interp::Smooth { t * t * (3.0 - 2.0 * t) } else { t };
                vals[i].iter().zip(&vals[i + 1]).map(|(a, b)| a + (b - a) * t).collect()
            }
            Interp::Spline => {
                let p0 = &vals[i.saturating_sub(1)];
                let (p1, p2) = (&vals[i], &vals[i + 1]);
                let p3 = &vals[(i + 2).min(last)];
                let (t2, t3) = (t * t, t * t * t);
                (0..p1.len())
                    .map(|c| {
                        0.5 * (2.0 * p1[c]
                            + (-p0[c] + p2[c]) * t
                            + (2.0 * p0[c] - 5.0 * p1[c] + 4.0 * p2[c] - p3[c]) * t2
                            + (-p0[c] + 3.0 * p1[c] - 3.0 * p2[c] + p3[c]) * t3)
                    })
                    .collect()
            }
        };
        Ok(to_value(out, scalar))
    }
}

fn find_node<'a>(raw: &'a Value, name: &str) -> Result<usize> {
    let nodes = raw.get("nodes").and_then(Value::as_array).ok_or_else(|| anyhow!("scene has no nodes"))?;
    nodes
        .iter()
        .position(|n| n.get("name").and_then(Value::as_str) == Some(name))
        .ok_or_else(|| {
            let names: Vec<_> = nodes.iter().filter_map(|n| n.get("name").and_then(Value::as_str)).collect();
            anyhow!("no node named '{name}' (named nodes: {})", if names.is_empty() { "none".into() } else { names.join(", ") })
        })
}

/// Write `value` into the scene JSON at the track's address, creating objects/arrays as needed.
pub fn apply(raw: &mut Value, track: &Track, value: Value) -> Result<()> {
    let mut cur: &mut Value = match &track.node {
        Some(n) => {
            let i = find_node(raw, n)?;
            &mut raw["nodes"][i]
        }
        None => raw,
    };
    let segs: Vec<&str> = track.path.split('/').filter(|s| !s.is_empty()).collect();
    for (k, seg) in segs.iter().enumerate() {
        let is_last = k + 1 == segs.len();
        let next_is_index = segs.get(k + 1).is_some_and(|s| s.parse::<usize>().is_ok());
        let fresh = || if next_is_index { Value::Array(vec![]) } else { Value::Object(Map::new()) };
        cur = match cur {
            Value::Array(arr) => {
                let idx: usize = seg.parse().map_err(|_| anyhow!("path segment '{seg}' must be an index"))?;
                while arr.len() <= idx {
                    arr.push(Value::Object(Map::new()));
                }
                if is_last {
                    arr[idx] = value;
                    return Ok(());
                }
                if !(arr[idx].is_object() || arr[idx].is_array()) {
                    arr[idx] = fresh();
                }
                &mut arr[idx]
            }
            Value::Object(map) => {
                if is_last {
                    map.insert(seg.to_string(), value);
                    return Ok(());
                }
                let e = map.entry(seg.to_string()).or_insert_with(fresh);
                if !(e.is_object() || e.is_array()) {
                    *e = fresh();
                }
                e
            }
            _ => bail!("path '{}' runs through a non-object at '{seg}'", track.path),
        };
    }
    Ok(())
}
