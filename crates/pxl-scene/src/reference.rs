//! Field reference generated from the scene types (doc comments + serde defaults) via
//! JSON Schema, so `pxl help field`, `pxl schema` and the SDK types never drift.

use crate::Scene;
use anyhow::{bail, Result};
use serde_json::{Map, Value};
use std::fmt::Write as _;

/// JSON Schema (draft 2020-12) for scene files.
pub fn json_schema() -> Value {
    let schema = schemars::schema_for!(Scene);
    serde_json::to_value(&schema).expect("schema serializes")
}

/// Entry points for lookups: `camera.fov`, `material.inputs.specular_ior`, `render.plate`...
/// Node types are variants of `Node` (tagged by "type"); the rest are named definitions.
const NODE_TYPES: &[&str] = &["mesh", "asset", "light", "camera"];
const ALIASES: &[(&str, &str)] = &[
    ("material", "Material"),
    ("inputs", "MaterialInputs"),
    ("geometry", "Geometry"),
    ("world", "World"),
    ("render", "RenderSettings"),
    ("animation", "Animation"),
    ("track", "Track"),
    ("orbit", "Orbit"),
    ("programs", "Programs"),
    ("simulation", "Simulation"),
    ("simulations", "Simulation"),
    ("fluid", "Fluid"),
    ("fluids", "Fluid"),
    ("fill", "FluidBlock"),
    ("plate", "Plate"),
];

fn defs(schema: &Value) -> &Map<String, Value> {
    schema.get("$defs").and_then(Value::as_object).expect("$defs")
}

/// Follow `$ref`, `Option` (anyOf with null), and arrays to the schema describing values.
fn resolve<'a>(schema: &'a Value, mut s: &'a Value) -> &'a Value {
    for _ in 0..8 {
        if let Some(r) = s.get("$ref").and_then(Value::as_str) {
            let name = r.rsplit('/').next().unwrap_or("");
            match defs(schema).get(name) {
                Some(d) => {
                    s = d;
                    continue;
                }
                None => return s,
            }
        }
        if let Some(any) = s.get("anyOf").and_then(Value::as_array) {
            let non_null: Vec<&Value> = any.iter().filter(|v| v.get("type") != Some(&Value::from("null"))).collect();
            if non_null.len() == 1 {
                s = non_null[0];
                continue;
            }
        }
        if s.get("type") == Some(&Value::from("array")) {
            if let Some(items) = s.get("items").filter(|i| i.is_object()) {
                if items.get("$ref").is_some() || items.get("properties").is_some() {
                    s = items;
                    continue;
                }
            }
        }
        return s;
    }
    s
}

fn type_name(schema: &Value, s: &Value) -> String {
    if let Some(r) = s.get("$ref").and_then(Value::as_str) {
        let name = r.rsplit('/').next().unwrap_or("");
        let d = defs(schema).get(name);
        if let Some(e) = d.and_then(|d| d.get("enum")).and_then(Value::as_array) {
            return e.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" | ");
        }
        if let Some(variants) = d.and_then(|d| d.get("oneOf")).and_then(Value::as_array) {
            let consts: Vec<String> = variants.iter().filter_map(|v| v.get("const").map(|c| c.to_string())).collect();
            if !consts.is_empty() && consts.len() == variants.len() {
                return consts.join(" | ");
            }
        }
        return name.to_string();
    }
    if let Some(any) = s.get("anyOf").and_then(Value::as_array) {
        let parts: Vec<String> = any
            .iter()
            .filter(|v| v.get("type") != Some(&Value::from("null")))
            .map(|v| type_name(schema, v))
            .collect();
        return parts.join(" | ");
    }
    if let Some(e) = s.get("enum").and_then(Value::as_array) {
        return e.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" | ");
    }
    match s.get("type") {
        Some(Value::String(t)) if t == "array" => {
            let items = s.get("items").map(|i| type_name(schema, i)).unwrap_or_else(|| "any".into());
            match (s.get("minItems"), s.get("maxItems")) {
                (Some(a), Some(b)) if a == b => format!("[{items}; {a}]"),
                _ => format!("[{items}]"),
            }
        }
        Some(Value::String(t)) => match t.as_str() {
            "number" => "number".into(),
            "integer" => "integer".into(),
            other => other.into(),
        },
        Some(Value::Array(ts)) => {
            // Option<T>: ["T", "null"] -> describe T (arrays keep their item type and length).
            let non_null: Vec<&Value> = ts.iter().filter(|t| *t != "null").collect();
            if non_null.len() == 1 {
                let mut single = s.clone();
                single["type"] = non_null[0].clone();
                return format!("{} | null", type_name(schema, &single));
            }
            non_null.iter().map(|t| t.as_str().unwrap_or("?").to_string()).collect::<Vec<_>>().join(" | ")
        }
        _ => {
            if s.get("prefixItems").is_some() {
                "[frame, value]".into()
            } else {
                "any".into()
            }
        }
    }
}

fn first_line(s: &Value) -> String {
    s.get("description").and_then(Value::as_str).unwrap_or("").lines().next().unwrap_or("").trim().to_string()
}

fn describe_object(schema: &Value, s: &Value, out: &mut String) {
    let s = resolve(schema, s);
    if let Some(variants) = s.get("oneOf").and_then(Value::as_array) {
        // Tagged enums (geometry `std`, node `type`): list the variants and their fields.
        for v in variants {
            let props = v.get("properties").and_then(Value::as_object);
            let tag = props.and_then(|p| p.iter().find_map(|(k, x)| x.get("const").map(|c| format!("{k}: {c}"))));
            let label = tag.or_else(|| v.get("const").map(|c| c.to_string())).unwrap_or_else(|| "variant".into());
            let fields: Vec<String> = props
                .map(|p| p.iter().filter(|(_, x)| x.get("const").is_none()).map(|(k, _)| k.clone()).collect())
                .unwrap_or_default();
            let desc = first_line(v);
            let _ = writeln!(out, "  {label}{}{}", if fields.is_empty() { String::new() } else { format!("  {{{}}}", fields.join(", ")) }, if desc.is_empty() { String::new() } else { format!("  — {desc}") });
        }
        return;
    }
    if let Some(props) = s.get("properties").and_then(Value::as_object) {
        let required: Vec<&str> = s.get("required").and_then(Value::as_array).map(|r| r.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
        for (k, v) in props {
            let mut line = format!("  {k}: {}", type_name(schema, v));
            if required.contains(&k.as_str()) {
                line.push_str(" (required)");
            } else if let Some(d) = v.get("default") {
                let _ = write!(line, " = {d}");
            }
            let desc = first_line(v);
            if !desc.is_empty() {
                let _ = write!(line, "  — {desc}");
            }
            let _ = writeln!(out, "{line}");
        }
    }
}

/// Help text for a dotted field path, e.g. `camera.fov` or `fluid.fill`.
pub fn field_help(path: &str) -> Result<String> {
    let schema = json_schema();
    let mut out = String::new();
    let segs: Vec<&str> = path.split(['.', '/']).filter(|s| !s.is_empty()).collect();
    if segs.is_empty() {
        let _ = writeln!(out, "scene (top level):");
        describe_object(&schema, &schema, &mut out);
        let names: Vec<&str> = NODE_TYPES.iter().copied().chain(ALIASES.iter().map(|a| a.0)).collect();
        let _ = writeln!(out, "\nlookups: {}", names.join(", "));
        return Ok(out);
    }
    let node_variant = |tag: &str| -> Option<&Value> {
        defs(&schema).get("Node")?.get("oneOf")?.as_array()?.iter().find(|v| {
            v.get("properties").and_then(|p| p.get("type")).and_then(|t| t.get("const")).and_then(Value::as_str) == Some(tag)
        })
    };
    let mut cur: &Value = if NODE_TYPES.contains(&segs[0]) {
        node_variant(segs[0]).expect("node variant exists")
    } else if let Some((_, d)) = ALIASES.iter().find(|(a, _)| *a == segs[0]) {
        defs(&schema).get(*d).expect("alias target exists")
    } else if let Some(p) = schema.get("properties").and_then(|p| p.get(segs[0])) {
        p
    } else {
        bail!(
            "unknown field '{}'; start with one of: {}",
            segs[0],
            NODE_TYPES.iter().copied().chain(ALIASES.iter().map(|a| a.0)).chain(["version", "nodes"]).collect::<Vec<_>>().join(", ")
        );
    };
    let mut leaf_info: Option<&Value> = None;
    for seg in &segs[1..] {
        let obj = resolve(&schema, cur);
        let next = obj.get("properties").and_then(|p| p.get(*seg)).or_else(|| {
            // Tagged enums: look through every variant's properties.
            obj.get("oneOf").and_then(Value::as_array).and_then(|vs| vs.iter().find_map(|v| v.get("properties").and_then(|p| p.get(*seg))))
        });
        match next {
            Some(n) => {
                leaf_info = Some(n);
                cur = n;
            }
            None => {
                let mut fields = String::new();
                describe_object(&schema, obj, &mut fields);
                bail!("'{path}': no field '{seg}' here. Fields:\n{fields}");
            }
        }
    }
    let shown = leaf_info.unwrap_or(cur);
    let _ = write!(out, "{path}: {}", type_name(&schema, shown));
    if let Some(d) = shown.get("default") {
        let _ = write!(out, " = {d}");
    }
    out.push('\n');
    let desc = shown.get("description").or_else(|| resolve(&schema, shown).get("description"));
    if let Some(d) = desc.and_then(Value::as_str) {
        for l in d.lines() {
            let _ = writeln!(out, "  {}", l.trim());
        }
    }
    let mut kids = String::new();
    describe_object(&schema, shown, &mut kids);
    if !kids.is_empty() {
        let _ = write!(out, "fields:\n{kids}");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looks_up_fields() {
        let h = field_help("camera.fov").unwrap();
        assert!(h.contains("40") && h.contains("Vertical field of view"), "{h}");
        let h = field_help("material.inputs.specular_ior").unwrap();
        assert!(h.contains("1.5"), "{h}");
        let h = field_help("geometry").unwrap();
        assert!(h.contains("sphere") && h.contains("bird"), "{h}");
        let h = field_help("fluid").unwrap();
        assert!(h.contains("bounds") && h.contains("fill"), "{h}");
        assert!(field_help("camera.fvo").unwrap_err().to_string().contains("fov"));
        assert!(field_help("").unwrap().contains("nodes"));
    }

    #[test]
    fn schema_validates_examples_shape() {
        let s = json_schema();
        assert!(s.get("$defs").unwrap().get("Node").is_some());
    }
}
