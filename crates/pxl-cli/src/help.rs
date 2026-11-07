use anyhow::{bail, Result};

const TOPICS: &[(&str, &str)] = &[
    ("index", include_str!("../../../docs/help/index.md")),
    ("scene", include_str!("../../../docs/help/scene.md")),
    ("render", include_str!("../../../docs/help/render.md")),
    ("programs", include_str!("../../../docs/help/programs.md")),
    ("simulations", include_str!("../../../docs/help/simulations.md")),
    ("vfx", include_str!("../../../docs/help/vfx.md")),
    ("fluids", include_str!("../../../docs/help/fluids.md")),
];

/// Rough token estimate (~4 characters per token for English + JSON).
pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

pub fn print(topic: Option<&str>, rest: &[String]) -> Result<()> {
    let t = topic.unwrap_or("index");
    match t {
        "--all" | "all" => {
            for (_, body) in TOPICS {
                println!("{body}");
            }
            Ok(())
        }
        "field" | "fields" => {
            print!("{}", pxl_scene::reference::field_help(&rest.join("."))?);
            Ok(())
        }
        "sizes" => {
            for (name, body) in TOPICS {
                println!("{name:<12} ~{} tokens", estimate_tokens(body));
            }
            Ok(())
        }
        _ => match TOPICS.iter().find(|(name, _)| *name == t) {
            Some((_, body)) => {
                println!("{body}");
                Ok(())
            }
            None => bail!(
                "unknown help topic '{t}'; topics: {}, field <path>, all, sizes",
                TOPICS.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
            ),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Keep the first help page short: ~2,500 tokens including a complete scene.
    #[test]
    fn first_page_fits_token_budget() {
        let index = TOPICS[0].1;
        assert!(estimate_tokens(index) <= 2500, "index page is ~{} tokens", estimate_tokens(index));
        assert!(index.contains("\"type\":\"camera\""), "first page must include a complete scene");
    }

    #[test]
    fn every_topic_mentioned_on_first_page() {
        let index = TOPICS[0].1;
        for (name, _) in &TOPICS[1..] {
            assert!(index.contains(&format!("pxl help {name}")), "index does not point to '{name}'");
        }
        assert!(index.contains("pxl help field"), "index should mention field lookups");
    }
}
