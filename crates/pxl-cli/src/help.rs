use anyhow::{bail, Result};

const TOPICS: &[(&str, &str)] = &[
    ("index", include_str!("../../../docs/help/index.md")),
    ("scene", include_str!("../../../docs/help/scene.md")),
    ("render", include_str!("../../../docs/help/render.md")),
    ("programs", include_str!("../../../docs/help/programs.md")),
    ("simulations", include_str!("../../../docs/help/simulations.md")),
];

pub fn print(topic: Option<&str>) -> Result<()> {
    let t = topic.unwrap_or("index");
    if t == "--all" || t == "all" {
        for (_, body) in TOPICS {
            println!("{body}");
        }
        return Ok(());
    }
    match TOPICS.iter().find(|(name, _)| *name == t) {
        Some((_, body)) => {
            println!("{body}");
            Ok(())
        }
        None => bail!(
            "unknown help topic '{t}'; topics: {}",
            TOPICS.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
        ),
    }
}
