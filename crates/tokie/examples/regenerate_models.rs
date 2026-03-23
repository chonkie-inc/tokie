//! Regenerate all .tkz model files with the current serialization format.
//!
//! This updates models to include new fields like normalizer and post_processor.

use std::path::Path;
use tokie::hf;
use tokie::PretokType;

fn regenerate(
    name: &str,
    json_path: &str,
    download_url: &str,
    pretok: Option<PretokType>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Download if not present
    if !Path::new(json_path).exists() {
        println!("Downloading {} tokenizer...", name);
        let status = std::process::Command::new("curl")
            .args(["-L", "-o", json_path, download_url])
            .status()?;
        if !status.success() {
            return Err(format!("Failed to download {}", name).into());
        }
    }

    println!("Regenerating {}...", name);

    let tokenizer = match pretok {
        Some(pt) => hf::from_json_with_pretokenizer(json_path, pt)?,
        None => hf::from_json(json_path)?,
    };

    println!(
        "  Loaded: {} tokens, {:?}, {:?}",
        tokenizer.vocab_size(),
        tokenizer.encoder_type(),
        tokenizer.pretokenizer_type(),
    );
    println!(
        "  Normalizer: {:?}, PostProcessor: {:?}",
        tokenizer.normalizer(),
        tokenizer.post_processor()
    );

    let path = format!("models/{}.tkz", name);
    tokenizer.to_file(&path)?;
    println!("  Saved to {}", path);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ensure models directory exists
    std::fs::create_dir_all("models")?;

    regenerate(
        "gpt2",
        "/tmp/gpt2_tokenizer.json",
        "https://huggingface.co/gpt2/raw/main/tokenizer.json",
        None,
    )?;

    regenerate(
        "cl100k",
        "/tmp/cl100k_tokenizer.json",
        "https://huggingface.co/Xenova/gpt-4/raw/main/tokenizer.json",
        Some(PretokType::Cl100k),
    )?;

    regenerate(
        "o200k",
        "/tmp/o200k_tokenizer.json",
        "https://huggingface.co/Xenova/gpt-4o/raw/main/tokenizer.json",
        Some(PretokType::O200k),
    )?;

    regenerate(
        "bert",
        "/tmp/bert_tokenizer.json",
        "https://huggingface.co/bert-base-uncased/raw/main/tokenizer.json",
        None,
    )?;

    regenerate(
        "llama3",
        "/tmp/llama3_tokenizer.json",
        "https://huggingface.co/meta-llama/Llama-3.2-1B/raw/main/tokenizer.json",
        None,
    )?;

    println!("\nAll models regenerated successfully!");
    Ok(())
}
