//! Verify all .tkz models load correctly with post-processor data.

use tokie::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = ["gpt2", "cl100k", "o200k", "bert", "llama3"];

    for name in models {
        let path = format!("models/{}.tkz", name);
        println!("Loading {}...", path);

        let tokenizer = Tokenizer::from_file(&path)?;

        println!(
            "  {} tokens, {:?}, {:?}",
            tokenizer.vocab_size(),
            tokenizer.encoder_type(),
            tokenizer.pretokenizer_type(),
        );
        println!(
            "  Normalizer: {:?}",
            tokenizer.normalizer()
        );
        println!(
            "  PostProcessor: {:?}",
            tokenizer.post_processor()
        );

        // Test encoding
        let text = "Hello";
        let tokens_no_special = tokenizer.encode(text, false);
        let tokens_with_special = tokenizer.encode(text, true);

        println!(
            "  encode(\"{}\", false) = {:?}",
            text, tokens_no_special
        );
        println!(
            "  encode(\"{}\", true)  = {:?}",
            text, tokens_with_special
        );
        println!();
    }

    println!("All models verified successfully!");
    Ok(())
}
