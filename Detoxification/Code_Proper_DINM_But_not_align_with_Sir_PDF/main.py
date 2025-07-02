# main.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from DINMDetoxifier import DINMDetoxifier
from DINMDetoxifierEvaluator import DINMDetoxifierEvaluator
from datasets import load_dataset
from tqdm import tqdm
from setup_logging import setup_logging

logger = setup_logging()

def load_datasets(sample_size=500):
    try:
        toxic_ds = load_dataset("allenai/real-toxicity-prompts", split="train")
        non_toxic_ds = load_dataset("wikitext", "wikitext-103-v1", split="train")

        toxic_texts = [x['prompt'] for x in toxic_ds if 'prompt' in x][:sample_size]
        non_toxic_texts = [x['text'] for x in non_toxic_ds if 'text' in x][:sample_size]

        logger.info(f"Datasets loaded: {len(toxic_texts)} toxic, {len(non_toxic_texts)} non-toxic")
        return toxic_texts, non_toxic_texts
    except Exception as e:
        logger.exception("Failed to load datasets.")
        raise

def get_hidden_activations(model, tokenizer, texts, layer_index=6, max_length=128):
    activations = []
    device = model.device
    model.eval()

    for text in tqdm(texts, desc="Extracting"):
        try:
            if not isinstance(text, str) or not text.strip():
                continue

            encoded = tokenizer.encode_plus(
                text.strip(),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            if outputs.hidden_states is None or len(outputs.hidden_states) <= layer_index:
                logger.warning(f"[SKIP] Missing hidden state for: {text[:30]}...")
                continue

            hidden_states = outputs.hidden_states[layer_index]
            activations.append(hidden_states.cpu())

        except Exception as e:
            logger.warning(f"[SKIP] Error processing text: {repr(text)}", exc_info=True)
            continue

    if not activations:
        logger.error("No valid activations were extracted.")
        raise RuntimeError("No valid activations were extracted. Check your input data or model config.")

    logger.info(f"Extracted activations for {len(activations)} samples.")
    return torch.stack(activations)

def main():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        layer_index = 6

        logger.info(f"Loading model to device: {device}")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        detoxifier = DINMDetoxifier(
            model=model,
            layer_index=layer_index,
            dampening_factor=0.5,
            activation_threshold=0.5,
            std_activations_threshold=0.2,
            adjusted_threshold=0.8
        )

        # Step 1: Load dataset
        logger.info("Loading datasets...")
        toxic_texts, non_toxic_texts = load_datasets(sample_size=200)

        logger.info(f"Sample toxic text: {toxic_texts[0] if toxic_texts else '[EMPTY]'}")

        # Step 2: Extract activations
        logger.info("Extracting hidden states...")
        toxic_acts = get_hidden_activations(model, tokenizer, [x["text"] for x in toxic_texts], layer_index)
        non_toxic_acts = get_hidden_activations(model, tokenizer, non_toxic_texts, layer_index)

        # Step 3: Train detoxifier
        logger.info("Training detoxifier...")
        detoxifier.train_detoxifier(toxic_acts, non_toxic_acts, epochs=5, lr=1e-3)

        # Step 4: Save weights
        detoxifier.save_weights("dinm_weights.pth")
        logger.info("Weights saved to dinm_weights.pth")

        # Step 5: Evaluate
        logger.info("Evaluating detoxification...")
        evaluator = DINMDetoxifierEvaluator(model, detoxifier, device)
        toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert", device=0 if device == "cuda" else -1)

        pairs = list(zip(
            [x["text"] for x in toxic_texts[:50]],
            non_toxic_texts[:50]
        ))

        results = evaluator.evaluate(pairs, toxicity_classifier=toxicity_classifier)

        for key, value in results.items():
            logger.info(f"{key.replace('_', ' ').title()}: {value:.4f}")

        # Step 6: Interactive
        while True:
            text = input("\nEnter toxic text (or 'quit'): ").strip()
            if text.lower() == "quit":
                logger.info("Exiting interactive session.")
                break
            logger.info("Generating response...")
            print("Original:", evaluator.generate_text(text))
            print("Detoxified:", evaluator.detoxify_and_generate(text))

    except Exception as e:
        logger.exception("An error occurred during execution.")

if __name__ == "__main__":
    main()
