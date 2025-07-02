# main.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from TripletDataset import TripletDataset
from DINMTrainer import DINMTrainer
from DINMDetoxifierEvaluator import DINMDetoxifierEvaluator
from torch.utils.data import DataLoader
from setup_logging import setup_logging

logger = setup_logging()

def main():
    """
    Main orchestration of DINM training and evaluation:
    1. Load (X, Y_unsafe, Y_safe) triplets
    2. Use DINMTrainer to detect toxic layer and edit Wᵛ
    3. Evaluate generated detoxified responses
    """
    try:
        # Set device and model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Load tokenizer and GPT-2 model
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

        # Load triplet data (prompt, unsafe, safe)
        triplet_path = "triplets.jsonl"  # <-- make sure this file exists
        logger.info(f"Loading triplet dataset from {triplet_path}")
        dataset = TripletDataset(triplet_path, tokenizer)
        # Use only first 100 samples for now
        subset = torch.utils.data.Subset(dataset, range(100))
        dataloader = DataLoader(subset, batch_size=4, shuffle=True)

        # Train with DINMTrainer
        logger.info("Starting DINM training with Wᵛ editing and loss L = Lₑ + λL𝒸")
        trainer = DINMTrainer(model=model, lambda_consistency=0.5, device=device)
        #trainer.train(dataloader, epochs=5, lr=1e-4)
        trainer.train(dataloader, epochs=10, lr=5e-4)

        # Save updated model
        model.save_pretrained("gpt2_dinm_edited")
        tokenizer.save_pretrained("gpt2_dinm_edited")
        logger.info("Edited model saved to gpt2_dinm_edited/")

        # Load detoxifier evaluator using modified model
        #detoxifier = lambda hidden: hidden  # identity since detox done at training
        # For each layer ℓ∈{6,8,9}, modify Wvℓ 
        layer_indices = [6, 8, 9]
        from DINMDetoxifier import DINMDetoxifier
        detoxifier = { 
            idx: 
                DINMDetoxifier (
                    model=model,
                    layer_index=idx,  # or the correct toxic layer index you used
                    dampening_factor=0.5,
                    activation_threshold=0.5,
                    std_activations_threshold=0.2,
                    adjusted_threshold=0.8
                ) 
            for idx in layer_indices
        }
        evaluator = DINMDetoxifierEvaluator(model, detoxifier, device)

        # Load a toxicity classifier pipeline
        toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert", device=0 if device == "cuda" else -1)

        # Convert triplet samples to evaluation pairs (unsafe → safe reference)
        eval_pairs = [(sample["unsafe"], sample["safe"]) for sample in dataset.samples[:50]]

        # Run evaluation
        logger.info("Running evaluation on detoxified samples...")
        results = evaluator.evaluate(eval_pairs, toxicity_classifier=toxicity_classifier)

        # Log final metrics
        for key, value in results.items():
            logger.info(f"{key.replace('_', ' ').title()}: {value:.4f}")

        # Optional: Interactive CLI
        logger.info("Entering interactive mode. Type 'quit' to exit.")
        while True:
            user_input = input("\nEnter toxic text (or 'quit'): ").strip()
            if user_input.lower() == "quit":
                logger.info("User exited interactive mode.")
                break
            try:
                print("Original:", evaluator.generate_text(user_input))
                print("Detoxified:", evaluator.detoxify_and_generate(user_input))
            except Exception as e:
                logger.warning(f"Failed to detoxify input: {e}", exc_info=True)

    except Exception as e:
        logger.exception("Fatal error in main execution.")

if __name__ == "__main__":
    main()
