# evaluate_dinm.py

import torch
import json
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from DINMDetoxifierEvaluator import DINMDetoxifierEvaluator
from setup_logging import setup_logging

logger = setup_logging()

def load_triplets(path):
    """
    Load detoxification evaluation triplets: (X, Y_unsafe, Y_safe)

    Returns:
        List of (toxic, safe) tuples to evaluate on
    """
    pairs = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if "unsafe" in data and "safe" in data:
                    pairs.append((data["unsafe"], data["safe"]))
        logger.info(f"Loaded {len(pairs)} triplet pairs from {path}")
    except Exception as e:
        logger.exception(f"Failed to load triplets from {path}")
        raise
    return pairs

def save_results(results, output_path):
    """
    Save final evaluation results to JSON

    Parameters:
        results: dict of evaluation scores
        output_path: output .json path
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")
    except Exception as e:
        logger.warning(f"Could not save results to {output_path}", exc_info=True)

def plot_metrics(metrics, output_path="dinm_eval_plot.png"):
    """
    Create bar plot of DINM detoxification performance.

    Parameters:
        metrics (dict): contains similarity_gain, fluency_ratio, toxicity_reduction
        output_path (str): PNG path
    """
    try:
        labels = [key.replace('_', ' ').title() for key in metrics.keys()]
        values = list(metrics.values())

        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, values, color=["#4472c4", "#70ad47", "#ed7d31"])
        plt.title("DINM Detoxification Evaluation")
        plt.ylabel("Metric Value")
        plt.ylim(0, max(values) * 1.2)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height, f"{height:.2f}", ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved plot to {output_path}")
    except Exception as e:
        logger.warning("Failed to generate evaluation plot", exc_info=True)

def main():
    """
    Main evaluation script for DINM-based detoxifier model.

    Workflow:
        1. Load edited GPT2 model + tokenizer
        2. Load evaluation triplets (toxic → reference safe)
        3. Run detoxification evaluation
        4. Log + save metrics and graphs
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {device}")

        # Load edited model
        model_path = "gpt2_dinm_edited"
        logger.info(f"Loading edited GPT2 model from: {model_path}")
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)

        # Setup evaluator with identity detoxifier (inference-time layer edited already)
        evaluator = DINMDetoxifierEvaluator(model, detoxifier=lambda h: h, device=device)

        # Load classifier
        toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert", device=0 if device == "cuda" else -1)

        # Load triplet pairs
        pair_path = "triplets.jsonl"
        pairs = load_triplets(pair_path)

        # Evaluate
        logger.info("Running DINM evaluation...")
        results = evaluator.evaluate(pairs[:50], toxicity_classifier=toxicity_classifier)

        for key, value in results.items():
            logger.info(f"{key.replace('_', ' ').title()}: {value:.4f}")

        # Save & Plot
        save_results(results, "dinm_eval_results.json")
        plot_metrics({k: v for k, v in results.items() if k != "success_rate"})

    except Exception as e:
        logger.exception("DINM evaluation failed.")

if __name__ == "__main__":
    main()
