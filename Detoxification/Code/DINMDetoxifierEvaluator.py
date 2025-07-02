# DINMDetoxifierEvaluator.py
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.spatial.distance import cosine
from setup_logging import setup_logging

logger = setup_logging()

class DINMDetoxifierEvaluator:
    def __init__(self, model: GPT2LMHeadModel, detoxifier, device=None):
        self.model = model.to(device or "cpu")
        self.detoxifier = detoxifier
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def generate_text(self, input_text: str, max_length: int = 50) -> str:
        try:
            inputs = self.tokenizer(text=input_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Generated text: {generated}")
            return generated
        except Exception as e:
            logger.warning(f"Text generation failed: {e}", exc_info=True)
            return ""

    def detoxify_and_generate(self, input_text: str, max_length: int = 50) -> str:
        try:
            inputs = self.tokenizer(text=input_text, return_tensors="pt").to(self.device)
            original_forward = self.model.transformer.h[self.detoxifier.layer_index].forward

            def patched_forward(*args, **kwargs):
                output = original_forward(*args, **kwargs)
                hidden_states = output[0] if isinstance(output, tuple) else output

                if hidden_states is None or hidden_states.shape[1] == 0 or torch.isnan(hidden_states).any():
                    logger.warning("Skipping detox due to invalid hidden state.")
                    return output

                detoxified = self.detoxifier(hidden_states)
                if detoxified.shape != hidden_states.shape:
                    logger.warning("Detoxified shape mismatch — fallback to original.")
                    return output

                return (detoxified,) + output[1:] if isinstance(output, tuple) else detoxified

            self.model.transformer.h[self.detoxifier.layer_index].forward = patched_forward
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            self.model.transformer.h[self.detoxifier.layer_index].forward = original_forward
            detoxed = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Detoxified text: {detoxed}")
            return detoxed
        except Exception as e:
            logger.warning(f"Detoxified generation failed: {e}", exc_info=True)
            return ""

    def calculate_similarity(self, text1: str, text2: str) -> float:
        try:
            emb1 = self._get_text_embedding(text1)
            emb2 = self._get_text_embedding(text2)
            similarity = 1 - cosine(emb1, emb2)
            logger.info(f"Cosine similarity: {similarity:.4f}")
            return similarity
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}", exc_info=True)
            return 0.0

    def _get_text_embedding(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            return np.zeros(self.model.config.hidden_size)

        try:
            inputs = self.tokenizer(text=text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.transformer(**inputs, output_hidden_states=True)
            return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        except Exception as e:
            logger.warning(f"Failed to embed text: {repr(text)} — {e}", exc_info=True)
            return np.zeros(self.model.config.hidden_size)

    def calculate_perplexity(self, text: str) -> float:
        if not isinstance(text, str) or not text.strip():
            return float("inf")
        try:
            inputs = self.tokenizer(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            perplexity = torch.exp(outputs.loss).item()
            logger.info(f"Perplexity: {perplexity:.4f}")
            return perplexity
        except Exception as e:
            logger.warning(f"Perplexity calculation failed for input: {repr(text)} — {e}", exc_info=True)
            return float("inf")

    def evaluate(self, text_pairs, num_samples=3, toxicity_classifier=None):
        similarity_gains = []
        toxicity_reductions = []
        fluency_ratios = []
        success_count = 0

        logger.info(f"Starting evaluation with {len(text_pairs)} text pairs, {num_samples} samples each.")
        for toxic, reference in text_pairs:
            for _ in range(num_samples):
                orig = self.generate_text(toxic)
                detox = self.detoxify_and_generate(toxic)

                # Similarity
                orig_sim = self.calculate_similarity(orig, reference)
                detox_sim = self.calculate_similarity(detox, reference)
                similarity_gains.append(detox_sim - orig_sim)

                # Toxicity
                if toxicity_classifier:
                    try:
                        orig_tox = toxicity_classifier(orig, truncation=True)[0]['score']
                        detox_tox = toxicity_classifier(detox, truncation=True)[0]['score']
                        toxicity_reductions.append(orig_tox - detox_tox)
                        if detox_tox < orig_tox:
                            success_count += 1
                        logger.info(f"Toxicity reduction: {orig_tox:.3f} → {detox_tox:.3f}")
                    except Exception as e:
                        logger.warning(f"Toxicity scoring failed: {e}", exc_info=True)

                # Fluency
                orig_ppl = self.calculate_perplexity(orig)
                detox_ppl = self.calculate_perplexity(detox)
                fluency_ratio = orig_ppl / max(detox_ppl, 1e-6)
                fluency_ratios.append(fluency_ratio)
                logger.info(f"Fluency ratio: {fluency_ratio:.3f}")

        results = {
            "similarity_gain": np.mean(similarity_gains),
            "fluency_ratio": np.mean(fluency_ratios),
        }

        if toxicity_classifier:
            results.update({
                "toxicity_reduction": np.mean(toxicity_reductions),
                "success_rate": success_count / (len(text_pairs) * num_samples)
            })

        logger.info("Evaluation completed.")
        logger.info(f"Results: {results}")
        return results
