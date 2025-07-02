
""" Phase 1: TripletDataset.py — Supervised Triplet Loader
This module:

1. Loads datasets with (X, Y_unsafe, Y_safe) triplets
2. Applies tokenizer
3. Returns tensors suitable for supervised training in DINM

Based on the DINM paper:
“We utilize (X, Yₛₐ𝒻ₑ, Yᵤₙₛₐ𝒻ₑ) triplets to guide our intervention towards improving detoxification fluency without compromising model coherence or general language ability."""


# TripletDataset.py
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import json
from setup_logging import setup_logging

logger = setup_logging()

class TripletDataset(Dataset):
    """
    A PyTorch Dataset for loading (X, Y_unsafe, Y_safe) triplet data.

    These triplets are used to guide detoxification training in the DINM method
    by computing losses between the LLM outputs on Y_safe vs Y_unsafe conditioned on the same X.

    Mathematically:
        Let X be the prompt.
        Y_unsafe = toxic generation from base LLM.
        Y_safe   = detoxified reference.
        
        DINM uses:
            - L_e = CE(P(y|x;θ_edited), Y_safe)
            - L_c = KL(P(y|x;θ_edited) || P(y|x;θ_original))  ← consistency on clean/general inputs
    """

    def __init__(self, path: str, tokenizer: PreTrainedTokenizer, max_length: int = 128):
        """
        Parameters:
            path (str): Path to JSONL file with triplets: {"prompt": ..., "unsafe": ..., "safe": ...}
            tokenizer (PreTrainedTokenizer): A tokenizer to encode texts
            max_length (int): Max input token length for truncation
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        ex = json.loads(line)
                        if all(k in ex for k in ("prompt", "unsafe", "safe")):
                            self.samples.append(ex)
                        else:
                            logger.warning(f"Skipping incomplete sample: {line}")
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping corrupt JSON line.")
        except Exception as e:
            logger.exception(f"Failed to load triplet dataset from {path}")
            raise

        logger.info(f"Loaded {len(self.samples)} triplet samples from {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            A dict with tokenized prompt, safe target, and unsafe target:
            {
                'input_ids': ...,        # tokenized X (prompt)
                'attention_mask': ...,   # attention mask
                'labels_safe': ...,      # tokenized Y_safe
                'labels_unsafe': ...     # tokenized Y_unsafe
            }
        """
        sample = self.samples[idx]

        prompt = sample["prompt"]
        y_safe = sample["safe"]
        y_unsafe = sample["unsafe"]

        try:
            encoded_prompt = self.tokenizer(prompt, return_tensors="pt", padding="max_length",
                                            truncation=True, max_length=self.max_length)
            encoded_safe = self.tokenizer(y_safe, return_tensors="pt", padding="max_length",
                                          truncation=True, max_length=self.max_length)
            encoded_unsafe = self.tokenizer(y_unsafe, return_tensors="pt", padding="max_length",
                                            truncation=True, max_length=self.max_length)

            return {
                "input_ids": encoded_prompt["input_ids"].squeeze(0),
                "attention_mask": encoded_prompt["attention_mask"].squeeze(0),
                "labels_safe": encoded_safe["input_ids"].squeeze(0),
                "labels_unsafe": encoded_unsafe["input_ids"].squeeze(0)
            }

        except Exception as e:
            logger.warning(f"Error encoding sample at index {idx}: {e}", exc_info=True)
            return {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
                "labels_safe": torch.zeros(self.max_length, dtype=torch.long),
                "labels_unsafe": torch.zeros(self.max_length, dtype=torch.long)
            }
