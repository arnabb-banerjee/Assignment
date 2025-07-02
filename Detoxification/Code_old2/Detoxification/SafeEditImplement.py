import torch
import re
from typing import List, Dict
from transformers import GPT2Tokenizer

class SafeEditImplement:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.banned_phrases = self._load_banned_phrases()
        self.safe_replacements = self._load_safe_replacements()

    def _load_banned_phrases(self) -> List[str]:
        return [
            "fuck",
            "shit",
            "bitch",
            "kill",
            "hurt",
            "rape",
            "asshole",
            "dick",
            "pussy",
            "bastard",
        ]

    def _load_safe_replacements(self) -> Dict[str, str]:
        return {
            "fuck": "express feelings",
            "shit": "stuff",
            "bitch": "person",
            "kill": "stop",
            "hurt": "affect",
            "rape": "harm",
            "asshole": "jerk",
            "dick": "person",
            "pussy": "cat",
            "bastard": "individual",
        }

    def contains_unsafe_content(self, text: str) -> bool:
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in self.banned_phrases)

    def apply_safe_replacements(self, text: str) -> str:
        for phrase, replacement in self.safe_replacements.items():
            text = re.sub(
                r"\b" + re.escape(phrase) + r"\b",
                replacement,
                text,
                flags=re.IGNORECASE,
            )
        return text

    def get_banned_word_ids(self) -> List[List[int]]:
        return [
            self.tokenizer.encode(phrase, add_special_tokens=False)
            for phrase in self.banned_phrases
        ]
