# DINMDetoxifier.py
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from setup_logging import setup_logging

logger = setup_logging()

class DINMDetoxifier:
    def __init__(self, model, layer_index, dampening_factor, activation_threshold, std_activations_threshold, adjusted_threshold):
        self.model = model
        self.layer_index = layer_index
        self.dampening_factor = torch.tensor(dampening_factor)
        self.activation_threshold = torch.tensor(activation_threshold)
        self.std_activations_threshold = torch.tensor(std_activations_threshold)
        self.adjusted_threshold = torch.tensor(adjusted_threshold)

        self.toxicity_weights = nn.Parameter(torch.randn(model.config.hidden_size))
        self.toxicity_bias = nn.Parameter(torch.zeros(1))

        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load GPT2 tokenizer.")

    def __call__(self, hidden_activations):
        return self.detoxify_hidden_layer(hidden_activations)

    def parameters(self):
        return [self.toxicity_weights, self.toxicity_bias]

    def train_detoxifier(self, toxic_activations, non_toxic_activations, epochs=10, lr=0.01):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            total_loss = 0
            try:
                for toxic, non_toxic in zip(toxic_activations, non_toxic_activations):
                    toxic_pred = self._predict_toxicity(toxic)
                    non_toxic_pred = self._predict_toxicity(non_toxic)
                    loss = torch.mean(non_toxic_pred) - torch.mean(toxic_pred)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(toxic_activations)
                logger.info(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

            except Exception as e:
                logger.exception(f"Training failed during epoch {epoch+1}")

    def _predict_toxicity(self, activations):
        mean_activations = activations.mean(dim=(0, 1))
        return torch.sigmoid(torch.dot(mean_activations, self.toxicity_weights) + self.toxicity_bias)

    def identify_toxic_neurons(self, hidden_activations):
        mean_activations = hidden_activations.mean(dim=(0, 1))
        if hidden_activations.shape[1] > 1:
            std_activations = hidden_activations.std(dim=(0, 1))
            result = (mean_activations > self.activation_threshold) & (std_activations > self.std_activations_threshold)
        else:
            result = mean_activations > (self.activation_threshold * self.adjusted_threshold)

        logger.debug(f"Toxic neurons identified: {result.sum().item()}")
        return result

    def detoxify_hidden_layer(self, hidden_activations):
        try:
            toxicity_scores = torch.sigmoid(hidden_activations * self.toxicity_weights + self.toxicity_bias)
            toxic_neurons = self.identify_toxic_neurons(hidden_activations)
            detoxified = hidden_activations.clone()
            detoxified[:, :, toxic_neurons] *= torch.sigmoid(self.dampening_factor)
            detoxified = torch.nan_to_num(detoxified)  # Safety: replace NaNs
            logger.debug("Detoxified hidden layer applied.")
            return detoxified
        except Exception as e:
            logger.exception("Detoxification failed.")
            return hidden_activations  # fallback

    def calculate_perplexity(self, text):
        if not isinstance(text, str) or not text.strip():
            logger.warning("Invalid input text for perplexity calculation.")
            return float("inf")

        try:
            inputs = self.tokenizer(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True
            ).to(self.model.device)

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

            perplexity = torch.exp(outputs.loss).item()
            logger.info(f"Perplexity: {perplexity:.4f}")
            return perplexity

        except Exception as e:
            logger.warning(f"Perplexity calculation failed for input: {repr(text)}", exc_info=True)
            return float("inf")

    def save_weights(self, path="dinm_weights.pth"):
        try:
            torch.save({
                'weights': self.toxicity_weights.detach().cpu(),
                'bias': self.toxicity_bias.detach().cpu()
            }, path)
            logger.info(f"Saved detoxifier weights to {path}")
        except Exception as e:
            logger.exception(f"Failed to save weights to {path}")

    def load_weights(self, path="dinm_weights.pth"):
        try:
            state = torch.load(path)
            self.toxicity_weights.data.copy_(state['weights'])
            self.toxicity_bias.data.copy_(state['bias'])
            logger.info(f"Loaded detoxifier weights from {path}")
        except Exception as e:
            logger.exception(f"Failed to load weights from {path}")
