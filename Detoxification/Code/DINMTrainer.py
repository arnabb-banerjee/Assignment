"""
Purpose of DINMTrainer.py

This trainer handles:
1. Toxic Layer Identification -> Detects the most altered layer between Ysafe and Yunsafe (Δ hidden states)
2. Wᵛ Editing -> Only updates the value projection layer in transformer block ℓ
3. DINM Loss Functions -> Implements: Lₑ (edit) + L𝒸 (consistency)
4. Logging + Comments -> Detailed logs, exception handling, and math explanations
"""
# DINMTrainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from setup_logging import setup_logging

logger = setup_logging()

class DINMTrainer:
    """
    DINMTrainer aligns with the ACL 2024 DINM paper to edit only the value projection Wᵛ_ℓ
    of a transformer layer ℓ, based on semantic contrast between Y_unsafe and Y_safe.
    
    It implements:
        - ℓ = argmax_i ||hᵢ(Y_safe) - hᵢ(Y_unsafe)|| (Toxic Layer Identification)
        - L_e = CrossEntropy(P(y|x, θ_edited), Y_safe)
        - L_c = KL(P(y|x, θ_edited) || P(y|x, θ_original))   ← consistency
    """

    def __init__(self, model: GPT2LMHeadModel, layer_index=None, lambda_consistency=1.0, device="cpu"):
        """
        Parameters:
            model: GPT2LMHeadModel (transformers)
            layer_index: int or None. If None, DINMTrainer will auto-detect toxic layer ℓ
            lambda_consistency: weight for L_c in final loss
            device: torch device
        """
        self.model = model.to(device)
        self.device = device
        self.layer_index = layer_index
        self.lambda_consistency = lambda_consistency

        self.original_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.original_model.eval()  # Used for computing L_c without gradients

        logger.info(f"DINMTrainer initialized. Layer to edit: {self.layer_index if self.layer_index is not None else 'AUTO'}")

    def find_toxic_layer(self, dataloader):
        """
        Auto-select toxic layer ℓ based on max mean L2 distance between hidden states
        of Y_safe and Y_unsafe. This matches DINM Equation 6.

        Returns:
            layer_index (int): most semantically shifted layer
        """
        logger.info("Identifying toxic layer via hidden state distance...")
        num_layers = self.model.config.n_layer
        layer_diffs = torch.zeros(num_layers).to(self.device)

        with torch.no_grad():
            for batch in dataloader:
                x = batch['input_ids'].to(self.device)
                m = batch['attention_mask'].to(self.device)
                y_safe = batch['labels_safe'].to(self.device)
                y_unsafe = batch['labels_unsafe'].to(self.device)

                # Run model on Y_safe and Y_unsafe (generation side)
                out_safe = self.model(input_ids=x, attention_mask=m, labels=y_safe, output_hidden_states=True)
                out_unsafe = self.model(input_ids=x, attention_mask=m, labels=y_unsafe, output_hidden_states=True)

                for i in range(num_layers):
                    h1 = out_safe.hidden_states[i].detach()
                    h2 = out_unsafe.hidden_states[i].detach()
                    diff = torch.norm(h1 - h2, p=2, dim=-1).mean()
                    layer_diffs[i] += diff

        selected_layer = torch.argmax(layer_diffs).item()
        logger.info(f"Most shifted layer (toxic layer): {selected_layer}")
        return selected_layer

    def freeze_except_value_projection(self):
        """
        Freezes all parameters except the value projection W_v in MLP of selected layer ℓ
        """
        logger.info("Freezing all model parameters except W_v in MLP of layer ℓ...")

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        block = self.model.transformer.h[self.layer_index]
        for name, param in block.mlp.named_parameters():
            if 'c_proj' in name:
                param.requires_grad = True
                logger.debug(f"Editing param: h.{self.layer_index}.mlp.{name}")

    def train(self, dataloader: DataLoader, epochs=5, lr=1e-4):
        """
        Trains the model to minimize L = L_e + λL_c

        Parameters:
            dataloader: DataLoader yielding batch dict with input_ids, attention_mask, labels_safe, labels_unsafe
            epochs: number of training epochs
            lr: learning rate
        """
        if self.layer_index is None:
            self.layer_index = self.find_toxic_layer(dataloader)

        self.freeze_except_value_projection()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id  # safe fallback

        ce_loss_fn = nn.CrossEntropyLoss(ignore_index=self.model.config.pad_token_id, reduction="mean")
        kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

        for epoch in range(epochs):
            total_loss = 0.0
            self.model.train()

            for batch in dataloader:
                try:
                    x = batch["input_ids"].to(self.device)
                    m = batch["attention_mask"].to(self.device)
                    y_safe = batch["labels_safe"].to(self.device)
                    y_unsafe = batch["labels_unsafe"].to(self.device)

                    # Get logits from edited model
                    out_edited = self.model(input_ids=x, attention_mask=m)
                    logits_edited = out_edited.logits

                    # L_e = cross entropy to safe targets
                    l_e = ce_loss_fn(logits_edited.view(-1, logits_edited.size(-1)), y_safe.view(-1))

                    # L_c = consistency to original model outputs on unsafe target
                    with torch.no_grad():
                        out_original = self.original_model(input_ids=x, attention_mask=m)
                        logits_orig = out_original.logits

                    p_orig = torch.nn.functional.log_softmax(logits_orig, dim=-1)
                    p_edited = torch.nn.functional.softmax(logits_edited, dim=-1)
                    l_c = kl_loss_fn(p_orig, p_edited)

                    loss = l_e + self.lambda_consistency * l_c

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                except Exception as e:
                    logger.warning(f"Training batch failed: {e}", exc_info=True)
                    continue

            avg_loss = total_loss / len(dataloader)
            #logger.info(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
