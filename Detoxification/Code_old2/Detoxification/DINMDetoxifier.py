import torch 
import logging
from typing import Optional, Tuple

class DINMDetoxifier:
    
    def __init__(self, model, dampening_factor: float, activation_threshold: float, std_activations_threshold: float, Adjusted_threshold: float):
        self.model = model
        self.dampening_factor = self._validate_factor(dampening_factor)
        self.activation_threshold = self._validate_threshold(activation_threshold)  # Used to flag 'toxic' neurons
        self.std_activations_threshold = self._validate_factor(std_activations_threshold)
        self.Adjusted_threshold = self._validate_factor(Adjusted_threshold)


    def _validate_factor(self, factor: float) -> float:
        if not 0 <= factor <= 1:
            raise ValueError("Dampening factor must be between 0 and 1")
        return factor

    def _validate_threshold(self, threshold: float) -> float:
        if threshold < 0:
            raise ValueError("Activation threshold must be non-negative")
        return threshold

    def identify_toxic_neurons(self, hidden_activations: torch.Tensor) -> torch.Tensor:
        # Identify potentially toxic neurons based on activation magnitude.
        # Args: hidden_activations: Tensor of shape (batch_size, seq_len, hidden_dim)
        # Returns: Boolean mask of shape (hidden_dim,) indicating which neurons to dampen """"

        # Compute mean activation across batch and sequence
        # dim=(0, 1) means you are taking the mean across both the batch dimension (dim=0) and the sequence/token dimension (dim=1).
        # As a result, you get one average value per hidden neuron (along dim=2), which reflects how each neuron behaves on average across the entire input text and batch.
        # This helps you understand the average activation pattern of the model, and is useful for identifying or correcting neurons that may be overly active or associated with toxic behavior.
        # Let say hidden_activations.shape == (1, 10, 768)
        #   1: Batch size
        #   10: Sequence length (number of tokens)
        #   768: Number of hidden units (GPT-2 has 768 units per layer)
        #   So, when you compute mean(dim=(0, 1)), the output will be a tensor of shape [768].
        #   This represents the average activation of each hidden unit, calculated over the entire batch and the full
        mean_activations = hidden_activations.mean(dim=(0, 1))  # shape: (hidden_dim,)


        # Only calculate std if we have sufficient sequence length
        if hidden_activations.shape[1] > 1:  # More than 1 token
            std_activations = hidden_activations.std(dim=(0, 1))
            toxic_neurons = (mean_activations > self.activation_threshold) & (std_activations > self.std_activations_threshold)  # Lower threshold for short texts
        else:
            toxic_neurons = mean_activations > self.activation_threshold * self.Adjusted_threshold  # Adjusted threshold

        return toxic_neurons

    # Apply detoxification by dampening the activations of identified toxic neurons.
    # Args: hidden_activations: Tensor of shape (batch_size, seq_len, hidden_dim)
    # Returns: Detoxified hidden activations tensor
    def detoxify_hidden_layer(self, hidden_activations: torch.Tensor) -> torch.Tensor:
        # Ensure we have a tensor (handle cases where it might be None or wrong type)
        if not isinstance(hidden_activations, torch.Tensor):
            raise ValueError("hidden_activations must be a torch.Tensor")

        # Validate tensor dimensions
        if hidden_activations.dim() != 3:
            raise ValueError(f"Expected 3D tensor (batch, seq, hidden), got {hidden_activations.dim()}D")

        """if isinstance(hidden_activations, tuple):
            hidden_activations = hidden_activations[0]

        if isinstance(hidden_activations, list):
            hidden_activations = torch.stack(hidden_activations) """

        # Core detoxification logic (can be expanded)
        toxic_neurons = self.identify_toxic_neurons(hidden_activations)

        # Create a mask to dampen only toxic neurons
        detoxified = hidden_activations.clone()

        # Dummy detoxification logic (replace with real manipulation logic)
        # Scalar | 0D Tensor | () | 42
        # Vector | 1D Tensor | (n,) | [1.0, 2.0, 3.0]
        # Matrix | 2D Tensor | (m, n) | [[1, 2], [3, 4]]
        # 3D Tensor | 3D Tensor | (batch_size, sequence_length, hidden_dim) | Used in NLP models like GPT-2
        # ND Tensor | nD Tensor | (d1, d2, ..., dn) | For complex tasks like video, images, audio
        #   tensor[0, :, :]	First batch, all tokens, all hidden units
        #   tensor[:, 0, :]	All batches, first token, all hidden units
        #   tensor[:, :, 0]	All batches, all tokens, first hidden unit
        #   tensor[:, :, 100:200]	All batches, all tokens, only hidden units 100–199
        # For example: scale down certain dimensions or zero out known toxic features
        detoxified[:, :, toxic_neurons] *= self.dampening_factor

        return detoxified
