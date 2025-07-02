import torch
from DINMDetoxifier import DINMDetoxifier
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper
import torch.nn.functional as F

class HiddenLayerCapture:
    def __init__(self, model_name="gpt2", layer_index=6, dampening_factor: float = 0.7, activation_threshold: float = 0.8):

        # Downloads a pretrained GPT-2 model with its weights.
        # Includes the LM Head, i.e., the final layer for language modeling: predicting the next token.
        # GPT-2 is a decoder-only Transformer used for causal language modeling.
        # It learns to predict the next token given previous ones.
        # Given a sequence of token IDs: x = [x1, x2, ..., xn]
        # GPT-2 learns: P(xt | x1, x2, ..., x(t-1))
        # Using
        # 1. Embedding Layer: Converts token IDs to dense vectors.
        # 2. Transformer Decoder Blocks: With self-attention and feedforward layers.
        # 3. LM Head (output layer): Maps final hidden state to vocabulary logits.
        # Final Output:
        #    The model outputs:
        #    logits: A matrix of shape (batch_size, sequence_length, vocab_size)
        #    These logits are used to compute probabilities over the vocabulary for next token prediction.
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # Load model and tokenizer
        # Tokenizers convert natural language (text) into a sequence of tokens 
        # (typically integers) that a model like GPT-2 can understand.
        # 1. Downloads and loads the vocabulary and tokenization rules used by the GPT-2 model.
        # 2. Allows to encode/decode text exactly like it was done during GPT-2 training.
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token
        self.layer_index = layer_index
        self.captured_activations = {}
        self.dampening_factor = dampening_factor
        self.activation_threshold = activation_threshold
        self.original_activations = None
        self.detoxified_activations = None
        self.original_output = None
        self.detoxified_output = None
        self.hook_handle = None
        self.detoxifier = DINMDetoxifier(self.model, dampening_factor, activation_threshold)


        
        
    def _hook_fn(self, module, input, output):
        # GPT-2 layers return tuples where output[0] is the hidden states
        if isinstance(output, tuple):
            self.original_activations = output[0].detach().clone()
            # Detoxify and store modified activations
            detoxified = self.detoxifier.detoxify_hidden_layer(output[0])
            self.detoxified_activations = detoxified.detach().clone()
            return (detoxified,) + output[1:]
        else:
            # Fallback for non-tuple outputs (shouldn't happen with GPT-2)
            self.original_activations = output
            detoxified_output = self.detoxifier.detoxify_hidden_layer(output)
            self.captured_activations[f"layer{self.layer_index}"] = detoxified_output
            return detoxified_output # here to implement detoxifying behavior


    def register_hook(self):
        # Register the hook on layer of selected index [10]
        # This line registers a forward hook on the 11th transformer block in GPT-2’s model architecture.
        # model.transformer.h | List of GPT-2 transformer blocks (decoder layers)
        # if selected index = [10] | 11th transformer block
        # .register_forward_hook(...) | Registers a hook to run during the forward pass of that layer
        # hook method | A function that will be called automatically when the layer runs
        
        if 0 <= self.layer_index < len(self.model.transformer.h):
            self.hook_handle = self.model.transformer.h[self.layer_index].register_forward_hook(self._hook_fn)

    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()

    def get_hidden_activations(self, input_text):
        self.register_hook()

        # Splits text using Byte Pair Encoding (BPE)
        # Looks up each token in the GPT-2 vocabulary
        # Returns a list of integers
        # Now the parameter - return_tensors="pt"
        #   This tells the tokenizer to return the result as a PyTorch tensor, not just a list.
        #   tokenizer.encode("AI is", return_tensors="pt")
        #   Output: tensor([[40, 318, 837]])
        #   This is now in the correct shape to pass to the GPT-2 model:
        #   Shape: (batch_size, sequence_length) → in this case: (1, 3)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        # Forward pass (triggers the hook)
        # PyTorch tracks operations on tensors to compute gradients during training via autograd (automatic differentiation).
        # However, during inference (prediction, evaluation, testing), gradients are not needed. 
        # We are not working on the training now. We are using the modified W, B for prediction purpose
        # Now when model with run with the inputs, our logic will run during the 11th block running
        with torch.no_grad():
            _ = self.model(input_ids=input_ids)

        self.remove_hook()
        # Return the captured activations (already stored during hook execution)
        # return self.captured_activations[f"layer{self.layer_index}"]
        return self.captured_activations.get(f"layer{self.layer_index}", None)
        
    def get_text_results(self, input_text, max_length=50):
        self.register_hook()
        
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        with torch.no_grad():
            # Generate original text
            original_output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            original_text = self.tokenizer.decode(original_output[0], skip_special_tokens=True)
            
            # Generate detoxified text by running forward pass again
            outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            if self.detoxified_activations is not None:
                # Initialize with input_ids
                generated = input_ids.clone()
                
                for _ in range(max_length - input_ids.shape[1]):
                    # Get logits using our modified hidden states
                    outputs = self.model(
                        input_ids=generated,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                    
                    # Replace the hidden state at our target layer
                    modified_hidden_states = list(outputs.hidden_states)
                    modified_hidden_states[self.layer_index + 1] = self.detoxified_activations
                    
                    # Get logits from final layer
                    logits = outputs.logits[:, -1, :]
                    
                    # Apply temperature
                    logits = logits / 0.7
                    
                    # Top-k filtering
                    top_k = 50
                    top_k = min(top_k, logits.size(-1))  # Safety check
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')
                    
                    # Convert to probabilities
                    probs = F.softmax(logits, dim=-1)
                    
                    # Top-p (nucleus) sampling - simplified version
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > 0.9
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Create a mask for indices to remove
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs = probs.clone()
                    probs[0, indices_to_remove] = 0  # Simplified version
                    
                    # If all probabilities are zero, fall back to random sampling
                    if probs.sum() == 0:
                        probs = torch.ones_like(probs) / probs.size(-1)
                    else:
                        probs = probs / probs.sum()  # Renormalize
                    
                    # Sample from the filtered distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Add the sampled token to the sequence
                    generated = torch.cat((generated, next_token), dim=-1)
                    
                    # Update attention mask
                    attention_mask = torch.cat(
                        (attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)),
                        dim=-1
                    )
                    
                    # Stop if we hit EOS token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                
                detox_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            else:
                detox_text = original_text
        
        self.remove_hook()
        return {
            "original_text": original_text,
            "detoxified_text": detox_text,
            "original_activations": self.original_activations,
            "detoxified_activations": self.detoxified_activations
        }

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model