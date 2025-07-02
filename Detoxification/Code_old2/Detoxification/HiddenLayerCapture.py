import torch
from DINMDetoxifier import DINMDetoxifier
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper
import torch.nn.functional as F
from SafeEditImplement import SafeEditImplement


class HiddenLayerCapture:
    def __init__(
        self,
        model_name,
        layer_index,
        dampening_factor: float,
        activation_threshold: float,
        std_activations_threshold: float,
        Adjusted_threshold: float,
    ):

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
        self.std_activations_threshold = std_activations_threshold
        self.Adjusted_threshold = Adjusted_threshold
        self.original_activations = None
        self.detoxified_activations = None
        self.original_output = None
        self.detoxified_output = None
        self.hook_handle = None
        self.detoxifier = DINMDetoxifier(
            self.model,
            dampening_factor,
            activation_threshold,
            std_activations_threshold,
            Adjusted_threshold,
        )
        self.safe_edit = SafeEditImplement(model_name)

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
            return detoxified_output  # here to implement detoxifying behavior

    def register_hook(self):
        # Register the hook on layer of selected index [10]
        # This line registers a forward hook on the 11th transformer block in GPT-2’s model architecture.
        # model.transformer.h | List of GPT-2 transformer blocks (decoder layers)
        # if selected index = [10] | 11th transformer block
        # .register_forward_hook(...) | Registers a hook to run during the forward pass of that layer
        # hook method | A function that will be called automatically when the layer runs

        if 0 <= self.layer_index < len(self.model.transformer.h):
            self.hook_handle = self.model.transformer.h[
                self.layer_index
            ].register_forward_hook(self._hook_fn)

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

    def get_text_results(self, input_text, max_length):

        # ===== 1. INPUT SANITIZATION =====
        # Concept: Replace toxic phrases before processing
        # Example: "stupid" => "unwise", "kill" => "stop"
        # Implementation: Uses predefined replacement dictionary
        processed_input = self.safe_edit.apply_safe_replacements(input_text)
        safe_edit_applied = processed_input != input_text  # Track replacements
        
        # ===== 2. TOKENIZATION =====
        # Mathematical: Text => Token IDs ∈ ℤ^{seq_len}
        # Example: "Hello world" => [15496, 995]

        # Requirements:
        # - Padding for batch processing (even single inputs)
        # - Truncation prevents memory overflow (max 512 tokens)
        # return_tensors="pt" | returns tensor and not list, touple, etc. 
        # - Essential for GPU acceleration and model compatibility
        # padding=True | Adds padding tokens to make all sequences equal length
        # - Required for batch processing (even with single inputs) 
        # truncation=True | Cuts off text exceeding model's max length (1024 for GPT-2)
        # - Prevents memory errors with long inputs
        inputs = self.tokenizer(processed_input, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # extracts the tokenized numerical representation of your input text
        # Example: "I want to steal" => [40, 665, 284, 17981] (GPT-2 token IDs)
        # 40 => "I"
        # 665 => " want"
        # 284 => " to"
        # 17981 => " steal" (toxic term)
        # decoded = tokenizer.decode([17981])  => Returns " steal"
        # Device placement (GPU/CPU)
        input_ids = inputs["input_ids"].to(self.model.device)
        
        # 1. Distinguishes real tokens from padding (1 = real, 0 = padding)
        # 2. Ensures the model only processes meaningful content
        # 3. Becomes critical when handling variable-length inputs
        # inputs = tokenizer("I want to steal", return_tensors="pt", padding=True)
        # For batch ["I", "I want"]:
        # input_ids = [[40, 50256, 50256], [40, 665, 50256]] 
        # attention_mask = [[1, 0, 0], [1, 1, 0]]
        attention_mask = inputs["attention_mask"].to(self.model.device)


        # ===== 3. ORIGINAL GENERATION =====
        self.register_hook()  # Prepare for activation capture

        try:
            with torch.no_grad():
                # Generate original text
                original_output = self.model.generate(
                
                    # High Temperature + Low Top_p => Risky outputs possible
                    # Low Temperature + High Top_k => Safe but generic
                    # toxicity = calculate_toxicity(input_text)
                    # temperature = 0.5 + (0.5 * (1 - toxicity))  # Adaptive
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=min(max_length, 512),  # Maximum tokens to generate (including input)
                    do_sample=True,

                    # Randomness Control Parameters
                    temperature=0.7, # Balanced creativity/safety, range=0.5-1.5, Lower = more predictable, safer outputs

                    # Hard stop
                    # logits = model_output.logits[:, -1, :]  # Get last token's predictions
                    # top_k_logits, top_k_indices = torch.topk(logits, k=50)  # Extract top 50
                    # Toxic token at rank #51 | without top_K Could be selected	| with top_k Automatically blocked
                    top_k=50, # Sufficient diversity, Limits sampling pool to top K probable tokens

                    # Soft Block
                    # Only consider the smallest set of tokens whose cumulative probability => 0.9, then randomly sample from them.
                    # Sort Probabilities | For each generation step, the model calculates probabilities for all possible tokens (e.g., [0.4, 0.3, 0.2, 0.1] for 4 tokens)
                    # Cumulative Sum | Sort probabilities descending and calculate cumulative sum: [0.4, 0.7, 0.9, 1.0]
                    # Dynamic Cutoff | Keeps first 3 tokens (0.4 + 0.3 + 0.2 = 0.9), Discards the rest (0.1 is excluded)
                    # Resample | Renormalize the selected tokens' probabilities and sample: [0.4/0.9, 0.3/0.9, 0.2/0.9] => [0.44, 0.33, 0.22]
                    # Why Use top_p for Detoxification?
                    # Adaptive Safety | Automatically excludes low-probability toxic tokens that wouldn't make the cumulative cutoff.
                    # Balanced Creativity | Unlike fixed top_k, it dynamically adjusts the token pool based on confidence
                    # For toxic inputs | Tends to select narrower, safer vocab
                    # For neutral inputs | Allows broader expression
                    # [hurt: 0.5, kill: 0.3, love: 0.15, hug: 0.05]
                    # Keeps [hurt, kill, love] (sum=0.95) => 33% chance of "love"
                    top_p=0.9, # Dynamic vocabulary selection ("nucleus sampling")

                    # repetition_penalty=1.2, # Moderate prevention
                    # A multiplicative penalty applied to already-generated tokens' probabilities
                    # adjusted_score = original_score / repetition_penalty
                    # 1.0 = No penalty (default), 1.1-1.5 = Moderate prevention, >2.0 = Strong suppression
                    # For, repetition_penalty=1.2
                    # A repeated token with original probability 0.6 => becomes 0.6/1.2 = 0.5
                    # Toxic word repetition	| "Die die die" possible | "Die once" more likely 
                    repetition_penalty=1.2, # Moderate prevention

                    # Safety-Critical Parameters
                    bad_words_ids=self.safe_edit.get_banned_word_ids(), # Combine with your SafeEdit replacements

                    # parameter plays a critical role in text generation, especially in your detoxification pipeline.
                    # Without padding | ["Hello", "Hi there"] => Error (different lengths)
                    # With padding | ["Hello [PAD]", "Hi there"] => Works
                    pad_token_id=self.tokenizer.eos_token_id, # Ensures clean generation termination
                    # num_return_sequences=1,
                    output_scores=True,  # For toxicity analysis, When True, returns token probabilities
                    return_dict_in_generate=True
                )

                original_text = self.tokenizer.decode(original_output.sequences[0], skip_special_tokens=True)

                # ===== 4. DETOXIFIED GENERATION =====
                if self.detoxified_activations is not None:
                    # Initialize generation state
                    generated = input_ids.clone()
                    attention_mask = attention_mask.clone()

                    # Generation constraints
                    gen_params = {
                        'temperature': 0.7,      # Controlled randomness
                        'top_k': 50,             # Vocabulary restriction
                        'top_p': 0.9,            # Probability mass cutoff
                        'min_new_tokens': 5,     # Minimum output length
                        'max_new_tokens': max_length - input_ids.shape[1]
                    }

                    # Token-by-token generation
                    for _ in range(gen_params['max_new_tokens']):
                        outputs = self.model(
                            input_ids=generated,
                            attention_mask=attention_mask,
                            output_hidden_states=True
                        )
                        
                        # === NEURON MODIFICATION ===
                        # Concept: Replace toxic patterns in hidden states
                        # Math: h_detox = h_original => mask + corrections
                        modified_hidden_states = list(outputs.hidden_states)
                        modified_hidden_states[self.layer_index + 1] = self.detoxified_activations
                        
                        # Sample next token with safety constraints
                        logits = outputs.logits[:, -1, :]
                        next_token = self._sample_next_token(
                            logits,
                            temperature=gen_params['temperature'],
                            top_k=gen_params['top_k'],
                            top_p=gen_params['top_p']
                        )
                        
                        # Update generation state
                        generated = torch.cat((generated, next_token), dim=-1)
                        attention_mask = torch.cat(
                            (attention_mask, torch.ones(1, 1, 
                            device=attention_mask.device)),
                            dim=-1
                        )
                        
                        # Early stopping conditions
                        if (next_token.item() == self.tokenizer.eos_token_id and 
                            generated.shape[1] >= gen_params['min_new_tokens']):
                            break

                    detox_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                else:
                    detox_text = original_text

        except Exception as e:
            print(f"Generation error: {str(e)}")
            original_text = f"[ERROR] {str(e)}"
            detox_text = f"[ERROR] {str(e)}"
            self.original_activations = None
            self.detoxified_activations = None
        finally:
            self.remove_hook()  # Cleanup hook

        # ===== 5. FINAL SAFETY CHECKS =====
        # Layer 1: Word-level replacement
        detox_text = self.safe_edit.apply_safe_replacements(detox_text)
        
        # Layer 2: Content validation
        if self.safe_edit.contains_unsafe_content(detox_text):
            detox_text = "[SAFETY FILTER] Content blocked"

        return {
            "original_text": original_text,
            "detoxified_text": detox_text,
            "original_activations": self.original_activations,
            "detoxified_activations": self.detoxified_activations,
            "safe_edit_applied": safe_edit_applied,
        }


    def _sample_next_token(self, logits, temperature=0.7, top_k=50, top_p=0.9):
        """
        Safe token sampling with mathematical safeguards
        
        Args:
            logits: Raw model outputs ∈ ℝ^{vocab_size}
            temperature: Sharpness parameter (T)
            top_k: Vocabulary cutoff
            top_p: Probability mass cutoff
            
        Returns:
            Selected token ID
            
        Methodology:
        1. Temperature scaling: p_i = exp(l_i/T) / ∑exp(l_j/T)
        2. Top-k filtering: Keep only k highest probability tokens  
        3. Nucleus sampling: Keep smallest set where ∑p > top_p
        4. Fallback strategies if sampling fails
        """
        try:
            # 1. Temperature scaling
            T = max(temperature, 0.1)
            scaled_logits = logits / T
            
            # 2. Top-k filtering
            if top_k > 0:
                top_k = min(top_k, scaled_logits.size(-1))
                values, _ = torch.topk(scaled_logits, top_k)
                scaled_logits[scaled_logits < values[:, -1:]] = -float('Inf')
            
            # Convert to probabilities
            probs = F.softmax(scaled_logits, dim=-1)
            
            # 3. Top-p (nucleus) sampling with dimension fix
            if 0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Create removal mask - keep first n where sum >= top_p
                sorted_remove = cumulative_probs > top_p
                sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                sorted_remove[..., 0] = 0
                
                # Expand dimensions for proper indexing
                sorted_remove_expanded = sorted_remove.expand_as(sorted_indices)
                
                # Remove low-probability tokens
                remove_indices = sorted_indices[sorted_remove_expanded]
                probs.scatter_(-1, remove_indices.unsqueeze(0), 0.0)
                
                # Renormalize if valid
                if probs.sum() > 0:
                    probs = probs / probs.sum()
            
            # 4. Fallback strategies
            if probs.sum() <= 0:
                probs = torch.ones_like(probs) / probs.size(-1)
            
            return torch.multinomial(probs, num_samples=1)
            
        except Exception as e:
            print(f"Sampling error: {str(e)} - Using greedy fallback")
            return torch.argmax(logits, dim=-1, keepdim=True)
            

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model
