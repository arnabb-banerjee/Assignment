import torch
from HiddenLayerCapture import HiddenLayerCapture

def main():
    # Input text (can be replaced with any prompt)
    text = "I love this beautiful day! But some people are just so stupid and ugly."

    print(f"\nProcessing: '{text}'")

    # Initialize hidden layer capture
    layer_capture = HiddenLayerCapture(
        model_name="gpt2",
        layer_index=6,  # Earlier layer
        dampening_factor=0.5,  # More aggressive dampening
        activation_threshold=0.5  # Lower threshold
    )

    # Get detoxified hidden activations
    #hidden_activations = layer_capture.get_hidden_activations(text)
    results = layer_capture.get_text_results(text)

    #print(f"\nDetoxified hidden activations (shape: {hidden_activations.shape}):")
    #print(hidden_activations)
    
    print("\n=== Original Output ===")
    print(results["original_text"])
    
    print("\n=== Detoxified Output ===")
    print(results["detoxified_text"])
    
    if results["original_activations"] is not None and results["detoxified_activations"] is not None:
        print("\n=== Activation Analysis ===")
        print(f"Original activations shape: {results['original_activations'].shape}")
        print(f"Detoxified activations shape: {results['detoxified_activations'].shape}")
        
        diff = torch.sum(
            results['original_activations'] != results['detoxified_activations']
        ).item()
        print(f"Number of neurons modified: {diff}")
    else:
        print("\nWarning: Failed to capture activations")

if __name__ == "__main__":
    main()
