import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

from model import GPT, Config

def generate_text(model, start_text, num_tokens=30):
    model.eval()
    # Determine the device the model is on
    device = next(model.parameters()).device
    
    enc = tiktoken.get_encoding("gpt2")
    encoded = enc.encode(start_text)
    # Move input tensor to the same device as the model
    x = torch.tensor(encoded).unsqueeze(0).to(device) # Add batch dimension (1, T)

    print(f"\nGenerating from prompt: '{start_text}'")
    print("-" * 40)
    
    for _ in range(num_tokens):
        with torch.no_grad():
            logits, _ = model(x)
            logits = logits[:, -1, :]
            
            # get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # sample from the distribution (or use argmax for greedy)
            # using top-k or just pure multinomial
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # append to the sequence
            x = torch.cat((x, idx_next), dim=1)

    # Decode
    output_text = enc.decode(x[0].tolist())
    print(output_text)
    print("-" * 40)

def main():
    # 1. SETUP
    config = Config()
    model = GPT(config)

    # 2. LOAD WEIGHTS
    weights_path = "weights/trained_weights.pth"
    print(f"Loading weights from {weights_path}...")
    
    # Ensure we handle CPU/GPU correctly, defaulting to CPU for safety if no GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print("Weights loaded successfully!")

    # 3. PREDICT
    generate_text(model, "Alan Turing was a")

if __name__ == "__main__":
    main()
