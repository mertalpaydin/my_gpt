import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from .config import Config


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding size must be divisible by number of heads"
        
        # --- KEY / QUERY / VALUE PROJECTION ---
        # We project the input 'x' into three separate vectors:
        # 1. Query (Q): What am I looking for?
        # 2. Key   (K): What do I contain?
        # 3. Value (V): What information should I pass along if I am found?
        #
        # We do this in a SINGLE matrix multiplication for efficiency.
        # Output size is 3 * n_embd because we stack Q, K, and V together.
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # --- OUTPUT PROJECTION ---
        # After we gather information from all heads, we project it back 
        # to the standard embedding size to mix the results.
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # --- CONFIGURATION STORAGE ---
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # --- THE CAUSAL MASK (The "Lower Triangular" Matrix) ---
        # In a "Causal" model (like GPT), a token can only look at the PAST.
        # It cannot look at the FUTURE.
        # We create a matrix of 1s and 0s. The 'tril' (triangle lower) function
        # ensures the upper right corner is 0 (masked out).
        # We use register_buffer so PyTorch knows this is part of the model state
        # but NOT a learnable parameter (gradient descent won't change it).
        # See @detailed_descriptions/causial_self_attention.ipynb for more details.
        self.register_buffer(
            "causal_mask", 
            torch.tril(torch.ones(
                config.block_size, 
                config.block_size
            )).view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        # Batch size (B), Sequence Length/Time (T), Embedding Dimension (C)
        B, T, C = x.size() 
        
        # --- STEP 1: CALCULATE Q, K, V ---
        # We pass x through the linear layer. 
        # Result shape: (B, T, 3 * C)
        qkv = self.qkv_proj(x)
        
        # We split the result into three separate tensors: Query, Key, Value.
        # Each has shape (B, T, C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # --- STEP 2: SPLIT INTO HEADS ---
        # We want multiple "heads" of attention to look at different things.
        # Example: Head 1 looks for grammar, Head 2 looks for context, etc.
        #
        # Transformation:
        # 1. .view(): Reshape (B, T, C) -> (B, T, num_heads, head_size)
        # 2. .transpose(): Swap T and num_heads -> (B, num_heads, T, head_size)
        # 
        # We swap them so the "Time" and "Head Size" are the last two dimensions.
        # This allows PyTorch to do matrix multiplication efficiently for all heads at once.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # --- STEP 3: ATTENTION SCORE (AFFINITY) ---
        # Equation: (Q @ K_transpose)
        # We calculate how "related" every token is to every other token.
        # shape: (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        # Result is a T x T matrix showing affinities.
        att = (q @ k.transpose(-2, -1)) 
        
        # --- STEP 4: SCALE ---
        # We divide by sqrt(head_size). This prevents the numbers from getting too huge.
        # If numbers are too big, Softmax acts like a "one-hot" vector (too sharp),
        # preventing the network from learning efficiently (vanishing gradients).
        att = att * (1.0 / math.sqrt(k.size(-1)))
        
        # --- STEP 5: MASK (The "Causal" part) ---
        # We look at our `causal_mask`. Wherever the mask is 0 (the future),
        # we replace the attention score with -infinity.
        # Why -inf? Because Softmax(-inf) = 0.
        # This effectively zeroes out any probability of attending to future tokens.
        att = att.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
        
        # --- STEP 6: SOFTMAX ---
        # Convert scores into probabilities (they sum to 1).
        att = F.softmax(att, dim=-1)
        
        # --- STEP 7: AGGREGATE VALUES ---
        # Now we know *how much* to care about each past token (att).
        # We multiply those probabilities by the Values (v).
        # "I am 80% related to token A, so give me 80% of token A's value."
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # --- STEP 8: REASSEMBLE HEADS ---
        # We undo the split.
        # 1. Transpose back: (B, T, nh, hs)
        # 2. Contiguous: Align memory (needed after transpose).
        # 3. View: Merge nh and hs back into C -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # --- STEP 9: OUTPUT PROJECTION ---
        # Mix the information one last time.
        y = self.out_proj(y)
        
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # --- MLP EXPLANATION ---
        # Attention gathers data. MLP "thinks" about it.
        # 1. Expand channel from n_embd -> 4 * n_embd
        self.fc_expand = nn.Linear(config.n_embd, 4 * config.n_embd)
        # 2. Non-linearity (GELU)
        self.act = nn.GELU(approximate='tanh')
        # 3. Project back to n_embd
        self.fc_reduce = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.fc_expand(x)
        x = self.act(x)
        x = self.fc_reduce(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Layer Norm 1 (Pre-Attention)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # Causal Self Attention (The Communicator)
        self.attn = CausalSelfAttention(config)
        # Layer Norm 2 (Pre-MLP)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # MLP (The Thinker)
        self.mlp = MLP(config)
        
    def forward(self, x):
        # Residual Connection 1:
        # x + Attention(Norm(x))
        # Gradient flows through 'x' unchanged ("Superhighway")
        x = x + self.attn(self.ln_1(x))
        
        # Residual Connection 2:
        # x + MLP(Norm(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Token Embeddings
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Position Embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Stack of Blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final Layer Norm
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # Language Model Head (Final Classifier)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # This forward pass isn't fully defined in your snippet yet, 
        # but typically it looks like this:
        B, T = idx.shape
        
        # 1. Get Token Embeddings
        tok_emb = self.transformer.wte(idx) 
        # 2. Get Position Embeddings
        pos_emb = self.transformer.wpe(torch.arange(T, device=idx.device))
        
        # 3. Combine
        x = tok_emb + pos_emb
        
        # 4. Run through Blocks
        for block in self.transformer.h:
            x = block(x)
            
        # 5. Final Norm
        x = self.transformer.ln_f(x)
        
        # 6. Final Logic (prediction)
        logits = self.lm_head(x)

        return logits