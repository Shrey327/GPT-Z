import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
from typing import List, Dict, Tuple, Optional
import json
import os

# Import our existing attention mechanisms
from Model.Model import MaskedSelfAttention, MultiHeadAttention, Attention
# Import BPE tokenizer from Tokenizer directory
from Tokenizer.Tokenizer import BPETokenizer


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class FeedForward(nn.Module):
    """Feed-forward network for transformer block"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer block using fixed attention mechanisms from Model.py"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Use the fixed MultiHeadAttention from Model.py
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=n_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class GPT(nn.Module):
    """GPT (Generative Pre-trained Transformer) model"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, 
                 n_layers: int = 6, d_ff: int = 2048, max_len: int = 1024, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask == 0
    
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create causal mask (1 for allowed positions, 0 for masked)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        
        # Token embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer blocks with causal masking
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            # Calculate cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    def generate(self, tokenizer: BPETokenizer, prompt: str, max_length: int = 100, 
                 temperature: float = 1.0, top_k: Optional[int] = None) -> str:
        """Generate text from prompt"""
        self.eval()
        
        # Encode prompt
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            self.cuda()
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions
                logits, _ = self.forward(generated)
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we hit end token (if we have one)
                if next_token.item() == 0:  # Assuming 0 is a special token
                    break
        
        # Decode generated text
        generated_text = tokenizer.decode(generated[0].tolist())
        return generated_text


def create_sample_corpus() -> List[str]:
    """Create a sample corpus for training"""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "In the beginning was the Word, and the Word was with God.",
        "All that glitters is not gold.",
        "The early bird catches the worm.",
        "Practice makes perfect.",
        "Where there's a will, there's a way.",
        "The pen is mightier than the sword.",
        "Actions speak louder than words.",
        "You can't judge a book by its cover.",
        "The grass is always greener on the other side.",
        "Don't count your chickens before they hatch.",
        "A stitch in time saves nine.",
        "Better late than never.",
        "The proof of the pudding is in the eating.",
        "Rome wasn't built in a day.",
        "Every cloud has a silver lining.",
        "Fortune favors the bold.",
        "The best things in life are free."
    ]


def train_gpt(model: GPT, tokenizer: BPETokenizer, corpus: List[str], 
              epochs: int = 10, batch_size: int = 4, learning_rate: float = 1e-3,
              device: str = 'cpu') -> None:
    """Train GPT model on corpus"""
    
    # Move model to device
    model = model.to(device)
    
    # Prepare training data
    all_text = " ".join(corpus)
    all_tokens = tokenizer.encode(all_text)
    
    # Create training batches
    def create_batches(tokens, batch_size, seq_len):
        batches = []
        for i in range(0, len(tokens) - seq_len, seq_len):
            batch_tokens = tokens[i:i + seq_len + 1]
            input_ids = torch.tensor(batch_tokens[:-1], dtype=torch.long)
            targets = torch.tensor(batch_tokens[1:], dtype=torch.long)
            batches.append((input_ids, targets))
        return batches
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    seq_len = 32  # Context length
    
    print(f"Training GPT model on {len(corpus)} documents...")
    print(f"Total tokens: {len(all_tokens)}")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    
    for epoch in range(epochs):
        model.train()
        batches = create_batches(all_tokens, batch_size, seq_len)
        total_loss = 0
        
        for i, (input_ids, targets) in enumerate(batches):
            input_ids = input_ids.unsqueeze(0).to(device)
            targets = targets.unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            logits, loss = model(input_ids, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(batches)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(batches)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Generate sample text
        model.eval()
        sample_text = model.generate(tokenizer, "The quick", max_length=20, temperature=0.8)
        print(f"Sample generation: {sample_text}")
        print("-" * 50)


def main():
    """Main function to demonstrate GPT implementation"""
    print("GPT-Z: A minimal GPT implementation")
    print("=" * 50)
    
    # Create sample corpus
    corpus = create_sample_corpus()
    print(f"Created corpus with {len(corpus)} documents")
    
    # Initialize and train tokenizer
    tokenizer = BPETokenizer(vocab_size=500)
    tokenizer.train(corpus)
    print(f"Tokenizer trained. Vocabulary size: {len(tokenizer.vocab)}")
    
    # Create GPT model
    model = GPT(
        vocab_size=len(tokenizer.vocab),
        d_model=128,  # Smaller for demo
        n_heads=4,
        n_layers=3,
        d_ff=512,
        max_len=128,
        dropout=0.1
    )
    
    print(f"GPT model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train_gpt(
        model=model,
        tokenizer=tokenizer,
        corpus=corpus,
        epochs=5,
        batch_size=2,
        learning_rate=1e-3,
        device=device
    )
    
    # Final generation examples
    print("\nFinal generation examples:")
    print("=" * 30)
    
    prompts = ["The quick", "To be or", "In the", "All that"]
    
    for prompt in prompts:
        generated = model.generate(tokenizer, prompt, max_length=30, temperature=0.8, top_k=10)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'")
        print()


if __name__ == "__main__":
    main()
