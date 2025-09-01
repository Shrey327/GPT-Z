#!/usr/bin/env python3
"""
Example usage of the GPT-Z implementation
"""

import torch
from app import GPT, create_sample_corpus
from Tokenizer.Tokenizer import BPETokenizer

def quick_demo():
    """Quick demonstration of GPT-Z capabilities"""
    print("GPT-Z Quick Demo")
    print("=" * 30)
    
    # Create a small corpus for quick training
    corpus = [
        "Hello world!",
        "This is a test.",
        "GPT models are amazing.",
        "Machine learning is fun.",
        "Transformers changed everything."
    ]
    
    # Initialize tokenizer
    print("1. Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(corpus)
    print(f"   Vocabulary size: {len(tokenizer.vocab)}")
    
    # Test tokenization
    test_text = "Hello world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"   Test: '{test_text}' -> {encoded} -> '{decoded}'")
    
    # Create a small GPT model
    print("\n2. Creating GPT model...")
    model = GPT(
        vocab_size=len(tokenizer.vocab),
        d_model=64,  # Very small for demo
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_len=32,
        dropout=0.1
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {param_count:,}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    test_tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    logits, loss = model(test_tokens)
    print(f"   Input shape: {test_tokens.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Loss (no targets): {loss}")
    
    # Test with targets
    targets = torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.long)
    logits, loss = model(test_tokens, targets)
    print(f"   Loss (with targets): {loss.item():.4f}")
    
    # Test generation (without training)
    print("\n4. Testing text generation...")
    try:
        generated = model.generate(tokenizer, "Hello", max_length=10, temperature=0.8)
        print(f"   Generated: '{generated}'")
    except Exception as e:
        print(f"   Generation failed (expected without training): {e}")
    
    print("\nDemo completed! Run 'python app.py' for full training example.")

def test_attention_components():
    """Test individual attention components"""
    print("\nTesting Attention Components")
    print("=" * 30)
    
    from Model.Model import SelfAttention, MaskedSelfAttention, MultiHeadAttention
    
    # Test data
    batch_size, seq_len, d_model = 1, 5, 8
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    
    # Test SelfAttention
    self_attn = SelfAttention(d_model=d_model, row_dim=1, col_dim=2)
    output = self_attn(x[0])  # Remove batch dimension for our implementation
    print(f"SelfAttention output shape: {output.shape}")
    
    # Test MaskedSelfAttention
    masked_attn = MaskedSelfAttention(d_model=d_model, row_dim=1, col_dim=2)
    mask = torch.tril(torch.ones(seq_len, seq_len)) == 0
    output = masked_attn(x[0], mask=mask)
    print(f"MaskedSelfAttention output shape: {output.shape}")
    
    # Test MultiHeadAttention
    multi_attn = MultiHeadAttention(d_model=d_model, num_heads=2)
    output = multi_attn(x, x, x)
    print(f"MultiHeadAttention output shape: {output.shape}")

if __name__ == "__main__":
    quick_demo()
    test_attention_components()
