# GPT-Z

A minimal and educational implementation of transformer-based language model components using PyTorch. This project implements key components including self-attention mechanisms, multi-head attention, and BPE tokenization that form the foundation of transformer architectures like GPT.

## Project Structure

```
GPT-Z/
├── Model/
│   └── Model.py          # Attention implementations (Self, Masked, Multi-Head)
├── Tokenizer/
│   └── Tokenizer.py      # BPE (Byte Pair Encoding) tokenizer implementation
├── BigramModel/
│   └── Birgram.py        # Placeholder for bigram model (not implemented)
├── main.py               # Main entry point (basic)
├── pyproject.toml        # Project configuration and dependencies
└── README.md            # This file
```

## Model Architecture

The implementation includes comprehensive transformer components:

### SelfAttention Class
- Implements basic self-attention mechanism
- Uses Query (W_q), Key (W_k), and Value (W_v) linear transformations
- Applies scaled dot-product attention
- No masking applied - can attend to all positions

### MaskedSelfAttention Class  
- Implements self-attention with causal masking
- Prevents the model from looking at future tokens during training
- Essential for autoregressive language modeling
- Uses masking to set future positions to very negative values before softmax

### Attention Class (Encoder-Decoder)
- General attention mechanism supporting different encodings for Q, K, and V
- Supports optional masking for flexible attention patterns
- Can be used for encoder-decoder architectures

### MultiHeadAttention Class
- Implements multi-head attention mechanism
- Configurable number of attention heads
- Concatenates outputs from multiple attention heads
- Foundation for transformer block implementations

## Tokenization

### BPE Tokenizer Implementation
- **✅ Complete BPE (Byte Pair Encoding) implementation**
- Character-level vocabulary initialization
- Iterative pair merging based on frequency
- Configurable number of merge operations
- Handles end-of-word tokens (`</w>`)
- Vocabulary management and deduplication

## Features

- ✅ **Self-attention mechanism** - Basic attention without masking
- ✅ **Masked self-attention** - Causal masking for autoregressive modeling  
- ✅ **General attention** - Encoder-decoder style attention
- ✅ **Multi-head attention** - Multiple parallel attention mechanisms
- ✅ **BPE Tokenizer** - Complete Byte Pair Encoding implementation
- ✅ **PyTorch-based** - Easy debugging and experimentation
- ✅ **Manual seed control** - Reproducible results
- ❌ **Bigram baseline model** - Not implemented (placeholder only)
- 🚧 **Full transformer architecture** - Planned (attention mechanisms ready)

## Requirements

- Python >= 3.13
- PyTorch >= 2.7.0

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. 

### Option 1: Using uv (Recommended)
```bash
# Clone the repository
git clone <your-repo-url>
cd GPT-Z

# Install dependencies with uv
uv sync
```

### Option 2: Using pip
```bash
# Clone the repository
git clone <your-repo-url>
cd GPT-Z

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install torch>=2.7.0
```

## Usage

### Running the Main Application
```bash
python main.py
```

### Using the Attention Mechanisms

```python
import torch
from Model.Model import SelfAttention, MaskedSelfAttention, Attention, MultiHeadAttention

# Example token encodings (3 tokens, 2-dimensional embeddings)
encodings_matrix = torch.tensor([[1.16, 0.23], [0.57, 1.36], [4.41, -2.16]])

# Set seed for reproducibility
torch.manual_seed(42)

# 1. Regular self-attention - can attend to all positions
self_attention = SelfAttention(d_model=2, row_dim=0, col_dim=1)
output = self_attention(encodings_matrix)
print("Self-attention output:", output)

# 2. Masked self-attention - causal masking for autoregressive modeling
masked_self_attention = MaskedSelfAttention(d_model=2, row_dim=0, col_dim=1)
mask = torch.tril(torch.ones(3, 3)) == 0  # Upper triangular mask
output_masked = masked_self_attention(encodings_matrix, mask=mask)
print("Masked self-attention output:", output_masked)

# 3. General attention (encoder-decoder style)
attention = Attention(d_model=2, row_dim=0, col_dim=1)
output_general = attention(encodings_matrix, encodings_matrix, encodings_matrix)
print("General attention output:", output_general)

# 4. Multi-head attention
multi_head = MultiHeadAttention(d_model=2, row_dim=0, col_dim=1, num_heads=2)
output_multi = multi_head(encodings_matrix, encodings_matrix, encodings_matrix)
print("Multi-head attention output:", output_multi)
```

### Using the BPE Tokenizer

```python
# The tokenizer implementation is in Tokenizer/Tokenizer.py
# It demonstrates BPE training on a small corpus:

# Example corpus
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# The implementation shows:
# 1. Character-level vocabulary initialization
# 2. Word splitting with end-of-word tokens
# 3. Pair frequency counting
# 4. Iterative merging of most frequent pairs
# 5. Vocabulary expansion with merged tokens

# Run the tokenizer:
python Tokenizer/Tokenizer.py
```

## Development Status

This is an educational project with the following implementation status:

- **✅ Attention Mechanisms**: Complete implementation of self-attention, masked self-attention, general attention, and multi-head attention
- **✅ BPE Tokenization**: Full working Byte Pair Encoding tokenizer with training and vocabulary management
- **❌ Bigram Model**: Not implemented (placeholder file only)
- **🚧 Transformer Blocks**: Individual components ready, need integration into full transformer architecture
- **📅 Planned**: 
  - Complete transformer block implementation
  - Positional encodings
  - Training loop
  - Text generation functionality
  - Model evaluation utilities

## Code Structure Details

### Attention Implementation Features
- **Configurable dimensions**: Flexible row/column dimensions for matrix operations
- **Optional masking**: Support for causal and custom masking patterns
- **Scaled dot-product**: Proper scaling by √d_k for numerical stability
- **Multiple heads**: Parallel attention computation with concatenation

### BPE Tokenizer Features
- **Frequency-based merging**: Merges most frequent adjacent character pairs
- **Vocabulary tracking**: Maintains growing vocabulary with new tokens
- **End-of-word handling**: Special `</w>` tokens for word boundaries
- **Configurable merges**: Adjustable number of BPE merge operations
- **Debugging output**: Detailed logging of merge process and statistics

## Contributing

1. Fork the repository
2. Create your feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and test them
4. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
5. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
6. Open a Pull Request

## Learning Resources

This implementation serves as an educational tool for understanding:
- **Attention mechanisms** in transformer architectures
- **BPE tokenization** used in modern language models
- **PyTorch implementation** patterns for deep learning components
- **Modular design** for ML system components

## Inspiration

This implementation is inspired by educational resources including:
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) - A minimal GPT implementation
- Andrej Karpathy's neural network tutorials
- "Attention Is All You Need" paper by Vaswani et al.
- Hugging Face tokenizers documentation

## License

This project is open source. Please check the original inspirations for their respective licenses.
