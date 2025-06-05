# GPT-Z

A minimal and educational implementation of a character-level transformer-based language model (GPT) using PyTorch. This project implements the key components of self-attention mechanisms that form the foundation of transformer architectures like GPT.

## Project Structure

```
GPT-Z/
├── Model/
│   └── Model.py          # Self-attention implementations
├── Tokenizer/
│   └── Tokenizer.py      # Tokenization components (in development)
├── BigramModel/
│   └── Birgram.py        # Bigram model implementation (in development)
├── main.py               # Main entry point
├── pyproject.toml        # Project configuration and dependencies
└── README.md            # This file
```

## Model Architecture

The current implementation includes foundational transformer components:

### SelfAttention Class
- Implements basic self-attention mechanism
- Uses Query (W_q), Key (W_k), and Value (W_v) linear transformations
- Applies scaled dot-product attention
- No masking applied - can see all positions

### MaskedSelfAttention Class  
- Implements self-attention with causal masking
- Prevents the model from looking at future tokens
- Essential for autoregressive language modeling
- Uses masking to set future positions to very negative values before softmax

## Features

- ✅ Self-attention mechanism implementation
- ✅ Masked self-attention for causal modeling
- ✅ PyTorch-based for easy debugging and experimentation
- ✅ Manual seed control for reproducibility
- 🚧 Character-level tokenization (in development)
- 🚧 Bigram baseline model (in development)
- 🚧 Full transformer architecture (planned)

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

### Using the Self-Attention Components

```python
import torch
from Model.Model import SelfAttention, MaskedSelfAttention

# Example token encodings (3 tokens, 2-dimensional embeddings)
encodings_matrix = torch.tensor([[1.16, 0.23], [0.57, 1.36], [4.41, -2.16]])

# Set seed for reproducibility
torch.manual_seed(42)

# Regular self-attention - can attend to all positions
self_attention = SelfAttention(d_model=2, row_dim=0, col_dim=1)
output = self_attention(encodings_matrix)
print("Self-attention output:")
print(output)

# Masked self-attention - causal masking for autoregressive modeling
masked_self_attention = MaskedSelfAttention(d_model=2, row_dim=0, col_dim=1)

# Create causal mask (lower triangular matrix)
mask = torch.tril(torch.ones(3, 3))
mask = mask == 0  # True where we want to mask (upper triangle)

output_masked = masked_self_attention(encodings_matrix, mask=mask)
print("Masked self-attention output:")
print(output_masked)
```

## Development Status

This is an educational project currently in early development:

- **✅ Core attention mechanisms**: Basic and masked self-attention are implemented
- **🚧 Tokenization**: Character-level tokenizer implementation in progress
- **🚧 Bigram model**: Simple baseline model for comparison
- **📅 Planned**: Full transformer blocks, positional encodings, training loop

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

## Inspiration

This implementation is inspired by educational resources including:
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) - A minimal GPT implementation
- Andrej Karpathy's neural network tutorials
- "Attention Is All You Need" paper by Vaswani et al.

## License

This project is open source. Please check the original inspirations for their respective licenses.
