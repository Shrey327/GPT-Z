import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, row_dim, col_dim):
        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.d_model = d_model
        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, token_encodings):
        q=self.W_q(token_encodings)
        k=self.W_k(token_encodings)
        v=self.W_v(token_encodings)

        sims = torch.matmul(q, k.transpose(-2, -1))

        scaled_sims = sims / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))

        attention_percents = F.softmax(scaled_sims, dim=-1)

        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores

encodings_matrix = torch.tensor([[1.16,0.23],[0.57,1.36],[4.41,-2.16]])

torch.manual_seed(42)

selfAttention= SelfAttention(d_model=2, row_dim=0, col_dim=1)

print(selfAttention(encodings_matrix))

class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model,row_dim,col_dim):
        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.d_model = d_model
        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, token_encodings, mask=None):

        q=self.W_q(token_encodings)
        k=self.W_k(token_encodings) 
        v=self.W_v(token_encodings)
        
        sims= torch.matmul(q, k.transpose(-2, -1))

        scaled_sims = sims / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask == 0, value=-1e20)

        attention_percents = F.softmax(scaled_sims, dim=-1)

        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores

encodings_matrix = torch.tensor([[1.16,0.23],[0.57,1.36],[4.41,-2.16]])
torch.manual_seed(42)
maskedselfAttention= MaskedSelfAttention(d_model=2, row_dim=0, col_dim=1)
mask = torch.tril(torch.ones(3, 3))
mask = mask == 0
print(mask)
print(maskedselfAttention(encodings_matrix, mask=mask))

class Attention(nn.Module):

    def __init__(self, d_model=2, row_dim=0, col_dim=1):
        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.d_model = d_model
        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        # Fix transpose syntax - use -2, -1 for last two dimensions
        sims = torch.matmul(q, k.transpose(-2, -1))

        # Fix scaling - use the last dimension of k
        scaled_sims = sims / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask == 0, value=-1e20)

        attention_percents = F.softmax(scaled_sims, dim=-1)

        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores


encodings_for_q = torch.tensor([[1.16, 0.23],
                                [0.57, 1.36],
                                [4.41, -2.16]])

encodings_for_k = torch.tensor([[1.16, 0.23],
                                [0.57, 1.36],
                                [4.41, -2.16]])

encodings_for_v = torch.tensor([[1.16, 0.23],
                                [0.57, 1.36],
                                [4.41, -2.16]])

## set the seed for the random number generator
torch.manual_seed(42)

## create an attention object
attention = Attention(d_model=2,
                      row_dim=0,
                      col_dim=1)

## calculate encoder-decoder attention
print(attention(encodings_for_q, encodings_for_k, encodings_for_v))

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.size()
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear transformation
        output = self.W_o(attention_output)
        return self.dropout(output)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output


torch.manual_seed(42)

# Test with single head
multiHeadAttention = MultiHeadAttention(d_model=2, num_heads=1)
# Add batch dimension for testing
test_input = encodings_for_q.unsqueeze(0)  # Shape: (1, 3, 2)
print("Single head attention:")
print(multiHeadAttention(test_input, test_input, test_input))

torch.manual_seed(42)

# Test with multiple heads
multiHeadAttention = MultiHeadAttention(d_model=4, num_heads=2)
# Create test input with d_model=4
test_input_4d = torch.randn(1, 3, 4)
print("Multi-head attention:")
print(multiHeadAttention(test_input_4d, test_input_4d, test_input_4d))







       